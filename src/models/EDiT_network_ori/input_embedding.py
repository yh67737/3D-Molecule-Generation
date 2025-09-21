import torch
import torch.nn as nn
import math
from torch_scatter import scatter

from e3nn import o3

from .radial_func import RadialProfile
from .tensor_product_rescale import LinearRS, FullyConnectedTensorProductRescale, TensorProductRescale, sort_irreps_even_first
from .tensor_product_rescale import DepthwiseTensorProduct
_RESCALE = True
_USE_BIAS = True

@torch.jit.script
def gaussian(x, mean, std):
    """高斯概率密度函数"""
    pi = 3.14159
    a = (2*pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


# From Graphormer
class GaussianRadialBasisLayer(torch.nn.Module):
    def __init__(self, num_basis, cutoff):
        super().__init__()
        self.num_basis = num_basis
        self.cutoff = cutoff + 0.0  # 确保是浮点数

        self.mean = torch.nn.Parameter(torch.zeros(1, self.num_basis))
        self.std = torch.nn.Parameter(torch.zeros(1, self.num_basis))

        self.weight = torch.nn.Parameter(torch.ones(1, 1))
        self.bias = torch.nn.Parameter(torch.zeros(1, 1))

        self.std_init_max = 1.0
        self.std_init_min = 1.0 / self.num_basis
        self.mean_init_max = 1.0
        self.mean_init_min = 0

        torch.nn.init.uniform_(self.mean, self.mean_init_min, self.mean_init_max)
        torch.nn.init.uniform_(self.std, self.std_init_min, self.std_init_max)
        torch.nn.init.constant_(self.weight, 1)
        torch.nn.init.constant_(self.bias, 0)

    def forward(self, dist):
        x = dist / self.cutoff
        x = x.unsqueeze(-1)
        x = self.weight * x + self.bias
        x = x.expand(-1, self.num_basis)

        mean = self.mean
        std = self.std.abs() + 1e-5

        x = gaussian(x, mean, std)
        return x

    def extra_repr(self):
        return 'mean_init_max={}, mean_init_min={}, std_init_max={}, std_init_min={}'.format(
            self.mean_init_max, self.mean_init_min, self.std_init_max, self.std_init_min)

    def extra_repr(self):
        # 辅助函数，用于打印模块信息
        return 'mean_init_max={}, mean_init_min={}, std_init_max={}, std_init_min={}'.format(
            self.mean_init_max, self.mean_init_min, self.std_init_max, self.std_init_min)



_MAX_ATOM_TYPE = 6 # 5种原子类型+1种吸收态 

class NodeEmbeddingNetwork(nn.Module):
    """
    修改自Equiformer的NodeEmbeddingNetwork，额外处理环信息
    
    输入:
        - 原子类型的one-hot编码
        - 环信息的二进制标志
    输出:
        - 初始纯标量(L=0)的节点irreps特征
    """
    
    def __init__(self, irreps_node_embedding: str = '128x0e+64x1e+32x2e', num_atom_types: int = _MAX_ATOM_TYPE, hidden_dim: int = 32, bias: bool = True):
        """
        Args:
            irreps_node_embedding (str): 输出的irreps字符串. 
            num_atom_types (int): 原子类型数量 (one-hot向量长度).
            hidden_dim (int): 首先将原子类型和条件标志位投射到hidden_dim相加.
            bias (bool): 线性层是否使用偏置.
        """
        super().__init__()
        
        # 保存输出irreps的定义
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        
        # 成环信息嵌入层
        self.ring_embedding = nn.Embedding(num_embeddings=2, embedding_dim=hidden_dim) 
        # 原子类型嵌入层
        self.atom_type_embedding = nn.Linear(num_atom_types, hidden_dim, bias=False)
        
        # 定义等变线性层
        self.num_input_features = hidden_dim
        input_irreps = o3.Irreps(f'{self.num_input_features}x0e')
        self.feature_lin = LinearRS(input_irreps, self.irreps_node_embedding, bias=bias) 
        
        # 线性层权重初始化
        self.feature_lin.tp.weight.data.mul_(self.num_input_features ** 0.5)
        
        
    def forward(self, atom_type_onehot: torch.Tensor, ring_info: torch.Tensor) -> tuple:
        """
        Args:
            atom_type_onehot (Tensor): one-hot编码的原子类型, shape [num_nodes, num_atom_types].
            ring_info (Tensor): 环信息, shape:[num_nodes,1]，取值为0或1.

        Returns:
            node_embedding: 融合后的初始节点特征.
        """
        # 确保ring_info形状正确[num_nodes, 1]
        if ring_info.ndim == 1:
            ring_info = ring_info.unsqueeze(-1)
            
        # 将成环标志(0/1)通过Embedding层映射
        ring_embeds = self.ring_embedding(ring_info.long().squeeze(-1))
        # 将原子类型通过Embedding层映射
        atom_type_embeds = self.atom_type_embedding(atom_type_onehot.float())
        combined_attr = ring_embeds + atom_type_embeds

        node_embedding = self.feature_lin(combined_attr)
        
        # 返回嵌入结果
        return node_embedding


# 定义化学键类型数量
_DEFAULT_NUM_BOND_TYPES = 5 # 无键、单键、双键、三键、芳香键


class EdgeEmbeddingNetwork(nn.Module):
    def __init__(self,
                 irreps_sh: str = '1x0e+1x1o+1x2e',
                 max_radius: float = 5.0,
                 number_of_basis: int = 128,
                 num_bond_types: int = _DEFAULT_NUM_BOND_TYPES,
                 bond_embedding_dim: int = 32,
                 irreps_out_fused: str = '128x0e+64x1o+32x2e'):
        super().__init__()
        self.max_radius = max_radius
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.irreps_out_fused = o3.Irreps(irreps_out_fused, )

        self.rbf = GaussianRadialBasisLayer(number_of_basis, cutoff=self.max_radius)
        self.bond_embedding = nn.Linear(num_bond_types, bond_embedding_dim, bias=False)

        scalar_dim_combined = number_of_basis + bond_embedding_dim
        irreps_scalar_combined = o3.Irreps(f'{scalar_dim_combined}x0e')

        self.fusion_tp = FullyConnectedTensorProductRescale(
            irreps_in1=self.irreps_sh,
            irreps_in2=irreps_scalar_combined,
            irreps_out=self.irreps_out_fused
        )

    def forward(self,
                pos: torch.Tensor,
                bond_type_onehot: torch.Tensor,
                edge_index: torch.Tensor) -> dict:
        edge_src, edge_dst = edge_index[0], edge_index[1]

        edge_vec = pos.index_select(0, edge_src) - pos.index_select(0, edge_dst)

        edge_attr = o3.spherical_harmonics(
            l=self.irreps_sh,
            x=edge_vec,
            normalize=True,
            normalization='component'
        )

        edge_length = edge_vec.norm(dim=1)
        edge_rbf = self.rbf(edge_length)

        bond_embeds = self.bond_embedding(bond_type_onehot.float())

        edge_scalars_combined = torch.cat([edge_rbf, bond_embeds], dim=-1)

        fused_edge_feature = self.fusion_tp(edge_attr, edge_scalars_combined)

        return {
            "fused_edge_feature": fused_edge_feature,
            "edge_attr_base": edge_attr,
            "edge_scalars_base": edge_rbf,
            "edge_src": edge_src,
            "edge_dst": edge_dst,
            "edge_vec": edge_vec
        }


class ScaledScatter(torch.nn.Module):
    def __init__(self, avg_aggregate_num):
        super().__init__()
        self.avg_aggregate_num = avg_aggregate_num + 0.0


    def forward(self, x, index, **kwargs):
        out = scatter(x, index, **kwargs)
        out = out.div(self.avg_aggregate_num ** 0.5)
        return out
    
    
    def extra_repr(self):
        return 'avg_aggregate_num={}'.format(self.avg_aggregate_num)


class EdgeDegreeEmbeddingNetwork(torch.nn.Module):
    def __init__(self, irreps_node_embedding, irreps_edge_attr, avg_aggregate_num, fc_neurons=[64, 64]): ### avg_aggregate_num
        super().__init__()
        self.exp = LinearRS(o3.Irreps('1x0e'), irreps_node_embedding, bias=_USE_BIAS, rescale=_RESCALE)
        self.dw = DepthwiseTensorProduct(
            irreps_node_embedding,
            irreps_edge_attr,
            irreps_node_embedding,
            internal_weights=False,
            bias=False
        )
        self.rad = RadialProfile(fc_neurons + [self.dw.tp.weight_numel])

        self.proj = LinearRS(self.dw.irreps_out.simplify(), irreps_node_embedding)
        self.scale_scatter = ScaledScatter(avg_aggregate_num)
        
    
    def forward(self, node_input, edge_attr, edge_scalars, edge_src, edge_dst, batch):
        node_features = torch.ones_like(node_input.narrow(1, 0, 1))
        node_features = self.exp(node_features)
        weight = self.rad(edge_scalars)
        edge_features = self.dw(node_features[edge_src], edge_attr, weight)
        edge_features = self.proj(edge_features)
        node_features = self.scale_scatter(edge_features, edge_dst, dim=0, 
            dim_size=node_features.shape[0])
        return node_features


class InputEmbeddingLayer(nn.Module):
    """
    一个完整的输入嵌入模块。
    它接收一个PyG Data对象，调用所有子嵌入网络，
    并返回一个包含所有E-DiT Block所需输入的字典。
    """

    def __init__(self,
                 # NodeEmbeddingNetwork 参数
                 irreps_node_embedding: str,
                 num_atom_types: int,
                 node_embedding_hidden_dim: int,
                 # EdgeEmbeddingNetwork 参数
                 irreps_sh: str,
                 max_radius: float,
                 num_rbf: int,
                 num_bond_types: int,
                 bond_embedding_dim: int,
                 irreps_edge_fused: str,
                 # EdgeDegreeEmbeddingNetwork 参数
                 avg_degree: float):
        super().__init__()

        # Irreps 定义
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.irreps_edge_fused = o3.Irreps(irreps_edge_fused)

        # 实例化三个核心子模块
        self.node_embedding_net = NodeEmbeddingNetwork(
            irreps_node_embedding=self.irreps_node_embedding,
            num_atom_types=num_atom_types,
            hidden_dim=node_embedding_hidden_dim
        )
        self.edge_embedding_net = EdgeEmbeddingNetwork(
            irreps_sh=self.irreps_sh,
            max_radius=max_radius,
            number_of_basis=num_rbf,
            num_bond_types=num_bond_types,
            bond_embedding_dim=bond_embedding_dim,
            irreps_out_fused=self.irreps_edge_fused
        )
        self.edge_degree_net = EdgeDegreeEmbeddingNetwork(
            irreps_node_embedding=self.irreps_node_embedding,
            irreps_edge_attr=self.irreps_sh,
            fc_neurons=[num_rbf, num_rbf],
            avg_aggregate_num=avg_degree
        )

    def forward(self, data) -> dict:
        """
        接收一个PyG Data/Batch对象并进行完整的嵌入操作。

        Args:
            data: 一个包含x, pos, edge_index, edge_attr, pring_out等属性的Data对象。

        Returns:
            一个包含所有E-DiT Block所需输入的字典。
        """
        # 处理边特征
        edge_info = self.edge_embedding_net(
            pos=data.pos,
            bond_type_onehot=data.edge_attr,
            edge_index=data.edge_index
        )

        # 处理节点初始特征
        initial_node_embedding = self.node_embedding_net(
            atom_type_onehot=data.x,
            ring_info=data.pring_out
        )

        # 为节点注入几何信息
        edge_degree_supplement = self.edge_degree_net(
            node_input=initial_node_embedding,
            edge_attr=edge_info['edge_attr_base'],
            edge_scalars=edge_info['edge_scalars_base'],
            edge_src=edge_info['edge_src'],
            edge_dst=edge_info['edge_dst'],
            batch=data.batch
        )

        # 最终节点特征融合
        final_node_features = initial_node_embedding + edge_degree_supplement

        # 准备E-DiT Block的全部输入
        node_attr = data.x

        return {
            "node_input": final_node_features,
            "node_attr": node_attr,
            "edge_src": edge_info['edge_src'],
            "edge_dst": edge_info['edge_dst'],
            "edge_index": data.edge_index,
            "edge_attr": edge_info['edge_attr_base'],
            "edge_scalars": edge_info['edge_scalars_base'],
            "edge_input": edge_info['fused_edge_feature'],
            "edge_attr_type": data.edge_attr,
            "batch": data.batch
        }