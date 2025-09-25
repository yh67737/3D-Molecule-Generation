import torch
import torch.nn as nn
from e3nn import o3
from torch_scatter import scatter_sum
from .layer_norm import AdaEquiLayerNorm
from .tensor_product_rescale import LinearRS, TensorProductRescale, sort_irreps_even_first
from .radial_func import RadialProfile
from .input_embedding import GaussianRadialBasisLayer

_RESCALE = True

def DepthwiseTensorProduct(irreps_node_input, irreps_edge_attr, irreps_node_output, 
    internal_weights=False, bias=True):
    '''
        The irreps of output is pre-determined. 
        `irreps_node_output` is used to get certain types of vectors.
    '''
    irreps_output = []
    instructions = []
    
    for i, (mul, ir_in) in enumerate(irreps_node_input):
        for j, (_, ir_edge) in enumerate(irreps_edge_attr):
            for ir_out in ir_in * ir_edge:
                if ir_out in irreps_node_output or ir_out == o3.Irrep(0, 1):
                    k = len(irreps_output)
                    irreps_output.append((mul, ir_out))
                    instructions.append((i, j, k, 'uvu', True))
        
    irreps_output = o3.Irreps(irreps_output)
    irreps_output, p, _ = sort_irreps_even_first(irreps_output) #irreps_output.sort()
    instructions = [(i_1, i_2, p[i_out], mode, train)
        for i_1, i_2, i_out, mode, train in instructions]
    tp = TensorProductRescale(irreps_node_input, irreps_edge_attr,
            irreps_output, instructions,
            internal_weights=internal_weights,
            shared_weights=internal_weights,
            bias=bias, rescale=_RESCALE)
    return tp

#等变的坐标更新模块，模仿Moldiff的BondFFN+PosUpdate，将BondFFN的代码融入PosUpdate结构中
class EquivariantPosUpdate(nn.Module):
    
    def __init__(self, 
                 irreps_node_in: str, 
                 irreps_edge_in: str, 
                 time_embedding_dim: int,
                 number_of_basis: int,
                 max_radius: float, 
                 fc_neurons: list,
                 hidden_irreps: str = '128x0e+64x1e+32x2e'):
        """
        Args:
            irreps_node_in (str): 输入的节点特征的不可约表示 (irreps).
            irreps_edge_in (str): 输入的边特征的不可约表示 (irreps).
            time_embedding_dim (int): 时间步嵌入的维度.
            hidden_irreps (str): 模块内部用于特征融合的中间irreps维度.
        """
        super().__init__()
        
        # 将输入的字符串转换为Irreps对象
        irreps_node_in = o3.Irreps(irreps_node_in)
        irreps_edge_in = o3.Irreps(irreps_edge_in)
        hidden_irreps = o3.Irreps(hidden_irreps)
        
        # 定义RBF层和径向函数网络的输入维度
        self.rbf = GaussianRadialBasisLayer(number_of_basis, cutoff=max_radius)
        radial_hidden_dim = [number_of_basis] + fc_neurons
        
        # 准备节点特征
        # 定义两个独立的线性层，用于分别变换源节点和目标节点特征
        self.src_node_transform = LinearRS(irreps_node_in, irreps_edge_in)
        self.dst_node_transform = LinearRS(irreps_node_in, irreps_edge_in)
        
        # 预测每条边的标量权重
        
        # a.融合两端节点特征：使用FCTP
        self.node_fusion_tp = DepthwiseTensorProduct(
            irreps_node_input=irreps_edge_in, 
            irreps_edge_attr=irreps_edge_in, 
            irreps_node_output=irreps_edge_in
        )
        # node_fusion_tp配套的径向网络
        self.node_fusion_rad = RadialProfile(radial_hidden_dim + [self.node_fusion_tp.tp.weight_numel])

        # b.将融合后的节点、初始边特征映射到隐藏维度
        self.node_transform = LinearRS(irreps_edge_in, hidden_irreps)
        self.edge_transform = LinearRS(irreps_edge_in, hidden_irreps)
        
        # c.融合节点和边特征
        self.edge_fusion_tp = DepthwiseTensorProduct(
            irreps_node_input=hidden_irreps,
            irreps_edge_attr=hidden_irreps,
            irreps_node_output=hidden_irreps
        )
        # edge_fusion_tp配套的径向网络
        self.edge_fusion_rad = RadialProfile(radial_hidden_dim + [self.edge_fusion_tp.tp.weight_numel])

        # d.层归一化引入时间特征 (遵循Pre-Norm架构)
        self.norm = AdaEquiLayerNorm(
            irreps=hidden_irreps, 
            time_embedding_dim=time_embedding_dim
        )
        
        # e.将归一化后的特征映射为最终的标量权重，输出为'1x0e'
        self.scalar_predictor = nn.Sequential(
            LinearRS(hidden_irreps, o3.Irreps('32x0e')), # 先映射到一个中间标量维度
            nn.SiLU(),
            LinearRS(o3.Irreps('32x0e'), o3.Irreps('1x0e'))  # 再映射到最终的1x0e
        )

    def forward(self, 
                h_node: torch.Tensor, 
                h_edge: torch.Tensor, 
                pos: torch.Tensor, 
                edge_index: torch.Tensor, 
                relative_vec,
                distance,
                t: torch.Tensor, 
                batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_node (Tensor): 节点 irreps 特征, shape [num_nodes, irreps_node_in.dim].
            h_edge (Tensor): 边 irreps 特征, shape [num_edges, irreps_edge_in.dim].
            pos (Tensor): 节点坐标, shape [num_nodes, 3].
            edge_index (Tensor): 边索引, shape [2, num_edges].
            t (Tensor): 扩散时间步 (每个图一个), shape [num_graphs].
            batch (Tensor): 节点所属的图索引, shape [num_nodes].

        Returns:
            delta_pos (Tensor): 预测的坐标更新量, shape [num_nodes, 3].
        """
        
        # 从边索引中分离出源节点(左)和目标节点(右)
        edge_src, edge_dst = edge_index

        # 通过RBF和RadialProfile生成DTP权重
        edge_scalars = self.rbf(distance)
        weight_node_fusion = self.node_fusion_rad(edge_scalars)
        weight_edge_fusion = self.edge_fusion_rad(edge_scalars)

        # 准备节点特征
        src_feat = self.src_node_transform(h_node[edge_src])
        dst_feat = self.dst_node_transform(h_node[edge_dst])

        # 预测每条边的标量权重
        fused_nodes = self.node_fusion_tp(src_feat, dst_feat, weight=weight_node_fusion)
        fused_nodes_trans = self.node_transform(fused_nodes)
        edge_trans = self.edge_transform(h_edge)
        fused_all = self.edge_fusion_tp(fused_nodes_trans, edge_trans, weight=weight_edge_fusion)

        batch_edge = batch[edge_src] # 为边特征分配对应的批次索引
        normalized_feat = self.norm(fused_all, t, batch_edge)   
        scalar_weight = self.scalar_predictor(normalized_feat) # shape: [num_edges, 1]

        # 计算力向量 (force_edge)
        distance_2d = distance.unsqueeze(-1)
        force_edge = scalar_weight * (relative_vec / distance_2d) / (distance_2d + 1.0)

        # 聚合力到节点上
        delta_pos = scatter_sum(force_edge, edge_src, dim=0, dim_size=h_node.shape[0])

        return delta_pos