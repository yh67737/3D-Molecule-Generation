import torch
import torch.nn as nn
from e3nn import o3
from e3nn.o3 import Irreps
from torch_scatter import scatter
import torch.nn.functional as F
from src.models.EDiT_network.layer_norm import AdaEquiLayerNorm
from src.models.EDiT_network.tensor_product_rescale import LinearRS, FullyConnectedTensorProductRescale

class StableNormActivation(nn.Module):
    """
    一个数值稳定的、自定义实现的 NormActivation。
    它接收一个 Irreps 张量，并对其应用非线性激活，同时保持等变性。
    """

    def __init__(self, irreps_in: Irreps, activation=F.silu):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = self.irreps_in
        self.activation = activation

        # 将输入的 Irreps 分解为标量部分和非标量部分
        self.scalar_irreps = o3.Irreps([(mul, ir) for mul, ir in self.irreps_in if ir.l == 0])
        self.non_scalar_irreps = o3.Irreps([(mul, ir) for mul, ir in self.irreps_in if ir.l > 0])

        # 记录每个非标量部分的维度信息，方便 forward 中使用
        self.non_scalar_dims = []
        if self.non_scalar_irreps.dim > 0:
            start_dim = self.scalar_irreps.dim
            for mul, ir in self.non_scalar_irreps:
                self.non_scalar_dims.append(
                    (start_dim, start_dim + mul * ir.dim, mul, ir.dim)
                )
                start_dim += mul * ir.dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 创建一个列表来收集所有处理过的特征部分
        output_parts = []

        # 1. 处理标量部分 (l=0)
        scalar_dim = self.scalar_irreps.dim
        if scalar_dim > 0:
            scalars = x[:, :scalar_dim]
            activated_scalars = self.activation(scalars)
            output_parts.append(activated_scalars)

        # 2. 处理非标量部分 (l>0)
        for start, end, mul, dim in self.non_scalar_dims:
            non_scalars = x[:, start:end].clone().view(-1, mul, dim)
            norm = torch.linalg.norm(non_scalars, dim=-1, keepdim=True)
            activated_norm = self.activation(norm)
            
            epsilon = 1e-8
            scaling_factor = activated_norm / (norm + epsilon)
            
            scaled_non_scalars = (non_scalars * scaling_factor).view(-1, mul * dim)
            output_parts.append(scaled_non_scalars)

        # 3. 使用 torch.cat 将所有部分拼接成一个新张量
        # 这是非原地操作，对 autograd 是安全的
        if len(output_parts) == 0:
            return torch.zeros_like(x)
        else:
            return torch.cat(output_parts, dim=-1)
        


class EquivariantBondFFN(nn.Module):
    """
    BondFFN的等变实现.
    """

    def __init__(self, irreps_bond: Irreps, irreps_node: Irreps, irreps_sh: Irreps,
                 irreps_message: Irreps, irreps_inter: Irreps = None):
        """
        Args:
            irreps_bond (Irreps): 边特征的Irreps.
            irreps_node (Irreps): 节点特征的Irreps.
            irreps_sh (Irreps): 球谐特征的Irreps.
            irreps_message (Irreps): 最终输出消息的Irreps.
            irreps_inter (Irreps, optional): 中间交互特征的Irreps. 如果为 None，则默认为 irreps_message.
        """
        super().__init__()
        self.irreps_bond = irreps_bond
        self.irreps_node = irreps_node
        self.irreps_sh = irreps_sh
        self.irreps_message = irreps_message
        if irreps_inter is None:
            irreps_inter = irreps_message

        # 初始节点和边特征映射
        self.bond_linear = LinearRS(self.irreps_bond, irreps_inter)
        self.node_linear = LinearRS(self.irreps_node, irreps_inter)

        irreps_concat_in = (irreps_inter + irreps_inter + self.irreps_sh).simplify()

        # 2. 定义一个融合网络，替代原来所有的 tensor_product 和 inter_module
        #    它接收拼接后的特征，输出最终的消息维度
        self.fusion_module = nn.Sequential(
            LinearRS(irreps_concat_in, self.irreps_message),
            StableNormActivation(self.irreps_message, F.silu),
            LinearRS(self.irreps_message, self.irreps_message)
        )

        #门控路径
        gate_mlp_irreps_in = (self.irreps_bond + self.irreps_node + self.irreps_sh).simplify()
        irreps_gate_hidden = gate_mlp_irreps_in

        self.gate_mlp = nn.Sequential(
            LinearRS(gate_mlp_irreps_in, irreps_gate_hidden),
            StableNormActivation(irreps_gate_hidden, F.silu),
            LinearRS(irreps_gate_hidden, o3.Irreps("1x0e")))

#     def forward(self, bond_feat_input, node_feat_input, sh_feat):
#         bond_feat_proj = self.bond_linear(bond_feat_input)
#         node_feat_proj = self.node_linear(node_feat_input)
#         inter_feat_1 = self.tensor_product_1(bond_feat_proj, node_feat_proj)
#         inter_feat_2 = self.tensor_product_2(inter_feat_1.clone(), sh_feat.clone())
#         inter_feat = self.inter_module(inter_feat_2)
#         inter_feat = inter_feat_2
#         # 门控输入
#         gate_input = torch.cat([bond_feat_input, node_feat_input, sh_feat], dim=-1)
#         gate_scalar = self.gate_mlp(gate_input)
#         gate_activation = torch.sigmoid(gate_scalar)

#         return inter_feat * gate_activation
    def forward(self, bond_feat_input, node_feat_input, sh_feat):
        bond_feat_proj = self.bond_linear(bond_feat_input)
        node_feat_proj = self.node_linear(node_feat_input)
        combined_feat = torch.cat([bond_feat_proj, node_feat_proj, sh_feat], dim=-1)
        inter_feat = self.fusion_module(combined_feat)
        # 门控输入
        gate_input = torch.cat([bond_feat_input, node_feat_input, sh_feat], dim=-1)
        gate_scalar = self.gate_mlp(gate_input)
        gate_activation = torch.sigmoid(gate_scalar)

        return inter_feat * gate_activation




class EdgeUpdateNetwork(nn.Module):
    def __init__(self, irreps_node: str, irreps_edge: str, irreps_sh: str,
                hidden_dim: int, num_rbf: int):
        """
        Args:
            irreps_node (str): 节点特征的Irreps.
            irreps_edge (str): 边特征的Irreps.
            irreps_sh (str): 方向特征的Irreps.
            hidden_dim (int): 定义标量融合隐藏层维度.
            num_rbf(int): 节点距离编码维度
        """
        super().__init__()

        self.irreps_edge = o3.Irreps(irreps_edge)
        self.irreps_node = o3.Irreps(irreps_node)
        self.irreps_sh = o3.Irreps(irreps_sh)
    
        scaling_factor = 2
        irreps_inter_list = [(mul // scaling_factor, ir) for mul, ir in self.irreps_edge]
        irreps_inter = o3.Irreps(irreps_inter_list).simplify()

        # 提取边特征的标量部分
        self.scalar_edge_irreps = o3.Irreps([(mul, ir) for mul, ir in self.irreps_edge if ir.l == 0])
        self.scalar_edge_dim = self.scalar_edge_irreps.dim

        # 定义标量融合MLP
        scalar_fusion_in_dim = self.scalar_edge_dim + num_rbf
        self.scalar_fusion_mlp = nn.Sequential(
            nn.Linear(scalar_fusion_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.scalar_edge_dim) 
        )

        self.bond_ffn_left = EquivariantBondFFN(
            self.irreps_edge, self.irreps_node, self.irreps_sh, self.irreps_edge, irreps_inter=irreps_inter)
        self.bond_ffn_right = EquivariantBondFFN(
            self.irreps_edge, self.irreps_node, self.irreps_sh, self.irreps_edge, irreps_inter=irreps_inter)

        self.node_ffn_left = LinearRS(self.irreps_node, self.irreps_edge)
        self.node_ffn_right = LinearRS(self.irreps_node, self.irreps_edge)
        self.self_ffn = LinearRS(self.irreps_edge, self.irreps_edge)
    
    def forward(self, h: torch.Tensor, e_in: torch.Tensor, edge_scalars: torch.Tensor, 
                edge_attr: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h (torch.Tensor): 节点特征
            e_in (torch.Tensor): 边特征
            edge_scalars：几何距离信息
            edge_attr：几何方向信息
            edge_index (torch.Tensor): 边索引

        Returns:
            torch.Tensor: 更新的边特征
        """
        # 准备输入特征
        h_node = h
        left_node, right_node = edge_index
        num_nodes = h_node.size(0)

        # 标量融合
        e_in_scalars = e_in.narrow(1, 0, self.scalar_edge_dim)
        e_in_high_order = e_in.narrow(1, self.scalar_edge_dim, e_in.shape[1] - self.scalar_edge_dim)
        combined_scalars = torch.cat([e_in_scalars, edge_scalars], dim=-1)
        fused_scalars = self.scalar_fusion_mlp(combined_scalars)
        h_bond = torch.cat([fused_scalars, e_in_high_order], dim=-1)

        # 计算五种类型的消息
        # a) 来自邻近边的消息：所有以i为端点的其他边e(k, i)传来的消息
        msg_from_neighbors_at_left = self.bond_ffn_left(h_bond, h_node[left_node], edge_attr)
        agg_at_nodes_from_left = scatter(msg_from_neighbors_at_left, right_node, dim=0, dim_size=num_nodes, reduce="add")
        final_msg_left = agg_at_nodes_from_left[left_node]
        
        # 所有以j为端点的其他边e(k, j)传来的消息
        msg_from_neighbors_at_right = self.bond_ffn_right(h_bond, h_node[right_node], edge_attr)
        agg_at_nodes_from_right = scatter(msg_from_neighbors_at_right, left_node, dim=0, dim_size=num_nodes, reduce="add")
        final_msg_right = agg_at_nodes_from_right[right_node]

        # 来自两端节点的消息
        msg_node_left = self.node_ffn_left(h_node[left_node])
        msg_node_right = self.node_ffn_right(h_node[right_node])
        
        # 来自边自身的更新
        msg_self = self.self_ffn(h_bond)

        # 聚合所有消息
        h_bond_aggregated = (final_msg_left + final_msg_right + 
                             msg_node_left + msg_node_right + 
                             msg_self)
        
        return h_bond_aggregated