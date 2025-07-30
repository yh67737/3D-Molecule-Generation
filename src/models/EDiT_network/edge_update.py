import torch
import torch.nn as nn
from e3nn import o3
from e3nn.o3 import Irreps
from torch_scatter import scatter
import e3nn
from e3nn.o3 import Linear
import torch.nn.functional as F
from e3nn.nn import NormActivation
from src.models.EDiT_network.layer_norm import AdaEquiLayerNorm
from src.models.EDiT_network.tensor_product_rescale import LinearRS
import math
import sys
import e3nn
import os


# 位于 edge_update.py 文件顶部

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
        # 创建一个与输入形状相同的输出张量来存放结果
        out = torch.zeros_like(x)

        # 1. 处理标量部分 (l=0)
        scalar_dim = self.scalar_irreps.dim
        if scalar_dim > 0:
            scalars = x[:, :scalar_dim]
            out[:, :scalar_dim] = self.activation(scalars)

        # 2. 处理非标量部分 (l>0)
        for start, end, mul, dim in self.non_scalar_dims:
            # 提取当前类型的非标量特征
            non_scalars = x[:, start:end].view(-1, mul, dim)

            # 计算范数，并加入一个极小的 epsilon 来防止除以零
            # 这是保证数值稳定性的核心
            norm = torch.linalg.norm(non_scalars, dim=-1, keepdim=True)
            activated_norm = self.activation(norm)

            # 使用一个安全的除法来缩放向量
            # 当 norm 接近0时，分母 (norm + epsilon) 不为0，避免了 NaN
            epsilon = 1e-8
            scaling_factor = activated_norm / (norm + epsilon)

            # 将缩放后的向量放回输出张量
            out[:, start:end] = (non_scalars * scaling_factor).view(-1, mul * dim)

        return out

class SinusoidalTimeEmbedding(nn.Module):
    """标准的正弦时间嵌入"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class InterModule(nn.Module):
    def __init__(self, irreps_in, irreps_mid, irreps_out):
        super().__init__()
        self.linear1 = Linear(irreps_in, irreps_mid)
        self.norm_act = NormActivation(irreps_mid, F.silu)
        self.linear2 = Linear(irreps_mid, irreps_out)

    def forward(self, x):
        # 经过第一个线性层
        x = self.linear1(x)

        # --- 核心修复 ---
        # 在进入计算敏感的 NormActivation 之前，强制转换为 float32
        original_dtype = x.dtype
        x = self.norm_act(x.to(torch.float32))
        # 计算完毕后，转换回原来的数据类型
        x = x.to(original_dtype)
        # --- 结束修复 ---

        # 经过第二个线性层
        x = self.linear2(x)
        return x


class EquivariantBondFFN(nn.Module):
    """
    BondFFN的等变实现。
    """

    def __init__(self, irreps_bond: Irreps, irreps_node: Irreps, irreps_time: Irreps, irreps_message: Irreps,
                 irreps_inter: Irreps = None):
        """
        Args:
            irreps_bond (Irreps): 边特征的 Irreps。
            irreps_node (Irreps): 节点特征的 Irreps。
            irreps_time (Irreps): 时间特征的 Irreps。
            irreps_message (Irreps): 最终输出消息的 Irreps。
            irreps_inter (Irreps, optional): 中间交互特征的 Irreps。如果为 None，则默认为 irreps_message。
        """
        super().__init__()
        self.irreps_bond = irreps_bond
        self.irreps_node = irreps_node
        self.irreps_time = irreps_time
        self.irreps_message = irreps_message

        if irreps_inter is None:
            irreps_inter = irreps_message

        # 1. 分别投射 (对应 bond_linear, node_linear)
        self.bond_linear = LinearRS(self.irreps_bond, irreps_inter)
        self.node_linear = LinearRS(self.irreps_node, irreps_inter)

        self.tensor_product = o3.FullTensorProduct(irreps_inter, irreps_inter)

        # 2. 从创建好的层中提取其输出 Irreps，用于定义下一层
        irreps_tp_out = self.tensor_product.irreps_out

        # The rest of your code remains the same
        self.inter_module = nn.Sequential(
            Linear(irreps_tp_out, self.irreps_message),
            StableNormActivation(self.irreps_message, F.silu),
            Linear(self.irreps_message, self.irreps_message)
        )

        ### 门控路径 ###
        # 4. 独立的门控 MLP (对应 gate)
        #    它的输入是原始特征，输出必须是一个标量，用于门控
        gate_mlp_irreps_in = (self.irreps_bond + self.irreps_node + self.irreps_time).simplify()

        # 定义一个隐藏层 Irreps，让门控 MLP 更有表达力
        irreps_gate_hidden = o3.Irreps(gate_mlp_irreps_in)

        irreps_gate_hidden.sort()
        irreps_gate_hidden = irreps_gate_hidden.simplify()

        self.gate_mlp = nn.Sequential(
            Linear(gate_mlp_irreps_in, irreps_gate_hidden),
            StableNormActivation(irreps_gate_hidden, F.silu),
            Linear(irreps_gate_hidden, o3.Irreps("1x0e"))
        )

    def forward(self, bond_feat_input, node_feat_input, time):
        # 1. 投射
        bond_feat_proj = self.bond_linear(bond_feat_input)
        node_feat_proj = self.node_linear(node_feat_input)

        # 2. 交互
        inter_feat = self.tensor_product(bond_feat_proj, node_feat_proj)

        # 3. MLP 处理
        inter_feat = self.inter_module(inter_feat)

        # 4a. 准备门控输入
        gate_input = torch.cat([bond_feat_input, node_feat_input, time], dim=-1)
        # 4b. 计算标量门控值
        gate_scalar = self.gate_mlp(gate_input)
        # 4c. 应用 sigmoid 激活
        gate_activation = torch.sigmoid(gate_scalar)

        # 将标量门控信号乘到主路径的输出上
        return inter_feat * gate_activation


class EdgeUpdateNetwork(nn.Module):
    """
    一个与 E_DiT_Block 兼容的、严格遵循原始流程的边更新网络。
    """

    def __init__(self, irreps_node: str, irreps_edge: str, hidden_dim: int, time_embedding_dim: int):
        """
        Args:
            irreps_node (str): 节点特征的 Irreps。
            irreps_edge (str): 边特征的 Irreps。
            hidden_dim (int): 用于定义内部隐藏层 Irreps 的维度。
            time_embedding_dim (int): 时间嵌入的维度，用于推断时间 Irreps。
        """
        super().__init__()

        self.irreps_edge = o3.Irreps(irreps_edge)
        self.irreps_node = o3.Irreps(irreps_node)

        self.time_embed = SinusoidalTimeEmbedding(time_embedding_dim)

        self.irreps_time = o3.Irreps(f'{time_embedding_dim}x0e')

        scaling_factor = 2
        irreps_inter_list = [(mul // scaling_factor, ir) for mul, ir in self.irreps_edge]
        irreps_inter = o3.Irreps(irreps_inter_list).simplify()

        self.bond_ffn_left = EquivariantBondFFN(
            self.irreps_edge, self.irreps_node, self.irreps_time, self.irreps_edge, irreps_inter=irreps_inter
        )
        self.bond_ffn_right = EquivariantBondFFN(
            self.irreps_edge, self.irreps_node, self.irreps_time, self.irreps_edge, irreps_inter=irreps_inter
        )

        self.node_ffn_left = Linear(self.irreps_node, self.irreps_edge)
        self.node_ffn_right = Linear(self.irreps_node, self.irreps_edge)
        self.self_ffn = Linear(self.irreps_edge, self.irreps_edge)
        self.final_norm_and_transform = AdaEquiLayerNorm(
            irreps=self.irreps_edge,
            time_embedding_dim=time_embedding_dim
        )

    def forward(self, h: torch.Tensor, e_in: torch.Tensor, edge_index: torch.Tensor, t: torch.Tensor,
                edge_batch: torch.Tensor) -> torch.Tensor:

        """
        Args:
            h (torch.Tensor): 节点特征 (来自 E_DiT_Block 的 node_output)。
            e_in (torch.Tensor): 边特征 (来自 E_DiT_Block 的 norm_edge_features)。
            edge_index (torch.Tensor): 边索引。
            t (torch.Tensor): 时间嵌入特征。

        Returns:
            torch.Tensor: 计算出的边特征更新量或新特征。
        """
        # 1. 将 E_DiT_Block 的参数名映射到我们内部使用的变量名
        bond_time = self.time_embed(t)
        h_node = h
        h_bond = e_in
        bond_index = edge_index

        # 2. 内部计算逻辑保持不变
        N = h_node.size(0)
        left_node, right_node = bond_index
        bond_time = self.time_embed(t)[edge_batch]

        msg_bond_left = self.bond_ffn_left(h_bond, h_node[left_node], bond_time)
        aggregated_at_right = scatter(msg_bond_left, right_node, dim=0, dim_size=N, reduce="add")
        final_msg_left = aggregated_at_right[left_node]

        msg_bond_right = self.bond_ffn_right(h_bond, h_node[right_node], bond_time)
        aggregated_at_left = scatter(msg_bond_right, left_node, dim=0, dim_size=N, reduce="add")
        final_msg_right = aggregated_at_left[right_node]

        msg_node_left = self.node_ffn_left(h_node[left_node])
        msg_node_right = self.node_ffn_right(h_node[right_node])
        msg_self = self.self_ffn(h_bond)

        h_bond_sum = (final_msg_left + final_msg_right + msg_node_left + msg_node_right + msg_self)
        update_amount = self.final_norm_and_transform(
            node_input=h_bond_sum,
            t=t,
            batch=edge_batch
        )  # 直接将聚合后的所有消息作为更新量

        # (可选) 如果想包含后处理，可以这样做，但这会改变原始 EdgeBlock 的流程
        # h_bond_out = self.layer_norm(h_bond_sum)
        # h_bond_out = self.act(h_bond_out)
        # h_bond_out = self.out_transform(h_bond_out)
        # update_amount = h_bond_out

        return update_amount