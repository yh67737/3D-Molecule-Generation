import torch
import torch.nn as nn
from torch_scatter import scatter

import e3nn
from e3nn import o3
from e3nn.o3 import Irreps

from tensor_product_rescale import LinearRS, FullyConnectedTensorProductRescale


class EdgeUpdateNetwork(nn.Module):
    """
    一个完全等变的边特征更新网络。

    该网络分两步更新边特征：
    1. 邻域聚合: 首先，通过一次消息传递丰富每个节点的表示，使其包含一阶邻域的信息。
    2. 等变更新: 然后，使用一个等变的MLP(基于张量积)来融合每个边的端点（已更新）和边自身的特征，
       计算出对边特征的等变更新量。
    """

    def __init__(self, irreps_node: str, irreps_edge: str, hidden_dim: int = 128):
        """
        初始化网络。

        Args:
            irreps_node (str): 输入的节点特征的Irreps。
            irreps_edge (str): 输入/输出的边特征的Irreps (现在可以包含向量和张量)。
            hidden_dim (int): (为保持接口兼容性而保留，当前版本未直接使用)。
        """
        super().__init__()

        self.irreps_node = o3.Irreps(irreps_node)
        self.irreps_edge = o3.Irreps(irreps_edge)

        # 步骤 1: 用于创建节点消息的线性层，这部分逻辑很棒，我们保留。
        self.message_creation_layer = LinearRS(self.irreps_node, self.irreps_node)

        # 步骤 2: 定义一个等变的更新模块，以替换掉原来所有基于标量的操作。
        # 这个模块的输入由 h_src, h_dst, e_in 拼接而成。
        irreps_update_input = self.irreps_node + self.irreps_node + self.irreps_edge

        # 它的输出必须和边特征的Irreps一致，以便进行残差连接。
        irreps_update_output = self.irreps_edge

        # 使用一个强大的等变线性层来完成从 (h_src, h_dst, e_in) -> e_update 的映射。
        self.edge_update_mlp = LinearRS(irreps_update_input, irreps_update_output)

    def forward(self, h: torch.Tensor, e_in: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        执行前向传播。

        Args:
            h (torch.Tensor): 节点特征张量。
            e_in (torch.Tensor): 输入的边特征张量。
            edge_index (torch.Tensor): 描述边连接的张量, 形状为 [2, num_edges]。

        Returns:
            torch.Tensor: 计算出的边的等变更新量。
        """
        num_nodes = h.shape[0]
        row, col = edge_index  # source, destination

        # 1. 为每个节点创建初始消息
        messages = self.message_creation_layer(h)

        # 2. 在每个目标节点聚合所有来自源节点的消息
        aggregated_messages = scatter(messages[row], col, dim=0, dim_size=num_nodes, reduce="add")

        # 3. 将聚合信息加到原始节点特征上，得到更新后的节点表示
        h_updated = h + aggregated_messages

        # 4. 获取每条边对应的、经过邻域聚合后的端点节点特征
        h_row = h_updated[row]
        h_col = h_updated[col]

        # 5. 将 [h_src, h_dst, e_in] 拼接成一个大的等变特征向量
        fusion_input = torch.cat([h_row, h_col, e_in], dim=-1)

        # 6. 将融合特征送入等变MLP，直接计算出等变的更新量
        update_amount = self.edge_update_mlp(fusion_input)

        # 7. 返回这个更新量 (在E_DiT_Block中会执行残差连接 e_out = e_in + update_amount)
        return update_amount