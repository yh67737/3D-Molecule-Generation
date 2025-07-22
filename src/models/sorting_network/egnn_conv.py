# egnn_conv.py

import torch
import torch.nn as nn
from torch_scatter import scatter_add

class EGNNConv(nn.Module):
    """
    手动实现的 E(n) Equivariant Graph Convolutional Layer (EGNNConv)
    参考 'E(3) Equivariant Diffusion for Molecules' (EDM) 官方实现。
    """
    def __init__(self, in_nf, hidden_nf, out_nf, edge_attr_nf=0, act_fn=nn.SiLU(), normalize=False, residual=True, attention=False):
        """
        Args:
            in_nf: 输入节点特征的维度
            hidden_nf: 隐藏层的维度
            out_nf: 输出节点特征的维度
            edge_attr_nf: 边特征的维度
            act_fn: 激活函数
            normalize: 是否对坐标更新进行归一化
            residual: 是否使用残差连接
            attention: 是否使用注意力机制
        """
        super(EGNNConv, self).__init__()
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_nf * 2 + 1 + edge_attr_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + in_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, out_nf)
        )

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid()
            )

    def forward(self, h, coord, edge_index, edge_attr=None):
        """
        前向传播
        Args:
            h: 节点特征, shape [num_nodes, in_nf]
            coord: 节点坐标, shape [num_nodes, 3]
            edge_index: 边索引, shape [2, num_edges]
            edge_attr: 边特征, shape [num_edges, edge_attr_nf]
        """
        row, col = edge_index
        
        # 计算坐标差和距离
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        # 构建边消息的输入
        edge_input_list = [h[row], h[col], radial]
        if edge_attr is not None:
            edge_input_list.append(edge_attr)
        edge_input = torch.cat(edge_input_list, dim=1)

        # 通过MLP计算边消息
        edge_message = self.edge_mlp(edge_input)

        # 计算注意力权重(可选)
        if self.attention:
            att_val = self.att_mlp(edge_message)
            edge_message = edge_message * att_val
            
        # 聚合消息到节点
        # 使用 torch_scatter.scatter_add 高效完成消息聚合
        # 将所有指向节点`i`的消息相加，得到节点`i`收到的总消息
        node_agg_message = scatter_add(edge_message, row, dim=0, dim_size=h.size(0))

        # 更新节点特征
        node_input = torch.cat([h, node_agg_message], dim=1)
        h_final = self.node_mlp(node_input)
        if self.residual:
            h_final = h + h_final

        return h_final, coord