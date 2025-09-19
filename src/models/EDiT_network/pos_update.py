import torch
import torch.nn as nn
from e3nn import o3
from torch_scatter import scatter_sum
from .layer_norm import AdaEquiLayerNorm
from .tensor_product_rescale import FullyConnectedTensorProductRescale, LinearRS

#等变的坐标更新模块，模仿Moldiff的BondFFN+PosUpdate，将BondFFN的代码融入PosUpdate结构中
class EquivariantPosUpdate(nn.Module):
    
    def __init__(self, 
                 irreps_node_in: str, 
                 irreps_edge_in: str, 
                 time_embedding_dim: int = 128,
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
        
        # 准备节点特征
        # 定义两个独立的线性层，用于分别变换源节点和目标节点特征
        self.src_node_transform = LinearRS(irreps_node_in, irreps_edge_in)
        self.dst_node_transform = LinearRS(irreps_node_in, irreps_edge_in)
        
        # 预测每条边的标量权重
        
        # a.融合两端节点特征：使用FCTP
        self.node_fusion_tp = FullyConnectedTensorProductRescale(
            irreps_in1=irreps_edge_in, 
            irreps_in2=irreps_edge_in, 
            irreps_out=irreps_edge_in
        )

        # b.将融合后的节点、初始边特征映射到隐藏维度
        self.node_transform = LinearRS(irreps_edge_in, hidden_irreps)
        self.edge_transform = LinearRS(irreps_edge_in, hidden_irreps)
        
        # c.融合节点和边特征
        self.edge_fusion_tp = FullyConnectedTensorProductRescale(
            irreps_in1=hidden_irreps, 
            irreps_in2=hidden_irreps, 
            irreps_out=hidden_irreps
        )

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

        # 准备节点特征
        src_feat = self.src_node_transform(h_node[edge_src])
        dst_feat = self.dst_node_transform(h_node[edge_dst])

        # 预测每条边的标量权重
        fused_nodes = self.node_fusion_tp(src_feat, dst_feat)
        fused_nodes_trans = self.node_transform(fused_nodes)
        edge_trans = self.edge_transform(h_edge)
        fused_all = self.edge_fusion_tp(fused_nodes_trans, edge_trans)
        batch_edge = batch[edge_src] # 为边特征分配对应的批次索引
        normalized_feat = self.norm(fused_all, t, batch_edge)   
        scalar_weight = self.scalar_predictor(normalized_feat) # shape: [num_edges, 1]

        # 计算力向量 (force_edge)
        # relative_vec = pos[edge_src] - pos[edge_dst]
        # distance = torch.norm(relative_vec, dim=-1, keepdim=True).clamp(min=1e-8)
        force_edge = scalar_weight * (relative_vec / distance) / (distance + 1.0)

        # 聚合力到节点上
        delta_pos = scatter_sum(force_edge, edge_src, dim=0, dim_size=h_node.shape[0])

        return delta_pos