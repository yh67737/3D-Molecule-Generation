# 基础组件
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.nn import EGNNConv 
from src.models.sorting_network.egnn_new import EquivariantBlock, SinusoidsEmbeddingNew
from torch_geometric.utils import to_dense_batch
import math

class PositionalEncoding(nn.Module):
    """
    function：正弦/余弦位置编码模块-->编码仅与d_model和max_len有关，与词向量无关

    """
    def __init__(self, d_model: int, max_len: int = 500):
        '''
        Args:
            d_model (int): The dimension of the embedding. 嵌入向量大小/节点特征h_0的维度大小
            max_len (int): The maximum possible length of a sequence. 序列最大长度/最大原子数

        '''
        super(PositionalEncoding, self).__init__() # 构造方法初始化

        ## 创建位置索引张量
        # position张量包含所有可能的位置pos(从0到max_len-1)
        # .unsqueeze(1)将position形状从 max_len]变为[max_len, 1]，为后续广播做准备
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1) # [max_len, 1]

        ## 计算频率：1 / 10000^(2i/d_model)
        # torch.arange(0, d_model, 2)提取所有偶数维度索引[0, 2, 4, ...,](不包含d_model)
        # `exp(x * (-log(y)))` 等价于 `y^(-x)`
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # [d_model//2]

        ## 初始化位置编码矩阵
        # 创建一个[max_len, d_model]大小的零矩阵，用于存放最终的编码结果
        pe = torch.zeros(max_len, d_model)

        ## 计算并填充pe偶数和奇数维度
        # 利用PyTorch的广播机制，[max_len, 1]的position和[d_model/2]的div_term相乘得到[max_len, d_model//2]，其(pos, i)元素值为pos / 10000^(2i/d_model)
        # pe[:, 0::2]选择所有偶数索引列
        pe[:, 0::2] = torch.sin(position * div_term)
        # pe[:, 1::2]选择所有奇数索引列
        pe[:, 1::2] = torch.cos(position * div_term)

        ## 将pe注册为模型的buffer
        # register_buffer将pe矩阵作为模型状态的一部分保存下来(state_dict)
        # 但它不是模型的参数，不会在反向传播时被更新梯度
        # 对于存储固定的非训练的数据（如位置编码）是标准做法；好处是当调用 model.to(device) 时，这个buffer会自动被移动到相应设备
        self.register_buffer('pe', pe) 

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t (torch.Tensor): A tensor of shape [num_nodes] containing position indices.
        Returns:
            torch.Tensor: Positional encodings of shape [num_nodes, d_model].
        """
        # 输入t包含需要查找的位置索引,如[0, 1, 2, 0](张量)
        # self.pe是预先计算好的[max_len, d_model]大小的编码表
        # PyTorch索引机制将从pe表中取出第0, 1, 2, 0行，组成[4, d_model]的输出张量返回
        return self.pe[t]

'''
# 边特征的融合处理较简单
class GNN_Core(nn.Module):
    """
    新的GNN核心，使用来自 egnn_new.py 的 EquivariantBlock。
    这个模块同时更新节点特征 h 和坐标 coords。
    """
    def __init__(self, feature_dim: int, edge_feature_dim: int, num_layers: int = 3, attention: bool = True):
        super(GNN_Core, self).__init__()
        
        self.egnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            # EquivariantBlock 内部已经包含了多层GCL用于特征更新，和一层用于坐标更新。
            # 这里我们将 inv_sublayers (内部GCL层数) 设为 2，n_layers (EquivariantBlock数量) 由外部的 num_layers 控制。
            # 'edge_feat_nf' 需要包含距离信息(1维)和你自己的边特征。
            self.egnn_layers.append(
                EquivariantBlock(
                    hidden_nf=feature_dim, 
                    edge_feat_nf=1 + edge_feature_dim, # 1维距离 + N维化学键特征
                    n_layers=2,  # 每个Block内部的GCL层数, 可调
                    attention=attention,
                    norm_diff=True,
                    tanh=False
                )
            )

    def forward(self, h: torch.Tensor, coords: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:  # 返回值改回只有 h
        """
        前向传播 (优化版)
        - 利用EGNN的等变消息传递来更新节点特征 h。
        - 保持输入的真实坐标 coords 不变，仅将其用于计算，不进行更新。
        
        Args:
            h: 节点特征, shape [num_nodes, feature_dim]
            coords: 节点坐标, shape [num_nodes, 3]
            edge_index: 边索引, shape [2, num_edges]
            edge_attr: 你的化学键特征, shape [num_edges, edge_feature_dim]
        
        Returns:
            torch.Tensor: The updated node features h_final.
        """
        # 计算距离，用作边特征的一部分
        radial, _ = coord2diff(coords, edge_index)

        # 准备完整的边特征
        full_edge_attr = torch.cat([radial, edge_attr], dim=1)
        
        # 循环通过所有EquivariantBlock只更新h，忽略返回的coords
        for egnn_block in self.egnn_layers:
            # 下一次循环传入的coords仍然是原始的、未经修改的真实坐标
            h, _ = egnn_block(h, coords, edge_index, edge_attr=full_edge_attr)
        
        # 只返回最终的节点特征
        return h
'''

'''
class GNN_Core(nn.Module):
    """
    GNN核心 (最终优化版)
    - 使用 SinusoidsEmbeddingNew 对距离进行高维编码。
    - 使用 MLP 对化学键特征进行高维编码。
    - 两种编码的维度匹配，执行加性融合。
    """
    def __init__(self, feature_dim: int, edge_feature_dim: int, num_layers: int = 3, attention: bool = True):
        """
        Args:
            feature_dim (int): 节点特征的维度。
            edge_feature_dim (int): 原始边（化学键）特征的维度。
            num_layers (int): EquivariantBlock 的层数。
            attention (bool): 是否在EGNN中使用注意力机制。
        """
        super(GNN_Core, self).__init__()

        ## 边特征融合模块
        # 实例化 SinusoidsEmbeddingNew 作为距离编码器
        self.sin_embedding = SinusoidsEmbeddingNew()
        # 获取其固定的输出维度
        edge_embedding_dim = self.sin_embedding.dim # 是EDM根据qm9设定的超参数计算得到的

        # 化学键特征的映射网络 (edge_feature_dim -> edge_embedding_dim)
        # 它的输出维度必须严格等于距离编码的维度，才能相加
        self.bond_mlp = nn.Sequential(
            nn.Linear(edge_feature_dim, edge_embedding_dim),
            nn.SiLU(),
            nn.Linear(edge_embedding_dim, edge_embedding_dim)
        )

        ## 定义等变网络
        self.egnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.egnn_layers.append(
                EquivariantBlock(
                    hidden_nf=feature_dim, 
                    # 最终输入到EGNN的边特征维度就是融合后的维度
                    edge_feat_nf=edge_embedding_dim,
                    n_layers=2, #可修改
                    attention=attention,
                    norm_diff=True,
                    tanh=False
                )
            )

    def forward(self, h: torch.Tensor, coords: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        # 计算原始距离并通过 SinusoidsEmbeddingNew 编码
        radial, _ = coord2diff(coords, edge_index)
        dist_embedding = self.sin_embedding(radial)
        
        # 将化学键特征通过 MLP 编码
        bond_embedding = self.bond_mlp(edge_attr)
        
        # 执行加性融合
        full_edge_attr = dist_embedding + bond_embedding
        
        # 循环通过所有 EquivariantBlock
        for egnn_block in self.egnn_layers:
            h, _ = egnn_block(h, coords, edge_index, edge_attr=full_edge_attr)
        
        return h
'''
    
class GNN_Core(nn.Module):
    """
    GNN核心 (最终优化版)
    - 使用 SinusoidsEmbeddingNew 对距离进行高维编码。
    - 使用 MLP 对化学键特征进行高维编码。
    - 两种编码再EGNN内部拼接融合
    """
    def __init__(self, feature_dim: int, edge_feature_dim: int, num_layers: int = 3, attention: bool = True):
        """
        Args:
            feature_dim (int): 节点特征的维度。
            edge_feature_dim (int): 原始边（化学键）特征的维度。
            num_layers (int): EquivariantBlock 的层数。
            attention (bool): 是否在EGNN中使用注意力机制。
        """
        super(GNN_Core, self).__init__()

        # 实例化SinusoidsEmbeddingNew
        self.sin_embedding = SinusoidsEmbeddingNew()
        dist_embedding_dim = self.sin_embedding.dim  # 值为12，是EDM根据qm9设定的超参数计算得到的

        # 定义化学键特征的映射网络
        bond_embedding_dim = 16 
        self.bond_mlp = nn.Sequential(
            nn.Linear(edge_feature_dim, bond_embedding_dim),
            nn.SiLU(),
            nn.Linear(bond_embedding_dim, bond_embedding_dim)
        )

        # 定义等变网络
        self.egnn_layers = nn.ModuleList()
        final_edge_feat_dim = dist_embedding_dim + bond_embedding_dim # 12 + 16 = 28

        for _ in range(num_layers):
            self.egnn_layers.append(
                EquivariantBlock(
                    hidden_nf=feature_dim, 
                    edge_feat_nf=final_edge_feat_dim,
                    n_layers=2,
                    attention=attention,
                    norm_diff=True,
                    tanh=False,
                    sin_embedding=self.sin_embedding 
                )
            )

    def forward(self, h: torch.Tensor, coords: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        
        # 将化学键特征通过MLP编码
        bond_embedding = self.bond_mlp(edge_attr)
        
        # 循环通过所有 EquivariantBlock
        for egnn_block in self.egnn_layers:
            h, _ = egnn_block(h, coords, edge_index, edge_attr=bond_embedding)
        
        return h


# SortingNetwork主模型
class SortingNetwork(nn.Module):
    def __init__(self, num_atom_features: int, num_bond_features: int, hidden_dim: int,
                 gnn_layers: int = 3, max_nodes: int = 500):
        super(SortingNetwork, self).__init__()
        
        # 设定超参数：隐藏层维度和最大图节点数
        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes

        ### 实例化基本组件
        
        ## 输入特征投影层(nn.Linear)：将原始原子特征维度(num_atom_features)映射到网络内部统一的隐藏维度(hidden_dim)
        # 确保后续所有特征向量维度一致，便于相加和处理
        self.input_proj = nn.Linear(num_atom_features, hidden_dim) 

        ## 成环信息独立嵌入层(默认输入特征是2维 num_embeddings=2)，映射为hidden_dim维的可学习向量
        self.ring_embedding = nn.Embedding(num_embeddings=2, embedding_dim=hidden_dim) 

        ## 位置编码模块：实例化正弦/余弦位置编码模块，将用于在自回归的每一步为已选择节点添加其在序列中的位置信息
        self.pe_module = PositionalEncoding(d_model=hidden_dim, max_len=max_nodes)
        
        # 图网络模块：实例化基于EGNN的图特征提取器，处理并融合节点、边、3D坐标三类信息 
        self.gnn_core = GNN_Core(
            feature_dim=hidden_dim,
            edge_feature_dim=num_bond_features, 
            num_layers=gnn_layers
        )

        ## 输出MLP(nn.Sequential)：两层的MLP，将GNN输出的高维节点特征转换为分数logits
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, 1)
        )

    def forward(self, data): 
        """
        实现自回归的序列采样，支持批处理
        Args:
            data (torch_geometric.data.Batch): A batch of graphs.
        Returns:
            tuple[list[list[int]], torch.Tensor]:
                - A list of sampled orderings for each graph in the batch.
                - A tensor of total log probabilities for each sequence, shape [batch_size].
        """
        ## 数据解包与初始化
        # 从PyG的Batch对象中取出所有需要数据
        x, pos, edge_index, edge_attr = data.x, data.pos, data.edge_index, data.edge_attr
        pring_out, batch = data.pring_out, data.batch # data.batch提供各个节点的身份证
        
        # 当前批次的图数量
        batch_size = data.num_graphs
        # 当前批次所有图的节点总数
        num_total_nodes = data.num_nodes
        # 获取当前设备（CPU/GPU）
        device = x.device

        ## 创建节点基础特征
        # 将原始原子特征投影到隐藏维度
        x_proj = self.input_proj(x)
        # 将成环标志(0/1)通过Embedding层映射为高维向量
        ring_embeds = self.ring_embedding(pring_out.long().squeeze(-1)) 
        # 获得融合了原子类型和成环信息的基础特征
        x_base = x_proj + ring_embeds  # shape：[num_total_nodes,hidden_dim]

        ## 初始化自回归循环的状态变量
        # node_positions记录每个节点在序列中的位置(t=0,1,2...), 全部初始化为-1表示未选中
        node_positions = torch.full((num_total_nodes,), -1, dtype=torch.long, device=device)
        # nodes_selected_mask记录每个节点是否已被选择，用于屏蔽；初始化为0(全未选中)
        nodes_selected_mask = torch.zeros(num_total_nodes, dtype=torch.bool, device=device) # 二维张量
        # total_log_probs存储批次中的每个图的序列累计对数概率 -->产生当前序列的概率大小
        total_log_probs = torch.zeros(batch_size, device=device)
        
        ## 获取批次中每个图的真实大小，用于控制循环
        # to_dense_batch返回掩码标记出哪些节点属于哪个图  
        _, real_nodes_mask = to_dense_batch(x, batch)
        num_nodes_per_graph = real_nodes_mask.sum(dim=1)
        # 循环次数由批次中最大图的节点数决定
        max_num_nodes_in_batch = int(num_nodes_per_graph.max())
        
        ## 准备列表用于存储每个图最终的排序结果(全局索引)
        orderings = [[] for _ in range(batch_size)]

        # 步骤a:自回归循环
        for t in range(max_num_nodes_in_batch):
            # 步骤b: 动态创建当前步t的节点特征
            x_t = x_base.clone() # 克隆一份原始特征，进行动态修改
            # 只为已选择的节点添加位置编码
            if t > 0: # 从第二步(t=1)开始，需要添加位置编码
                # 找到所有已被选择节点的全局索引
                selected_nodes_indices = nodes_selected_mask.nonzero().squeeze(-1)
                # 找到这些节点被选择时的位置（0,1,2...）
                positions_of_selected = node_positions[selected_nodes_indices]
                # 为已选节点添加位置编码
                x_t[selected_nodes_indices] += self.pe_module(positions_of_selected) 

            # 步骤c: GNN计算
            # 将动态更新后的节点特征和图的其他信息传入GNN
            node_embeddings = self.gnn_core(x_t, pos, edge_index, edge_attr) # edge_index根据PyG的稀疏批处理测略相应地做了改变，不再保持每张图从0开始编号，而是被转换成了全局编号
            
            # 将GNN输出的嵌入通过MLP转换为logits
            logits = self.output_mlp(node_embeddings).squeeze(-1) # Shape: [num_total_nodes]


            # 步骤d: 屏蔽已选择节点
            logits = torch.where(nodes_selected_mask.detach(), -float('inf'), logits)

            # 批处理softmax和采样
            dense_logits, _ = to_dense_batch(logits, batch, fill_value=-float('inf'))
            
            # =================== [开始最终修复] ===================
            # 核心修复：处理全为-inf的行，以防止在计算softmax时产生NaN

            # 1. 找到那些所有 logits 都为 -inf 的行（即已选完的图）
            all_inf_mask = torch.all(dense_logits == -float('inf'), dim=1)
            
            # 2. 对这些行进行“无害化”处理：
            #    我们将这些行中的第一个元素设置为0。这样，torch.max的结果就是0，
            #    避免了 -inf - (-inf) 的情况。后续 softmax 将会为该行生成一个
            #    one-hot 向量 [1, 0, 0, ...]，这是一个有效的概率分布，
            #    可以安全地传入 multinomial，尽管我们后续不会使用它的采样结果。
            if all_inf_mask.any():
                dense_logits[all_inf_mask, 0] = 0.0

            # 3. 现在可以安全地进行数值稳定的 softmax 计算了
            logits_stable = dense_logits - torch.max(dense_logits, dim=1, keepdim=True).values
            probs = F.softmax(logits_stable, dim=1)
            
            # =================== [结束最终修复] ===================

            # 步骤e: 采样
            graphs_not_done_mask = (t < num_nodes_per_graph)
            if not graphs_not_done_mask.any():
                break
            
            # multinomial 现在可以安全执行，因为它收到的 probs 不会再包含 NaN
            sampled_local_indices = torch.multinomial(probs[graphs_not_done_mask].detach(), num_samples=1).squeeze(-1)

            # 步骤f: 记录对数概率并更新状态
            log_prob_t = torch.log(probs[graphs_not_done_mask].gather(1, sampled_local_indices.unsqueeze(-1)).squeeze(-1) + 1e-9)
            
            # 只对未完成的图进行概率累加，自动忽略那些我们“无害化处理”过的已完成的图
            total_log_probs[graphs_not_done_mask] += log_prob_t

            # 将采样的局部索引转换为全局索引
            batch_indices_not_done = graphs_not_done_mask.nonzero().squeeze(-1)

            ptr = data.ptr if hasattr(data, 'ptr') else None
            if ptr is None:
                ptr = torch.cat([torch.tensor([0], device=device), num_nodes_per_graph.cumsum(dim=0)])

            ptr_not_done = ptr[batch_indices_not_done]
            sampled_global_indices = ptr_not_done + sampled_local_indices

            # 1. 克隆当前的 mask，得到一个全新的张量，它与计算图无关
            next_nodes_selected_mask = nodes_selected_mask.clone()
            
            # 2. 在这个新的 mask 上进行修改
            next_nodes_selected_mask[sampled_global_indices] = True
            
            # 3. 将变量名重新指向这个新的、已更新的 mask，用于下一次循环
            nodes_selected_mask = next_nodes_selected_mask
            
            # (旧的位置编码更新逻辑也需要相应调整)
            node_positions[sampled_global_indices] = t
            
            # 记录排序结果
            for i, global_idx in enumerate(sampled_global_indices): 
                graph_idx = batch_indices_not_done[i]
                orderings[graph_idx].append(global_idx.item())
        
        # 将排序结果从全局索引转化为局部索引
        local_orderings = []
        for i in range(batch_size):
            global_ordering_list = orderings[i]
            if not global_ordering_list:
                local_orderings.append([])
                continue

            global_ordering_tensor = torch.tensor(orderings[i], device=device, dtype=torch.long)
            graph_start_ptr = data.ptr[i] if hasattr(data, 'ptr') else ptr[i]
            local_ordering_tensor = global_ordering_tensor - graph_start_ptr
            local_orderings.append(local_ordering_tensor.tolist())

        return local_orderings, total_log_probs