import torch

from e3nn import o3
from e3nn.util.jit import compile_mode
import torch.nn as nn
from .mul_head_graph_attention import get_norm_layer, GraphAttention,FeedForwardNetwork
from .tensor_product_rescale import FullyConnectedTensorProductRescale
from .drop import GraphDropPath
from .edge_update import EdgeUpdateNetwork
from src.training.train_B import check_tensors

_RESCALE = True

# @compile_mode('script')
class E_DiT_Block(torch.nn.Module):
    '''
        一个同时更新节点和边特征的等变Transformer块 (Equivariant Denoising Transformer Block)。
        
        该模块采用双路并行的处理架构：
        1. 节点路径: 完全复用Equiformer的 'Norm -> Attention -> Add -> Norm -> FFN -> Add' 结构来更新节点。
        2. 边路径:   采用类似的结构 'Norm -> EdgeUpdate -> Add -> Norm -> FFN -> Add' 来更新边。
        
        两个路径都使用我们设计的自适应层归一化 AdaEquiLayerNorm，使其行为受时间步t调节。  
    '''
    
    def __init__(self,
        # 节点和边的Irreps定义
        irreps_node_input, irreps_node_attr, # irreps_node_attr是原子类型，Equiformer中是整数表示，而我们用one-hot注意适应
        irreps_edge_attr, irreps_node_output, # irreps_edge_attr 是Equifomer中原始的用来更新节点特征过程的边特征，注意与我们后来添加的被更新的边特征区分
        irreps_edge_input, irreps_edge_output, irreps_edge_attr_type, # 新添加的被更新的边特征-->根据EdgeUpdateNetwork实际需要添加其他
        fc_neurons,edge_update_hidden_dim,
        irreps_head, num_heads, irreps_pre_attn=None,
        rescale_degree=False, nonlinear_message=False,
        alpha_drop=0.1, proj_drop=0.1,
        drop_path_rate=0.0,
        # FNN 参数
        irreps_mlp_mid=None,

        # 归一化和嵌入参数
        time_embedding_dim = 128,
        norm_layer='layer'):
        
        super().__init__()

        ## 为节点更新路径实例化所有模块
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.irreps_pre_attn = self.irreps_node_input if irreps_pre_attn is None \
            else o3.Irreps(irreps_pre_attn)
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message
        self.irreps_mlp_mid = o3.Irreps(irreps_mlp_mid) if irreps_mlp_mid is not None \
            else self.irreps_node_input
        
        # 节点自适应归一化层1
        self.node_norm_1 = get_norm_layer(norm_layer)(irreps=self.irreps_node_input, time_embedding_dim=time_embedding_dim)
        # 节点更新的核心：图注意力模块
        self.ga = GraphAttention(irreps_node_input=self.irreps_node_input, 
            irreps_node_attr=self.irreps_node_attr,
            irreps_edge_attr=self.irreps_edge_attr, 
            irreps_node_output=self.irreps_node_input,
            fc_neurons=fc_neurons,
            irreps_head=self.irreps_head, 
            num_heads=self.num_heads, 
            irreps_pre_attn=self.irreps_pre_attn, 
            rescale_degree=self.rescale_degree, 
            nonlinear_message=self.nonlinear_message,
            alpha_drop=alpha_drop, 
            proj_drop=proj_drop)
        
        # 节点自适应归一化层2
        self.node_norm_2 = get_norm_layer(norm_layer)(irreps=self.irreps_node_input, time_embedding_dim=time_embedding_dim)
        #self.concat_norm_output = ConcatIrrepsTensor(self.irreps_node_input, 
        #    self.irreps_node_input)
        
        # 节点的前馈网络
        self.node_ffn = FeedForwardNetwork(
            irreps_node_input=self.irreps_node_input, #self.concat_norm_output.irreps_out, 
            irreps_node_attr=self.irreps_node_attr,
            irreps_node_output=self.irreps_node_output, 
            irreps_mlp_mid=self.irreps_mlp_mid,
            proj_drop=proj_drop)
        self.ffn_shortcut = None
        if self.irreps_node_input != self.irreps_node_output:
            self.ffn_shortcut = FullyConnectedTensorProductRescale(
                self.irreps_node_input, self.irreps_node_attr, 
                self.irreps_node_output, 
                bias=True, rescale=_RESCALE)
        
        ## 为边更新路径实例化所有模块
        self.irreps_edge_input = irreps_edge_input
        self.irreps_edge_output = irreps_edge_output
        self.irreps_edge_attr_type = irreps_edge_attr_type

        # 边归一化层
        self.edge_norm_1 = get_norm_layer(norm_layer)(irreps=irreps_edge_input, time_embedding_dim=time_embedding_dim)
        self.edge_norm_2 = get_norm_layer(norm_layer)(irreps=irreps_edge_input, time_embedding_dim=time_embedding_dim)
        
        # 边更新网络
        self.edge_updater = EdgeUpdateNetwork(
            irreps_node=self.irreps_node_input,
            irreps_edge=self.irreps_edge_input,
            hidden_dim=edge_update_hidden_dim,
            time_embedding_dim=time_embedding_dim
        )

        # 边的前馈网络 (与节点的FFN是独立的实例，拥有不同权重)
        self.edge_ffn = FeedForwardNetwork(
            irreps_node_input=self.irreps_edge_input,
            irreps_node_attr=self.irreps_edge_attr_type,
            irreps_node_output=self.irreps_edge_output,
            irreps_mlp_mid=self.irreps_mlp_mid,
            proj_drop=proj_drop
        )

        # 共享路径失活模块：该模块模块无参数、无状态，创建一次和两个效果一样
        self.drop_path = GraphDropPath(drop_path_rate) if drop_path_rate > 0. else None
        self.edge_updater_gate = torch.nn.Parameter(torch.tensor([1e-3]))
        self.edge_ffn_gate = torch.nn.Parameter(torch.tensor([1e-3]))

        self.node_ga_gate = torch.nn.Parameter(torch.tensor([1e-3]))
        self.node_ffn_gate = torch.nn.Parameter(torch.tensor([1e-3]))

    def forward(self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalars,
                edge_input, edge_attr_type, edge_index, t, batch, **kwargs):
        """
        执行一个E_DiT_Block的前向传播。

        Args:
            node_input (torch.Tensor): 节点特征, Irreps由__init__中的irreps_node_input定义。
            node_attr (torch.Tensor) : 节点类型，Irreps由__init__中的irreps_node_attr定义
            edge_input (torch.Tensor): 边特征, Irreps由__init__中的irreps_edge_input定义。
            edge_attr_type (torch.Tensor): 边特征, Irreps由__init__中的irreps_edge_attr_type定义。
            edge_attr (torch.Tensor): 基础边几何特征(球谐函数)。
            edge_scalars (torch.Tensor): 基础边距离特征(RBF)。
            t (torch.Tensor): 时间步。
            batch (torch.Tensor): 节点到图的分配索引。

        Returns:
            (torch.Tensor, torch.Tensor): 更新后的节点特征和边特征。
        """
        ## 节点更新路径 (Node Update Path) 
        node_output = node_input
        node_features = node_input
        # Pre-Normalization
        node_features = self.node_norm_1(node_features, t, batch)
        # node_norm_1_output = node_features

        # Graph Attention
        node_features = self.ga(
            node_input=node_features, 
            node_attr=node_attr, 
            edge_src=edge_src, edge_dst=edge_dst, 
            edge_attr=edge_attr, edge_scalars=edge_scalars,
            batch=batch)

        node_features = node_features * self.node_ga_gate
        
        if self.drop_path is not None:
            node_features = self.drop_path(node_features, batch)
        # 第一次残差连接
        node_output = node_output + node_features
        
        node_features = node_output
        # Pre-Normalization
        node_features = self.node_norm_2(node_features, t, batch)
        # node_features = self.concat_norm_output(node_norm_1_output, node_features)

        # Feed-Forward Network        
        node_features = self.node_ffn(node_features, node_attr)

        node_features = node_features * self.node_ffn_gate

        if self.ffn_shortcut is not None:
            node_output = self.ffn_shortcut(node_output, node_attr)
        if self.drop_path is not None:
            node_features = self.drop_path(node_features, batch)
        # 第二次残差连接
        node_output = node_output + node_features

        # 边更新路径 (Edge Update Path)
        edge_output = edge_input.clone()
        edge_features = edge_input
        # 获取每条边所属的图的索引
        edge_batch = batch[edge_src]

        # --- 第一个残差分支 (Edge Update) ---
        # Pre-Normalization
        norm_edge_features = self.edge_norm_1(edge_features, t, edge_batch)
        # print('before updater:',norm_edge_features)

        # 使用完全更新后的节点特征node_output作为上下文信息
        edge_update_amount = self.edge_updater(
            h=node_output,
            e_in=norm_edge_features,
            edge_index=edge_index,
            t=t,
            edge_batch=edge_batch
        )
        edge_update_amount = edge_update_amount * self.edge_updater_gate

        if self.drop_path is not None:
            edge_update_amount = self.drop_path(edge_update_amount, edge_batch)
        # 第一次残差连接
        edge_output = edge_output + edge_update_amount
        # print(f"after updater:",edge_output)

        # --- 第二个残差分支 (FFN) ---
        edge_features = edge_output
        # Pre-Normalization
        norm_edge_features = self.edge_norm_2(edge_features, t, edge_batch)
        # Feed-Forward Network
        # print(f"before ffn:", norm_edge_features)
        ffn_output = self.edge_ffn(norm_edge_features, edge_attr_type)

        ffn_output = ffn_output * self.edge_ffn_gate

        if self.drop_path is not None:
            ffn_output = self.drop_path(ffn_output, edge_batch)
        # 第二次残差连接
        edge_output = edge_output + ffn_output
        # print(f"after ffn:", edge_output)
        return node_output, edge_output
