import torch
import torch.nn as nn
from e3nn.o3 import Irreps
from .tensor_product_rescale import LinearRS, FullyConnectedTensorProductRescale
from .mul_head_graph_attention import get_norm_layer
from .layer_norm import TimestepEmbedder
from .input_embedding import InputEmbeddingLayer
from .e_dit_block import E_DiT_Block

class MultiTaskHead(nn.Module):
    """
    接收E-DiT Block提纯后的最终特征，并根据自回归任务的
    掩码(mask)，对新生成的原子和化学键进行属性预测
    """

    def __init__(self, 
                 irreps_final_node_feature: str, 
                 irreps_final_edge_feature: str, 
                 args):
        """
        Args:
            irreps_final_node_feature(str): 输入的最终节点特征的Irreps定义
            irreps_final_edge_feature(str): 输入的最终边特征的Irreps定义
            args: 包含模型配置的参数对象，如隐藏层维度、类别数等
        """
        super().__init__()
        self.irreps_final_node_feature = Irreps(irreps_final_node_feature)
        self.irreps_final_edge_feature = Irreps(irreps_final_edge_feature) 

        # 原子类型预测头
        self.node_scalar_proj = LinearRS(self.irreps_final_node_feature, Irreps(args.irreps_node_attr))
        # 化学键类型预测头
        self.edge_scalar_proj = LinearRS(self.irreps_final_edge_feature, Irreps(args.irreps_edge_attr_type))
        
    def forward(self,
                h_final: torch.Tensor,
                e_final: torch.Tensor,
                r_t_final: torch.Tensor,
                target_node_mask: torch.Tensor,
                target_edge_mask: torch.Tensor) -> dict:
        """
        Args:
            h_final (torch.Tensor): 最终节点特征张量。
            e_final (torch.Tensor): 最终边特征张量。
            r_t_final (torch.Tensor): 最终坐标张量。
            target_node_mask (torch.Tensor): 布尔掩码，标记值为True的是目标节点 (仅一个)。
            target_edge_mask (torch.Tensor): 布尔掩码，标记值为True的是目标边。
            edge_index (torch.Tensor): 完整的边索引。

        Returns:
            一个包含所有预测结果的字典。
        """

        # 坐标预测：直接使用去噪网络输出的最终坐标作为预测结果
        pred_pos = r_t_final[target_node_mask]

        # 原子类型预测
        # 使用掩码提取目标节点特征
        target_node_features = h_final[target_node_mask]
        pred_atom_type = self.node_scalar_proj(target_node_features)

        # 化学键类型预测
        # 使用掩码提取目标边的特征
        target_edge_features = e_final[target_edge_mask]
        pred_bond_type = self.edge_scalar_proj(target_edge_features)

        return {
            "atom_type_logits": pred_atom_type,  # shape: [1, num_atom_types]
            "bond_logits": pred_bond_type,  # shape: [num_target_edges, num_bond_types]
            "predicted_r0": pred_pos               # shape: [num_all_nodes, 3]
        }


class E_DiT_Network(nn.Module):
    """
    最终的E-DiT网络主体，负责组装和调用所有模块。
    """
    def __init__(self, args):
        super().__init__()
        self.args = args

        # 输入嵌入层
        self.embedding_layer = InputEmbeddingLayer(
            # NodeEmbeddingNetwork参数
            irreps_node_embedding=args.irreps_node_hidden,
            num_atom_types=args.num_atom_types,
            node_embedding_hidden_dim=args.node_embedding_hidden_dim,
            # EdgeEmbeddingNetwork参数
            irreps_sh=args.irreps_sh,
            max_radius=args.rbf_cutoff,
            num_rbf=args.num_rbf,
            num_bond_types=args.num_bond_types,
            bond_embedding_dim=args.bond_embedding_dim,
            irreps_edge_fused=args.irreps_edge,
            # EdgeDegreeEmbeddingNetwork参数
            avg_degree=args.avg_degree,
            # PositionalEncoding参数
            max_seq_len=args.max_seq_len)
        
        # 时间步嵌入层
        self.time_embedding = TimestepEmbedder(args.time_embed_dim)

        # E-DiT Block堆叠
        self.blocks = nn.ModuleList()

        # 将config中的Irreps字符串转换为Irreps对象
        irreps_node_hidden = Irreps(args.irreps_node_hidden)
        irreps_edge = Irreps(args.irreps_edge)
        irreps_final_node_feature = Irreps(args.irreps_final_node_feature)
        irreps_final_edge_feature = Irreps(args.irreps_final_edge_feature)
        irreps_node_attr = Irreps(args.irreps_node_attr)
        irreps_edge_attr_type = Irreps(args.irreps_edge_attr_type)
        irreps_sh = Irreps(args.irreps_sh)
        irreps_head = Irreps(args.irreps_head)
        irreps_mlp_mid = Irreps(args.irreps_mlp_mid)
        fc_neurons_list = [args.num_rbf] + args.fc_neurons
        norm_layer = args.norm_layer

        for i in range(args.num_blocks):
            if i != (args.num_blocks - 1):
                irreps_block_node_output = irreps_node_hidden
                irreps_block_edge_output = irreps_edge
            else:
                irreps_block_node_output = irreps_final_node_feature
                irreps_block_edge_output = irreps_final_edge_feature
            block = E_DiT_Block(
                max_radius=args.rbf_cutoff,
                number_of_basis = args.num_rbf,
                # 节点和边的Irreps定义
                irreps_node_input=irreps_node_hidden,
                irreps_node_attr=irreps_node_attr,
                irreps_edge_attr=irreps_sh,  # 用于更新节点的几何边特征
                irreps_node_output=irreps_block_node_output,  
                irreps_edge_input=irreps_edge,
                irreps_edge_output=irreps_block_edge_output,  
                irreps_edge_attr_type=irreps_edge_attr_type,
                edge_update_hidden_dim=args.edge_update_hidden_dim, 
                # 注意力机制参数
                fc_neurons=fc_neurons_list,
                irreps_head=irreps_head,
                num_heads=args.num_heads,
                # FNN 参数
                irreps_mlp_mid=irreps_mlp_mid,
                # 归一化和嵌入参数
                time_embedding_dim=args.time_embed_dim,
                norm_layer=norm_layer,
                # 正则化参数
                alpha_drop=args.alpha_drop,
                proj_drop=args.proj_drop,
                drop_path_rate=args.drop_path_rate,
                # 其他可选参数
                rescale_degree=args.rescale_degree,
                nonlinear_message=args.nonlinear_message,
                irreps_pre_attn=args.irreps_pre_attn
            )
            self.blocks.append(block)
            
        # 最终层归一化
        self.final_norm = get_norm_layer(norm_layer)(
            irreps=irreps_final_node_feature,
            time_embedding_dim=args.time_embed_dim
        )
        self.final_edge_norm = get_norm_layer(norm_layer)(
            irreps=irreps_final_edge_feature,
            time_embedding_dim=args.time_embed_dim
        )
        # 多任务输出头
        self.output_heads = MultiTaskHead(
            irreps_final_node_feature=irreps_final_node_feature,
            irreps_final_edge_feature=irreps_final_edge_feature,
            args=args
        )

    def forward(self,
                data,
                t: torch.Tensor,
                target_node_mask: torch.Tensor,
                target_edge_mask: torch.Tensor) -> dict:
        
        # 输入嵌入，将原始数据Lifting到等变空间
        embedded_inputs = self.embedding_layer(data)
        h = embedded_inputs["node_input"]
        e = embedded_inputs["edge_input"]
        pos = data.pos

        # 检查并创建batch索引
        if hasattr(data, 'batch') and data.batch is not None:
            batch = data.batch
        else:
            # 创建一个形状为 [num_nodes] 的张量，所有元素都是0
            batch = torch.zeros(data.num_nodes, dtype=torch.long, device=data.x.device)
        embedded_inputs['batch'] = batch
        
        # 依次通过所有E-DiT Block，进行深度特征提纯
        for i, block in enumerate(self.blocks):
            # 在每次调用前，更新字典中的节点和边特征为上一轮的输出
            embedded_inputs['node_input'] = h
            embedded_inputs['edge_input'] = e
            embedded_inputs['pos'] = pos

            # 将 target_node_mask 添加到传递给 block 的参数字典中
            embedded_inputs['target_node_mask'] = target_node_mask

            # 使用字典解包 ** 来传递所有参数，并单独传递时间步 t
            h, e, pos = block(t=t, **embedded_inputs)

        # 最终层归一化
        h = self.final_norm(h, t, batch)
        edge_batch = batch[data.edge_index[0]]
        e = self.final_edge_norm(e, t, edge_batch)

        r_t_final = pos 

        # 调用多任务输出头预测
        predictions = self.output_heads(
            h_final=h,
            e_final=e,
            r_t_final=r_t_final,
            target_node_mask=target_node_mask,  # 直接使用传入的原子掩码
            target_edge_mask=target_edge_mask  # 直接使用传入的边掩码
        )

        return predictions