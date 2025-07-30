# e_dit_network.py

import torch
import torch.nn as nn
from e3nn import o3
from e3nn.o3 import Irreps
from .tensor_product_rescale import LinearRS, FullyConnectedTensorProductRescale
from .mul_head_graph_attention import get_norm_layer
from .layer_norm import TimestepEmbedder

# 从您已经创建的文件中导入核心组件
from .input_embedding import InputEmbeddingLayer
from .e_dit_block import E_DiT_Block


class MultiTaskHead(nn.Module):
    """
    多任务输出头。
    """

    def __init__(self, irreps_node_in: str, irreps_edge_in: str, args):
        super().__init__()
        self.irreps_node_in = Irreps(irreps_node_in)
        self.irreps_edge_in = Irreps(irreps_edge_in)

        out_drop_prob = args.out_drop

        # --- (1) 原子类型预测头 (Atom Type Head) ---
        # 创建一个投影层，用于从完整的节点特征中安全地提取标量部分
        self.irreps_node_scalar = Irreps([(mul, ir) for mul, ir in self.irreps_node_in if ir.l == 0])
        self.node_scalar_proj = LinearRS(self.irreps_node_in, self.irreps_node_scalar)

        self.atom_type_head = nn.Sequential(
            nn.Linear(self.irreps_node_scalar.dim, args.hidden_dim),
            nn.SiLU(),
            nn.Dropout(out_drop_prob),
            nn.Linear(args.hidden_dim, args.num_atom_types)
        )

        # --- (2) 坐标预测头 (Coordinate Head) ---
        # 创建一个投影层，用于提取矢量(L=1)部分
        irreps_vector_in = Irreps(
            [(mul, ir) for mul, ir in self.irreps_node_in if ir.l == 1 and ir.p == -1])  # 仅保留 '1o'
        self.node_vector_proj = LinearRS(irreps_in=self.irreps_node_in, irreps_out=irreps_vector_in)
        self.coord_dropout = nn.Dropout(out_drop_prob)
        self.coord_head = LinearRS(irreps_in=irreps_vector_in, irreps_out='1x1o')

        # --- (3) 化学键预测头 (Bond Prediction Head) ---
        # a. 用于节点间等变交互的张量积，只取标量输出
        self.bond_head_tp = FullyConnectedTensorProductRescale(
            self.irreps_node_in, self.irreps_node_in, "0e"
        )

        # b. 接收融合后信息的最终MLP
        num_edge_scalar = self.irreps_edge_in.dim
        mlp_input_dim = self.bond_head_tp.irreps_out.dim + num_edge_scalar
        self.bond_head_mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, args.hidden_dim),
            nn.SiLU(),
            nn.Dropout(out_drop_prob),
            nn.Linear(args.hidden_dim, args.num_bond_types)
        )

    def forward(self,
                h_final: torch.Tensor,
                e_final: torch.Tensor,
                target_node_mask: torch.Tensor,
                target_edge_mask: torch.Tensor,
                edge_index: torch.Tensor) -> dict:
        """
        Args:
            h_final (torch.Tensor): 最终节点特征张量。
            e_final (torch.Tensor): 最终边特征张量。
            target_node_mask (torch.Tensor): 布尔掩码，标记目标节点。
            edge_index_to_predict (torch.Tensor): 需要预测的边的索引。
        """

        # --- 1. 原子类型预测 ---
        target_node_features = h_final[target_node_mask]
        # 首先提取标量部分
        scalar_features = self.node_scalar_proj(target_node_features)
        atom_type_logits = self.atom_type_head(scalar_features)

        # --- 2. 坐标预测 (预测 r0_hat) ---
        vector_features = self.node_vector_proj(target_node_features)
        vector_features = self.coord_dropout(vector_features)
        predicted_r0 = self.coord_head(vector_features)

        # --- 3. 化学键预测 ---
        # a. 使用 edge_mask 筛选出需要预测的边
        edge_index_to_predict = edge_index[:, target_edge_mask]
        row, col = edge_index_to_predict

        # b. 节点交互：获取目标边两端节点的特征进行张量积
        interaction_scalars = self.bond_head_tp(h_final[row], h_final[col])

        # c. 提取目标边的特征
        target_edge_features = e_final[target_edge_mask]

        # d. 融合信息
        bond_head_input = torch.cat([interaction_scalars, target_edge_features], dim=-1)

        # e. MLP解码
        bond_logits = self.bond_head_mlp(bond_head_input)

        return {
            'atom_type_logits': atom_type_logits,
            'predicted_r0': predicted_r0,
            'bond_logits': bond_logits,
        }

class E_DiT_Network(nn.Module):
    """
    最终的E-DiT网络主体，负责组装和调用所有模块。
    """
    def __init__(self, args):
        super().__init__()
        self.args = args

        # (1) 输入嵌入层
        self.embedding_layer = InputEmbeddingLayer(
            # NodeEmbeddingNetwork 参数
            irreps_node_embedding=args.irreps_node_hidden,
            num_atom_types=args.num_atom_types,
            node_embedding_hidden_dim=args.node_embedding_hidden_dim,

            # EdgeEmbeddingNetwork 参数
            irreps_sh=args.irreps_sh,
            max_radius=args.rbf_cutoff,
            num_rbf=args.num_rbf,
            num_bond_types=args.num_bond_types,
            bond_embedding_dim=args.bond_embedding_dim,
            irreps_edge_fused=args.irreps_edge,

            # EdgeDegreeEmbeddingNetwork 参数
            avg_degree=args.avg_degree
        )
        
        # (2) 时间步嵌入层
        self.time_embedding = TimestepEmbedder(args.time_embed_dim)

        # (3) E-DiT Block堆叠
        self.blocks = nn.ModuleList()

        # 将config中的Irreps字符串转换为Irreps对象
        irreps_node_hidden = Irreps(args.irreps_node_hidden)
        irreps_edge = Irreps(args.irreps_edge)
        irreps_node_attr = Irreps(args.irreps_node_attr)
        irreps_edge_attr_type = Irreps(args.irreps_edge_attr_type)
        irreps_sh = Irreps(args.irreps_sh)
        irreps_head = Irreps(args.irreps_head)
        irreps_mlp_mid = Irreps(args.irreps_mlp_mid)
        fc_neurons_list = [args.num_rbf] + args.fc_neurons
        norm_layer = args.norm_layer

        for _ in range(args.num_blocks):
            # 实例化 E_DiT_Block 并传入所有必要的参数
            block = E_DiT_Block(
                # 节点和边的Irreps定义
                irreps_node_input=irreps_node_hidden,
                irreps_node_attr=irreps_node_attr,
                irreps_edge_attr=irreps_sh,  # 用于更新节点的几何边特征
                irreps_node_output=irreps_node_hidden,  # Block的输出与输入Irreps保持一致

                irreps_edge_input=irreps_edge,
                irreps_edge_output=irreps_edge,  # 边的输出与输入Irreps也保持一致
                irreps_edge_attr_type=irreps_edge_attr_type,

                # 注意力机制参数
                fc_neurons=fc_neurons_list,
                irreps_head=irreps_head,
                num_heads=args.num_heads,

                # FNN 参数
                irreps_mlp_mid=irreps_mlp_mid,

                edge_update_hidden_dim=args.edge_update_hidden_dim,

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
            
        # (4) 最终层归一化
        self.final_norm = get_norm_layer(norm_layer)(
            irreps=irreps_node_hidden,
            time_embedding_dim=args.time_embed_dim
        )
        self.final_edge_norm = get_norm_layer(norm_layer)(
            irreps=irreps_edge,
            time_embedding_dim=args.time_embed_dim
        )
        # (5) 多任务输出头
        self.output_heads = MultiTaskHead(
            irreps_node_in=irreps_node_hidden,
            irreps_edge_in=irreps_edge,
            args=args
        )

    def forward(self,
                data,
                t: torch.Tensor,
                target_node_mask: torch.Tensor,
                target_edge_mask: torch.Tensor) -> dict:
        # 步骤1：输入嵌入，将原始数据Lifting到等变空间
        embedded_inputs = self.embedding_layer(data)
        h = embedded_inputs["node_input"]
        e = embedded_inputs["edge_input"]
        # print('h:', h)
        # print('e before edit:', e)
        # t_embed = self.time_embedding(t)  #未使用，归一层中自带时间嵌入
        
        # 步骤2：依次通过所有E-DiT Block，进行深度特征提纯
        for block in self.blocks:
            # 在每次调用前，更新字典中的节点和边特征为上一轮的输出
            embedded_inputs['node_input'] = h
            embedded_inputs['edge_input'] = e

            # 使用字典解包 ** 来传递所有参数，并单独传递时间步 t
            h, e = block(t=t, **embedded_inputs)
            # print('h after block:', h)
            # print('e after block:', e)
        # print('h before norm:', h)
        # print('e before norm:', e)
        # 步骤3：最终层归一化
        h = self.final_norm(h, t, data.batch)
        edge_batch = data.batch[data.edge_index[0]]
        e = self.final_edge_norm(e, t, edge_batch)
        # print('h after edit:', h.shape)
        # print('e after edit:', e.shape)

        # 步骤4：调用多任务输出头
        predictions = self.output_heads(
            h_final=h,
            e_final=e,
            target_node_mask=target_node_mask,  # 直接使用传入的原子掩码
            target_edge_mask=target_edge_mask,  # 直接使用传入的边掩码
            edge_index=data.edge_index  # 传入完整的边索引
        )
        return predictions