# e_dit_network.py

import torch
import torch.nn as nn
from e3nn import o3
from e3nn.o3 import Irreps
from .tensor_product_rescale import LinearRS, FullyConnectedTensorProductRescale
from .mul_head_graph_attention import get_norm_layer
from .layer_norm import TimestepEmbedder
from torch_scatter import scatter_sum

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

        # 提取标量维度用于后续 MLP
        self.irreps_node_scalar = Irreps([(mul, ir) for mul, ir in self.irreps_node_in if ir.l == 0])
        num_node_scalar = self.irreps_node_scalar.dim
        num_edge_scalar = self.irreps_edge_in.dim

        out_drop_prob = args.out_drop

        # --- (1) 原子类型预测头 (Atom Type Head) ---
        # 创建一个投影层，用于从完整的节点特征中安全地提取标量部分
        # self.irreps_node_scalar = Irreps([(mul, ir) for mul, ir in self.irreps_node_in if ir.l == 0])
        self.node_scalar_proj = LinearRS(self.irreps_node_in, self.irreps_node_scalar)

        self.atom_type_head = nn.Sequential(
            nn.Linear(self.irreps_node_scalar.dim, args.hidden_dim),
            nn.SiLU(),
            nn.Dropout(out_drop_prob),
            nn.Linear(args.hidden_dim, args.num_atom_types)
        )

        # --- (2) 坐标预测头 (Coordinate Head) ---
        # 完全借鉴 PosUpdate 的思想来计算 delta_pos
        # a. 两个独立的 MLP，用于将起始/终止节点的标量特征投影到边空间
        #    输入是节点的标量部分，输出维度可以是一个超参数，比如 hidden_dim
        self.left_lin_edge = nn.Linear(num_node_scalar, args.hidden_dim)
        self.right_lin_edge = nn.Linear(num_node_scalar, args.hidden_dim)

        # b. 一个 MLP (edge_lin)，用于融合所有信息并计算最终的标量“力”
        #    输入维度 = 边标量特征 + 节点交互特征 (hidden_dim)
        edge_lin_input_dim = num_edge_scalar + args.hidden_dim
        self.edge_lin = nn.Sequential(
            nn.Linear(edge_lin_input_dim, args.hidden_dim),
            nn.SiLU(),
            nn.Linear(args.hidden_dim, 1) # 输出一个标量 (weight_edge)
        )

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
                r_t_final: torch.Tensor,
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
        row, col = edge_index
        
        # a. 提取所有节点的标量特征，用于坐标预测
        h_final_scalar = self.node_scalar_proj(h_final)

        # b. 投影节点特征到边空间
        left_feat = self.left_lin_edge(h_final_scalar[row])
        right_feat = self.right_lin_edge(h_final_scalar[col])
        
        # c. 创建节点交互特征
        node_interaction_feat = left_feat * right_feat  # 元素级乘法
        
        # d. 融合边特征和节点交互特征，计算标量“力”
        edge_lin_input = torch.cat([e_final, node_interaction_feat], dim=-1)
        weight_edge = self.edge_lin(edge_lin_input) # shape: [num_edges, 1]
        
        # e. 计算几何信息
        relative_vec = r_t_final[row] - r_t_final[col]
        distance = torch.linalg.norm(relative_vec, dim=-1, keepdim=True) + 1e-8
        
        # f. 计算方向性的、带距离衰减的力向量
        force_edge = weight_edge * (relative_vec / distance) / (distance + 1.0)

        # g. 聚合得到每个节点的总位移 delta_pos
        delta_pos_all_nodes = scatter_sum(force_edge, row, dim=0, dim_size=h_final.shape[0])

        # h. [关键] 将更新量加到原始的加噪坐标上，得到对 x₀ 的预测
        predicted_r0_all_nodes = r_t_final + delta_pos_all_nodes

        # i. 只保留目标节点的预测结果
        predicted_r0 = predicted_r0_all_nodes[target_node_mask]


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

        # print(f"EmbeddingLayer output shapes -- h: {h.shape}, e: {e.shape}")

        # --- 关键修改：检查并创建 batch 索引 ---
        # 如果 data.batch 不存在 (对于单个Data对象，它就是None)，
        # 我们就为它创建一个全零的 batch 索引张量。
        if hasattr(data, 'batch') and data.batch is not None:
            batch = data.batch
        else:
            # 创建一个形状为 [num_nodes] 的张量，所有元素都是0
            batch = torch.zeros(data.num_nodes, dtype=torch.long, device=data.x.device)
    
        # 将正确的 batch 索引也放入 embedded_inputs 字典中
        embedded_inputs['batch'] = batch
        # ----------------------------------------



        # print('h:', h)
        # print('e before edit:', e)
        # t_embed = self.time_embedding(t)  #未使用，归一层中自带时间嵌入
        
        # 步骤2：依次通过所有E-DiT Block，进行深度特征提纯
        # for block in self.blocks:
        for i, block in enumerate(self.blocks):
            # 在每次调用前，更新字典中的节点和边特征为上一轮的输出
            embedded_inputs['node_input'] = h
            embedded_inputs['edge_input'] = e

            # --- 添加调试打印 ---
            # print(f"\n--- Before Block {i} ---")
            # print(f"  h shape: {h.shape}")
            # print(f"  e shape: {e.shape}")
            # print(f"  batch tensor: {embedded_inputs['batch']}")

            # 使用字典解包 ** 来传递所有参数，并单独传递时间步 t
            h, e = block(t=t, **embedded_inputs)

            # print(f"--- After Block {i} ---")
            # print(f"  h shape: {h.shape}")
            # print(f"  e shape: {e.shape}")
            # --------------------

            # print('h after block:', h)
            # print('e after block:', e)
        # print('h before norm:', h)
        # print('e before norm:', e)
        # 步骤3：最终层归一化
        h = self.final_norm(h, t, batch)
        edge_batch = batch[data.edge_index[0]]
        e = self.final_edge_norm(e, t, edge_batch)
        # print('h after edit:', h.shape)
        # print('e after edit:', e.shape)

        r_t_final = data.pos 

        # 步骤4：调用多任务输出头
        predictions = self.output_heads(
            h_final=h,
            e_final=e,
            r_t_final=r_t_final,
            target_node_mask=target_node_mask,  # 直接使用传入的原子掩码
            target_edge_mask=target_edge_mask,  # 直接使用传入的边掩码
            edge_index=data.edge_index  # 传入完整的边索引
        )
        return predictions