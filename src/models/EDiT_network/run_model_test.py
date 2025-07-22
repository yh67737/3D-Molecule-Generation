# 文件名: run_model_test.py

import torch
import torch.nn as nn
from e3nn.o3 import Irreps
from torch_geometric.data import Data, Batch
import os
from mul_head_graph_attention import get_norm_layer

from parser_data import create_parser
from e_dit_network import E_DiT_Network


# 如果上面的导入报错，请确保您的文件夹结构正确，
# 或者将所有类定义都放在一个文件中进行测试。


# =============================================================================
# 1. 定义一个完整的、用于测试的config字典
# =============================================================================
def get_test_config():
    """返回一个用于测试的、包含所有必要参数的配置字典"""
    config = {
        # --- 模型整体参数 ---
        'num_blocks': 6,  # 堆叠的E-DiT Block 的数量
        'num_heads': 4,  # 图注意力机制中的多头注意力头数
        'L_max': 2,  # 球谐函数的最高阶数

        # --- Irreps 定义 ---
        'irreps_node_hidden': '128x0e+64x1o+32x2e',  # Block输入的节点特征维度
        'irreps_edge': '128x0e+64x1o+32x2e',  # Block输入的边特征维度
        'irreps_node_attr': '6x0e',  # 节点原子类型one-hot
        'irreps_edge_attr_type': '5x0e',  # 边化学键one-hot
        'irreps_sh': '1x0e+1x1e+1x2e',  # 球谐函数的Irreps (L_max=1)
        'irreps_head': '32x0e+16x1o+8x2e',  # 单个注意力头的Irreps结构
        'irreps_mlp_mid': '384x0e+192x1o+96x2e',  # FFN中间层的Irreps

        # --- 嵌入层和径向函数参数 ---
        'num_atom_types': 6, #数据集中原子类型数
        'num_bond_types': 5, #数据集中边类型数
        'node_embedding_hidden_dim': 64,  # NodeEmbeddingNetwork中MLP的维度
        'bond_embedding_dim': 64,  # EdgeEmbeddingNetwork中MLP的维度
        'num_rbf': 128, #径向基函数（用于编码距离）的基函数数量
        'rbf_cutoff': 5.0, #径向基函数（用于编码距离）的截断半径
        'fc_neurons': [64, 64],
        'avg_degree': 10.0,  # 数据集平均度，在rescale_degree里使用

        # --- 多头注意力参数 ---
        'nonlinear_message': False, #计算注意力值向量时是否加入非线性激活步骤
        'rescale_degree': False, #注意力模块的最后，聚合到每个节点的特征向量是否乘以该节点的度
        'irreps_pre_attn': None, #在核心注意力计算之前是否对节点特征进行一次可选的线性变换

        # --- 时间步嵌入和归一化 ---
        'time_embed_dim': 128, #时间步`t`被嵌入后的向量维度
        'norm_layer': 'layer', #归一化类型选择
        'edge_update_hidden_dim': 64,  # EdgeUpdateNetwork中MLP的维度

        # --- 正则化 ---
        'alpha_drop': 0.2, #注意力中的dropout
        'proj_drop': 0.0, #最终投影层中的dropout
        'drop_path_rate': 0.0, #随机路径中的dropout
        'out_drop': 0.0, #多头输出中的dropout

        # --- 输出头MLP参数 ---
        'hidden_dim': 128

    }
    return config


# =============================================================================
# 2. 构造一个符合您数据格式的虚拟图输入
# =============================================================================
def create_sample_graph(num_nodes=5, num_edges=8, args=None):
    """创建一个符合您数据格式的单个PyG Data对象"""
    if args is None:
        raise ValueError("args 对象必须被提供。")

    atom_types = torch.randint(0, args.num_atom_types, (num_nodes,))
    x = nn.functional.one_hot(atom_types, num_classes=args.num_atom_types).float()

    pos = torch.randn(num_nodes, 3)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))

    bond_types = torch.randint(0, args.num_bond_types, (num_edges,))
    edge_attr = nn.functional.one_hot(bond_types, num_classes=args.num_bond_types).float()

    pring_out = torch.randint(0, 2, (num_nodes, 1)).float()
    is_last = torch.zeros(num_nodes, 1, dtype=torch.bool)
    is_last[-1] = True

    return Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr, pring_out=pring_out, is_last=is_last)

# =============================================================================
# 3. 主测试函数
# =============================================================================
def main():
    print("--- 开始 E-DiT Network 结构与数据流验证 ---")

    parser = create_parser()
    args = parser.parse_args([])

    print("1. 构造虚拟输入数据...")

    graph1 = create_sample_graph(num_nodes=5, num_edges=8, args=args)
    graph2 = create_sample_graph(num_nodes=7, num_edges=12, args=args)
    test_batch = Batch.from_data_list([graph1, graph2])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   使用设备: {device}")

    model = E_DiT_Network(args).to(device)
    test_batch = test_batch.to(device)
    timesteps = torch.randint(0, 1000, (test_batch.num_graphs,), device=device)

    print("\n[可选] 正在验证零初始化策略...")

    try:
        # 验证节点 FFN
        ffn_last_layer_weights = model.blocks[0].node_ffn.fctp_2.tp.weight
        is_ffn_zero_init = torch.all(ffn_last_layer_weights == 0)
        print(f" - 节点FFN最后一个线性层是否被零初始化? {is_ffn_zero_init}")
        assert is_ffn_zero_init

        # 验证边更新网络的权重
        edge_updater_weights = model.blocks[0].edge_updater.edge_update_mlp.tp.weight
        is_edge_updater_zero_init = torch.all(edge_updater_weights == 0)
        print(f" - 边更新网络MLP是否被零初始化? {is_edge_updater_zero_init}")
        assert is_edge_updater_zero_init

        # 验证边的 FFN
        edge_ffn_weights = model.blocks[0].edge_ffn.fctp_2.tp.weight
        is_edge_ffn_zero_init = torch.all(edge_ffn_weights == 0)
        print(f" - 边FFN最后一个线性层是否被零初始化? {is_edge_ffn_zero_init}")
        assert is_edge_ffn_zero_init

        print("   零初始化验证通过。")
    except Exception as e:
        print(f"   零初始化验证失败: {e}")

    # --- 创建掩码 ---
    target_node_mask = test_batch.is_last.squeeze(-1)
    node_indices = torch.arange(test_batch.num_nodes, device=device)
    target_node_indices = node_indices[target_node_mask]
    edge_src, edge_dst = test_batch.edge_index
    is_src_target = torch.isin(edge_src, target_node_indices)
    is_dst_target = torch.isin(edge_dst, target_node_indices)
    target_edge_mask = is_src_target | is_dst_target

    print("\n2. 模型实例化完成，准备执行前向传播...")
    print(
        f"   测试批次包含 {test_batch.num_graphs} 个图, 总计 {test_batch.num_nodes} 个节点, {test_batch.num_edges} 条边。")
    print(f"   目标节点数量: {target_node_mask.sum().item()} 个")
    print(f"   目标边数量: {target_edge_mask.sum().item()} 条")

    model.eval()
    try:
        with torch.no_grad():
            predictions = model(test_batch, timesteps, target_node_mask, target_edge_mask)
            print("\n3. 前向传播成功！")
            print("\n4. 检验输出结果的格式和形状...")

            assert isinstance(predictions, dict), "输出应该是一个字典"
            print(f"   模型输出键: {list(predictions.keys())}")

            atom_logits = predictions['atom_type_logits']
            num_target_nodes = target_node_mask.sum().item()

            print(f"   原子类型logits形状: {atom_logits.shape} (应为 [{num_target_nodes}, {args.num_atom_types}])")
            assert atom_logits.shape == (num_target_nodes, args.num_atom_types)

            coord_preds = predictions['predicted_r0']
            print(f"   预测坐标形状: {coord_preds.shape} (应为 [{num_target_nodes}, 3])")
            assert coord_preds.shape == (num_target_nodes, 3)

            bond_logits = predictions['bond_logits']
            num_target_edges = target_edge_mask.sum().item()

            print(
                f"   化学键logits形状: {bond_logits.shape} (应为 [{num_target_edges}, {args.num_bond_types}])")
            assert bond_logits.shape == (num_target_edges, args.num_bond_types)

            print("\n--- 所有检查通过！模型主体结构和数据流已按新接口正确对齐。 ---")

    except Exception as e:
        print(f"\n!!!!!! 测试过程中发生错误 !!!!!!")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
