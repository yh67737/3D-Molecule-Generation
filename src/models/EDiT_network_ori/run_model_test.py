# 文件名: run_model_test.py

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
import os
import sys
from argparse import Namespace
import e3nn  # 确保 e3nn 被导入

# 导入您的模型。请确保您的项目路径已正确设置。
from src.models.EDiT_network_ori.e_dit_network import E_DiT_Network

try:
    torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
    if sys.version_info >= (3, 8) and os.path.isdir(torch_lib_path):
        os.add_dll_directory(torch_lib_path)
except (AttributeError, TypeError):
    pass


# =============================================================================
# 1. 定义测试配置 (保持不变)
# =============================================================================
def get_test_config():
    """返回一个用于测试的、包含所有必要参数的配置字典"""
    config = {
        'num_blocks': 6, 'num_heads': 4, 'L_max': 2,
        'irreps_node_hidden': '128x0e+64x1o+32x2e', 'irreps_edge': '128x0e+64x1o+32x2e',
        'irreps_node_attr': '6x0e', 'irreps_edge_attr_type': '5x0e',
        'irreps_sh': '1x0e+1x1e+1x2e', 'irreps_head': '32x0e+16x1o+8x2e',
        'irreps_mlp_mid': '384x0e+192x1o+96x2e', 'num_atom_types': 6, 'num_bond_types': 5,
        'node_embedding_hidden_dim': 64, 'bond_embedding_dim': 64,
        'num_rbf': 128, 'rbf_cutoff': 5.0, 'fc_neurons': [64, 64],
        'avg_degree': 10.0, 'nonlinear_message': False, 'rescale_degree': False,
        'irreps_pre_attn': None, 'time_embed_dim': 128, 'norm_layer': 'layer',
        'edge_update_hidden_dim': 64, 'alpha_drop': 0.2, 'proj_drop': 0.0,
        'drop_path_rate': 0.0, 'out_drop': 0.0, 'hidden_dim': 128
    }
    return config


# =============================================================================
# 2. 构造虚拟图输入 (保持不变)
# =============================================================================
def create_sample_graph(num_nodes=5, num_edges=8, args=None):
    """创建一个符合您数据格式的单个PyG Data对象"""
    if args is None: raise ValueError("args 对象必须被提供。")
    atom_types = torch.randint(0, args.num_atom_types, (num_nodes,))
    x = nn.functional.one_hot(atom_types, num_classes=args.num_atom_types).float()
    pos = torch.randn(num_nodes, 3)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    bond_types = torch.randint(0, args.num_bond_types, (num_edges,))
    edge_attr = nn.functional.one_hot(bond_types, num_classes=args.num_bond_types).float()
    is_last = torch.zeros(num_nodes, 1, dtype=torch.bool);
    is_last[-1] = True
    pring_out = torch.randint(0, 2, (num_nodes, 1)).float()

    return Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr,
                pring_out=pring_out, is_last=is_last)


# =============================================================================
# 3. 主测试函数 (核心修改)
# =============================================================================
def main():
    print("--- 开始 E-DiT Network 结构与数据流验证 ---")

    # --- 步骤 1: 准备配置和数据 ---
    config = get_test_config()
    args = Namespace(**config)
    print("1. 构造虚拟输入数据...")
    graph1 = create_sample_graph(num_nodes=5, num_edges=8, args=args)
    graph2 = create_sample_graph(num_nodes=7, num_edges=12, args=args)
    test_batch = Batch.from_data_list([graph1, graph2])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 步骤 2: 创建模型并移至设备 ---
    print("\n2. 正在实例化 E_DiT_Network...")
    model = E_DiT_Network(args)
    print(f"   将模型移动到设备: {device}...")
    model.to(device)  # 先移动设备！
    print("   模型已移动。")

    # 将数据也移至设备
    test_batch = test_batch.to(device)
    timesteps = torch.randint(0, 1000, (test_batch.num_graphs,), device=device)

    # --- 步骤 3: 在模型移动到设备后，执行正确的零初始化 ---
    print("\n3. 正在执行零初始化策略...")
    try:
        with torch.no_grad():
            for block in model.blocks:
                # --- 错误修正 1 & 2 的应用 ---
                # a) 节点路径: GraphAttention (处理 ParameterList)
                ga_proj = block.ga.proj
                if hasattr(ga_proj, 'weight'): torch.nn.init.constant_(ga_proj.weight, 0)
                if hasattr(ga_proj, 'bias') and ga_proj.bias is not None:
                    for b in ga_proj.bias:  # 遍历列表中的每个偏置张量
                        torch.nn.init.constant_(b, 0)

                # b) 节点路径: FFN (使用重锤方法)
                for param in block.node_ffn.parameters(): param.fill_(0)

                # c) 边路径: EdgeUpdateNetwork 的最终层
                for param in block.edge_updater.final_norm_and_transform.parameters(): param.fill_(0)

                # d) 边路径: FFN (使用重锤方法)
                for param in block.edge_ffn.parameters(): param.fill_(0)
        print("   零初始化执行完毕。")
    except Exception as e:
        print(f"   在执行零初始化时发生错误: {e}")
        import traceback;traceback.print_exc()

    # --- 步骤 4: 验证初始化是否成功 ---
    print("\n4. 正在验证零初始化结果...")
    try:
        # 验证边路径的 `final_norm_and_transform`
        final_norm_to_check = model.blocks[0].edge_updater.final_norm_and_transform
        is_init_1 = all(torch.all(p == 0) for p in final_norm_to_check.parameters())
        print(f" - EdgeUpdateNetwork 的最终层是否被零初始化? {is_init_1}")
        assert is_init_1

        # 验证边的 FFN
        edge_ffn_weights = model.blocks[0].edge_ffn.fctp_2.tp.weight
        is_edge_ffn_zero_init = torch.all(edge_ffn_weights == 0)
        print(f" - 边FFN最后一个线性层是否被零初始化? {is_edge_ffn_zero_init}")
        assert is_edge_ffn_zero_init

        print("   零初始化验证通过。")
    except Exception as e:
        print(f"   零初始化验证失败: {e}")
        import traceback;
        traceback.print_exc()

    # --- 步骤 5: 执行前向传播测试 ---
    # 创建掩码
    target_node_mask = test_batch.is_last.squeeze(-1)
    node_indices = torch.arange(test_batch.num_nodes, device=device)
    target_node_indices = node_indices[target_node_mask]
    edge_src, edge_dst = test_batch.edge_index
    is_src_target = torch.isin(edge_src, target_node_indices)
    is_dst_target = torch.isin(edge_dst, target_node_indices)
    target_edge_mask = is_src_target | is_dst_target

    print("\n5. 模型实例化与初始化完成，准备执行前向传播...")
    print(
        f"   测试批次包含 {test_batch.num_graphs} 个图, 总计 {test_batch.num_nodes} 个节点, {test_batch.num_edges} 条边。")

    model.eval()
    try:
        with torch.no_grad():
            # 您可以将您的诊断代码暂时放在模型定义中，以观察内部数据流
            predictions = model(test_batch, timesteps, target_node_mask, target_edge_mask)
            print("\n6. 前向传播成功！")
            # ... (后续的输出形状检查保持不变)

    except Exception as e:
        print(f"\n!!!!!! 测试过程中发生错误 !!!!!!")
        import traceback;
        traceback.print_exc()


if __name__ == '__main__':
    main()