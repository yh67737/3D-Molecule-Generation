import pickle
import torch
import numpy as np
from pathlib import Path

# --- 配置和映射 ---

# 1. 原子类型映射 (用于显示原子符号)
# 确保这个映射与您的训练数据一致
ATOM_MAP = ['H', 'C', 'N', 'O', 'F', 'Absorbing']

# 2. 键类型映射 (用于解析 edge_attr)
# 索引 -> 描述: 0=单, 1=双, 2=三, 3=芳香, 4=无键
BOND_TYPE_MAP = {
    0: 'Single',
    1: 'Double',
    2: 'Triple',
    3: 'Aromatic',
    4: 'No Bond'  # 假设索引4代表无键连接
}

# --- 主分析函数 ---

def analyze_edge_distances_and_types(pkl_path):
    """
    加载生成的分子数据，并详细列出每条边的类型和计算出的距离。

    Args:
        pkl_path (str): .pkl 文件的路径。
    """
    print(f"[*] 正在从 '{pkl_path}' 读取数据...")
    try:
        # 使用 pickle 加载数据
        with open(pkl_path, 'rb') as f:
            molecule_data_list = pickle.load(f)
    except FileNotFoundError:
        print(f"[!] 错误：文件 '{pkl_path}' 未找到。请检查路径是否正确。")
        return
    except Exception as e:
        print(f"[!] 加载文件时发生错误: {e}")
        return

    print(f"[✓] 读取成功！共找到 {len(molecule_data_list)} 个分子。\n")

    # --- 遍历每个生成的分子 ---
    for i, data in enumerate(molecule_data_list):
        print(f"================ Molecule {i+1} ================")
        
        # 提取所需数据
        try:
            positions = data.pos.cpu()
            edge_index = data.edge_index.cpu()
            edge_attrs_onehot = data.edge_attr.cpu()
            atom_features_onehot = data.x.cpu()
        except AttributeError as e:
            print(f"[!] 数据对象不完整，缺少属性: {e}。跳过此分子。")
            continue

        # 1. 解析原子类型，用于更清晰地显示
        atom_indices = torch.argmax(atom_features_onehot, dim=1)
        atom_symbols = [ATOM_MAP[idx.item()] for idx in atom_indices]

        # 2. 解析边的类型
        # edge_attrs_onehot 的形状是 [num_edges, num_bond_types]
        # 使用 argmax 获取每个边的类型索引
        bond_type_indices = torch.argmax(edge_attrs_onehot, dim=1)

        print(f"原子总数: {len(atom_symbols)}")
        print(f"边总数 (来自 edge_index): {edge_index.shape[1]}")
        print("-" * 40)
        print("详细边分析 (索引 | 原子对 | 预测类型 | 计算距离):")

        # 3. 遍历 edge_index 中的每一条边
        num_edges = edge_index.shape[1]
        for edge_idx in range(num_edges):
            # 获取边连接的两个原子的索引 (u, v)
            u = edge_index[0, edge_idx].item()
            v = edge_index[1, edge_idx].item()

            # 获取原子的符号表示
            symbol_u = atom_symbols[u] if u < len(atom_symbols) else 'N/A'
            symbol_v = atom_symbols[v] if v < len(atom_symbols) else 'N/A'

            # 获取原子的坐标
            pos_u = positions[u]
            pos_v = positions[v]

            # 计算欧氏距离
            distance = torch.dist(pos_u, pos_v).item()

            # 获取该边的键类型
            type_idx = bond_type_indices[edge_idx].item()
            bond_type_str = BOND_TYPE_MAP.get(type_idx, f"Unknown({type_idx})")

            # 打印结果
            # 格式化输出: 边序号 | 原子1(符号) <-> 原子2(符号) | 键类型 | 距离
            print(f"Edge {edge_idx:3d}: Atom {u}({symbol_u}) <-> Atom {v}({symbol_v}) | "
                  f"Type: {bond_type_str:<10} | Distance: {distance:.4f} Å")
        
        print("\n")


# --- 执行 ---
if __name__ == "__main__":
    # --- 用户配置区 ---
    # 1. 输入文件路径 (请修改为您的实际路径)
    PKL_FILE_PATH = "/root/autodl-tmp/molecular_generation_project/output/2025-09-07_13-25-53/generated_pyg/generated_molecules_from_best_model.pkl"

    # 2. 运行分析函数
    analyze_edge_distances_and_types(PKL_FILE_PATH)