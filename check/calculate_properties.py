import torch
import pickle
import numpy as np
import os
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import QED

# --- 配置和映射 (保持不变) ---
# 这些常量用于将 PyTorch Geometric 图数据转换为 RDKit 分子对象
ATOM_MAP = ['H', 'C', 'N', 'O', 'F']
BOND_TYPE_MAP = {
    0: Chem.rdchem.BondType.SINGLE,
    1: Chem.rdchem.BondType.DOUBLE,
    2: Chem.rdchem.BondType.TRIPLE,
    3: Chem.rdchem.BondType.AROMATIC
}
ABSORBING_ATOM_TYPE_INDEX = 5
NO_BOND_EDGE_TYPE_INDEX = 4

# --- 核心转换函数 (保持不变) ---
# 这个函数负责将图数据转换为 RDKit 可以理解的分子结构
def pyg_to_rdkit_mol(pyg_data):
    """
    将 PyTorch Geometric (PyG) 图数据对象转换为 RDKit 分子对象。
    """
    try:
        mol = Chem.RWMol()
        node_map = {}
        atom_features = pyg_data.x.cpu().numpy()
        atom_types_idx = np.argmax(atom_features, axis=1)

        # 1. 添加原子
        for i, atom_idx in enumerate(atom_types_idx):
            # 忽略用于图生成的特殊“吸收”节点
            if atom_idx != ABSORBING_ATOM_TYPE_INDEX:
                atom_symbol = ATOM_MAP[atom_idx]
                mol.AddAtom(Chem.Atom(atom_symbol))
                node_map[i] = len(node_map)

        # 如果没有有效的原子，则返回 None
        if not node_map:
            return None

        # 2. 添加化学键
        edge_index = pyg_data.edge_index.cpu().numpy()
        edge_attrs = pyg_data.edge_attr.cpu().numpy()
        bond_types_idx = np.argmax(edge_attrs, axis=1)

        for i in range(edge_index.shape[1]):
            u, v = edge_index[:, i]
            bond_type_idx = bond_types_idx[i]
            
            # 确保原子和键都有效
            if u in node_map and v in node_map and bond_type_idx != NO_BOND_EDGE_TYPE_INDEX:
                rdkit_u, rdkit_v = node_map[u], node_map[v]
                # 避免重复添加键
                if mol.GetBondBetweenAtoms(rdkit_u, rdkit_v) is None:
                    bond_type = BOND_TYPE_MAP.get(bond_type_idx)
                    if bond_type:
                        mol.AddBond(rdkit_u, rdkit_v, bond_type)

        # 3. 清理和验证分子
        final_mol = mol.GetMol()
        Chem.SanitizeMol(final_mol) # 检查化合价等化学合理性
        return final_mol
    except Exception:
        # 如果在转换或验证过程中出现任何错误，则返回 None
        return None

# --- 主函数 (仅计算 QED) ---
def calculate_qed(pkl_path):
    """
    从 pickle 文件加载分子数据，并计算所有有效分子的平均 QED 分数。
    """
    print(f"[*] 正在从 '{pkl_path}' 加载生成的分子...")
    try:
        with open(pkl_path, 'rb') as f:
            generated_data = pickle.load(f)
    except FileNotFoundError:
        print(f"[!] 错误: 文件 '{pkl_path}' 未找到。")
        return

    print(f"[✓] 加载成功，共 {len(generated_data)} 个分子。")

    qed_scores = []
    valid_molecule_count = 0
    
    print("\n--- 开始逐个计算 QED 分数 (仅对有效分子) ---")
    for data in tqdm(generated_data, desc="正在处理分子"):
        mol = pyg_to_rdkit_mol(data)
        
        # 只对成功转换并验证的分子进行计算
        if mol:
            valid_molecule_count += 1
            qed_score = QED.qed(mol)
            qed_scores.append(qed_score)

    print("\n--- 评估总结 ---")
    print(f"总处理分子数: {len(generated_data)}")
    print(f"有效分子数 (用于计算): {valid_molecule_count}")

    if valid_molecule_count > 0:
        avg_qed = np.mean(qed_scores)
        print("-" * 30)
        print(f"平均 QED 分数: {avg_qed:.4f}") # 提高一点精度
        print("-" * 30)
    else:
        print("\n[!] 没有找到有效的分子，无法计算平均分数。")

if __name__ == '__main__':
    # 指定包含生成的分子数据的文件路径
    generated_molecules_path = 'generated_molecules_from_best_model.pkl'
    calculate_qed(generated_molecules_path)