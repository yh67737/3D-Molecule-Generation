import torch
import pickle
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import rdmolops

# --- 1. 配置和全局映射 (保持不变) ---
ATOM_MAP = ['H', 'C', 'N', 'O', 'F']
BOND_TYPE_MAP = {
    0: Chem.rdchem.BondType.SINGLE,
    1: Chem.rdchem.BondType.DOUBLE,
    2: Chem.rdchem.BondType.TRIPLE,
}
ABSORBING_ATOM_TYPE_INDEX = 5
NO_BOND_EDGE_TYPE_INDEX = 3

# --- 2. 核心辅助函数 ---

def pyg_to_rdkit_mol(pyg_data):
    """
    将单个 PyG Data 对象转换为 RDKit Mol 对象。
    如果分子无效 (无法转换或通过化学检查)，则返回 None。
    """
    try:
        mol = Chem.RWMol()
        node_map = {}
        atom_features = pyg_data.x.cpu().numpy()
        atom_types_idx = np.argmax(atom_features, axis=1)

        for i, atom_idx in enumerate(atom_types_idx):
            if atom_idx != ABSORBING_ATOM_TYPE_INDEX:
                atom_symbol = ATOM_MAP[atom_idx]
                rdkit_atom = Chem.Atom(atom_symbol)
                new_idx = mol.AddAtom(rdkit_atom)
                node_map[i] = new_idx

        if not node_map:
            return None

        edge_index = pyg_data.edge_index.cpu().numpy()
        edge_attrs = pyg_data.edge_attr.cpu().numpy()
        bond_types_idx = np.argmax(edge_attrs, axis=1)

        for i in range(edge_index.shape[1]):
            u, v = edge_index[:, i]
            bond_type_idx = bond_types_idx[i]
            
            if u in node_map and v in node_map and bond_type_idx != NO_BOND_EDGE_TYPE_INDEX:
                rdkit_u, rdkit_v = node_map[u], node_map[v]
                if mol.GetBondBetweenAtoms(rdkit_u, rdkit_v) is None:
                    bond_type = BOND_TYPE_MAP.get(bond_type_idx)
                    if bond_type:
                        mol.AddBond(rdkit_u, rdkit_v, bond_type)

        final_mol = mol.GetMol()
        Chem.SanitizeMol(final_mol) # 关键的化学正确性检查
        return final_mol

    except Exception:
        return None

def check_atom_stability(mol):
    """
    检查分子的原子价键是否稳定。
    """
    problems = rdmolops.DetectChemistryProblems(mol)
    return len(problems) == 0

# --- 3. 集成后的主功能函数 ---

def evaluate_and_save_molecules(generated_mols_list, training_mols_list, output_dir):
    """
    对生成的分子进行全面评估，并保存所有有效的分子。
    
    Args:
        generated_mols_list (list): 生成的 PyG Data 对象列表。
        training_mols_list (list): 训练集的 PyG Data 对象列表。
        output_dir (str): 保存有效 .mol 文件的目录路径。
    """
    print("\n--- 阶段 1: 准备新颖性评估的参考集 ---")
    training_smiles_set = set()
    for data in tqdm(training_mols_list, desc="Processing training set"):
        mol = pyg_to_rdkit_mol(data)
        if mol:
            smi = Chem.MolToSmiles(mol, isomericSmiles=False)
            training_smiles_set.add(smi)
    print(f"[✓] 训练集参考构建完毕，共 {len(training_smiles_set)} 个独特的分子结构。")

    print("\n--- 阶段 2: 处理生成的分子 (评估和保存) ---")
    total_generated = len(generated_mols_list)
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"[*] 有效的分子结构将被保存到 '{output_dir}' 文件夹中。")
    
    valid_mols = []
    valid_smiles = []
    atom_stable_count = 0
    
    for i, data in enumerate(tqdm(generated_mols_list, desc="Processing generated set")):
        mol = pyg_to_rdkit_mol(data)
        if mol:
            # 1. 收集有效分子信息用于评估
            valid_mols.append(mol)
            smi = Chem.MolToSmiles(mol, isomericSmiles=False)
            valid_smiles.append(smi)
            
            if check_atom_stability(mol):
                atom_stable_count += 1
            
            # 2. 保存有效的分子为 .mol 文件
            mol_filename = os.path.join(output_dir, f"valid_molecule_{i + 1}.mol")
            Chem.MolToMolFile(mol, mol_filename)
                
    # --- 阶段 3: 计算并打印最终评估报告 ---
    
    # 有效性 (Validity)
    validity = len(valid_mols) / total_generated if total_generated > 0 else 0
    
    # 分子稳定性 (Molecule Stability) - 定义为与有效性相同
    molecule_stability = validity
    
    # 原子稳定性 (Atom Stability) - 基于有效分子的比例
    atom_stability = atom_stable_count / len(valid_mols) if len(valid_mols) > 0 else 0

    # 独特性 (Uniqueness)
    unique_smiles_set = set(valid_smiles)
    uniqueness = len(unique_smiles_set) / len(valid_smiles) if len(valid_smiles) > 0 else 0

    # 新颖性 (Novelty)
    novel_count = len(unique_smiles_set - training_smiles_set)
    novelty = novel_count / len(unique_smiles_set) if len(unique_smiles_set) > 0 else 0

    print("\n\n--- 最终评估结果 ---")
    print(f"总生成数量: {total_generated}")
    print(f"有效分子数: {len(valid_mols)}")
    print(f"独特分子数: {len(unique_smiles_set)}")
    print(f"新颖分子数: {novel_count}")
    print("-" * 30)
    print("核心指标:")
    print(f"  - 有效性 (Validity): {validity:.2%}")
    print(f"  - 独特性 (Uniqueness): {uniqueness:.2%}")
    print(f"  - 新颖性 (Novelty): {novelty:.2%}")
    print("-" * 30)
    print("QM9 稳定性指标:")
    print(f"  - 分子稳定性 (Molecule Stability): {molecule_stability:.2%}")
    print(f"  - 原子稳定性 (Atom Stability): {atom_stability:.2%}")
    print("-" * 30)
    print(f"[✓] 评估完毕！所有 {len(valid_mols)} 个有效分子的 .mol 文件已保存在 '{output_dir}' 目录中。\n")


if __name__ == '__main__':
    # --- 用户配置区 ---
    # 输入文件
    TRAINING_DATASET_PATH = 'src/data/gdb9_pyg_dataset_fc_no_aromatic_removed.pt'
    GENERATED_MOLECULES_PATH = 'output/2025-10-12_01-06-18/generated_pyg/generated_molecules_from_best_model.pkl'
    
    # 输出目录
    OUTPUT_DIRECTORY_FOR_VALID_MOLS = 'valid_molecules_output'
    
    # --- 脚本执行区 ---
    print("--- 开始执行分子评估与保存脚本 ---")
    
    try:
        # 加载训练数据
        print(f"[*] 正在加载训练数据集: '{TRAINING_DATASET_PATH}'...")
        training_data = torch.load(TRAINING_DATASET_PATH, map_location='cpu', weights_only=False)
        print(f"[✓] 加载成功，共 {len(training_data)} 个分子。")

        # 加载生成数据
        print(f"[*] 正在加载生成的分子: '{GENERATED_MOLECULES_PATH}'...")
        with open(GENERATED_MOLECULES_PATH, 'rb') as f:
            generated_data = pickle.load(f)
        print(f"[✓] 加载成功，共 {len(generated_data)} 个分子。")

        # 运行主函数
        evaluate_and_save_molecules(
            generated_mols_list=generated_data,
            training_mols_list=training_data,
            output_dir=OUTPUT_DIRECTORY_FOR_VALID_MOLS
        )
        
    except FileNotFoundError as e:
        print(f"\n[!] 错误: 文件未找到 - {e}")
        print("[!] 请检查 'TRAINING_DATASET_PATH' 和 'GENERATED_MOLECULES_PATH' 的路径是否正确。")
    except Exception as e:
        print(f"\n[!] 发生未知错误: {e}")