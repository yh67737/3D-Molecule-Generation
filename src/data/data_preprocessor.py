# data_preprocessor(no_aromatic&H).py
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from tqdm import tqdm
import os

def process_gdb9_sdf_kekulize_removeH(sdf_path):
    """
    处理 gdb9.sdf 文件。
    1. 使用 RDKit 的 Kekulize 将芳香键转换为单/双键。
    2. 移除分子中所有的氢原子 (H)。
    3. 提取原子、坐标、键和环信息，构建PyG数据对象。
    """
    print("开始处理SDF文件 (将进行Kekulize去芳香化和去H操作)...")
    
    # removeHs=False 保留原始氢原子，以便后续处理
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=True)

    pyg_data_list = []
    kekulization_failures = 0
    
    # 定义原子类型
    atom_types = ['C', 'N', 'O', 'F'] # H原子将被移除
    atom_map = {symbol: i for i, symbol in enumerate(atom_types)}

    # 定义键类型，不包含 AROMATIC
    bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        'NO_BOND'
    ]
    bond_map = {bond_type: i for i, bond_type in enumerate(bond_types)}
    no_bond_one_hot = [0] * len(bond_types)
    no_bond_one_hot[bond_map['NO_BOND']] = 1

    for mol in tqdm(suppl, desc="处理分子"):
        if mol is None:
            continue

        # --- 主要修改点 START ---
        try:
            # 1. 对分子进行Kekulize操作，将芳香键转换为单双键
            Chem.Kekulize(mol)
        except Chem.rdchem.KekulizeException:
            kekulization_failures += 1
            continue  # 如果一个分子无法被Kekulize，则跳过

        # 2. 去除所有氢原子 (参考MolDiff)
        # 这一步会返回一个新的分子对象，且原子索引会重新计算
        mol = Chem.RemoveAllHs(mol)
        # --- 主要修改点 END ---

        num_atoms = mol.GetNumAtoms()
        if num_atoms == 0:
            continue
            
        # --- 1. 提取原子特征 (One-Hot) ---
        atom_features_list = []
        valid_molecule = True
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol not in atom_map:
                valid_molecule = False
                break
            one_hot = [0] * len(atom_types)
            one_hot[atom_map[symbol]] = 1
            atom_features_list.append(one_hot)
        
        if not valid_molecule:
            continue # 如果含有未定义的原子类型，跳过
            
        x = torch.tensor(atom_features_list, dtype=torch.float)
        
        # --- 2. 坐标零中心化 ---
        try:
            conf = mol.GetConformer()
            pos = conf.GetPositions()
            center = np.mean(pos, axis=0)
            pos = pos - center
        except ValueError:
            # 如果去H后没有原子了，或没有构象信息，则跳过
            continue
        
        # --- 3. 检测环信息 ---
        # GetSymmSSSR在新版RDKit中可能不是必须的，但保留无害
        try:
            AllChem.GetSymmSSSR(mol)
        except: # 捕获一些罕见的RDKit错误
            continue
            
        ring_info = mol.GetRingInfo()
        atom_ring_size = [0] * num_atoms
        for ring in ring_info.AtomRings():
            ring_size = len(ring)
            for atom_idx in ring:
                if atom_ring_size[atom_idx] == 0 or ring_size < atom_ring_size[atom_idx]:
                    atom_ring_size[atom_idx] = ring_size
        
        pring_out_1d = (torch.tensor(atom_ring_size, dtype=torch.float) > 0).float()
        pring_out = pring_out_1d.unsqueeze(1)

        # --- 4. 提取边信息 (全连接图) ---
        existing_bonds = {}
        for bond in mol.GetBonds():
            # Kekulize之后，已经没有AROMATIC类型的键了
            rdkit_bond_type = bond.GetBondType()
            if rdkit_bond_type in bond_map:
                bond_one_hot = [0] * len(bond_types)
                bond_one_hot[bond_map[rdkit_bond_type]] = 1
                i, j = sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
                existing_bonds[(i, j)] = bond_one_hot

        edge_indices = []
        edge_attrs_list = []
        
        for i in range(num_atoms):
            for j in range(num_atoms):
                if i == j: continue
                edge_indices.append((i, j))
                key = tuple(sorted((i, j)))
                edge_attrs_list.append(existing_bonds.get(key, no_bond_one_hot))

        if len(edge_indices) > 0:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs_list, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, len(bond_types)), dtype=torch.float)

        data = Data(x=x, pos=torch.tensor(pos, dtype=torch.float), edge_index=edge_index, edge_attr=edge_attr, pring_out=pring_out)
        pyg_data_list.append(data)

    print(f"\n处理完成！")
    print(f"共生成 {len(pyg_data_list)} 个分子的PyG数据对象。")
    print(f"因无法Kekulize而被跳过的分子数量: {kekulization_failures}")
    return pyg_data_list

# --- 主程序 ---
if __name__ == '__main__':
    sdf_file_path = 'gdb9.sdf'  # 确保你的SDF文件在这个路径
    if os.path.exists(sdf_file_path):
        # 使用更新后的函数
        dataset = process_gdb9_sdf_kekulize_removeH(sdf_file_path)
        
        if dataset:
            # 更新输出文件名以反映新的处理方式
            output_path = 'gdb9_pyg_dataset_kekulized_noH.pt'
            print(f"\n正在将处理好的数据集保存到 '{output_path}'...")
            torch.save(dataset, output_path)
            print("保存完成！")
            
            # 检查第一个数据对象
            print("\n检查第一个数据对象:")
            first_data = dataset[0]
            print(first_data)
            print(f"原子数 (x.shape): {first_data.x.shape[0]}")
            print(f"边数 (edge_index.shape): {first_data.edge_index.shape[1]}")
    else:
        print(f"错误: 未找到SDF文件 '{sdf_file_path}'")