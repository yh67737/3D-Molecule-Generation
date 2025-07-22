import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from tqdm import tqdm
import os

def process_gdb9_sdf(sdf_path):
    """
    处理gdb9.sdf文件，将其转换为PyTorch Geometric的Data对象列表。

    处理流程:
    1. 读取SDF中的每个分子。
    2. 提取原子类型、坐标、键信息。
    3. 将坐标零中心化。
    4. 为每个原子添加环信息标签。
    5. 将分子转换为一个全连接图的PyG.Data对象，其中边被标记为真实化学键或'无键'。
    """
    print("开始处理SDF文件...")
    
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=True)

    pyg_data_list = []
    
    # QM9中可能出现的原子类型，并增加一个用于生成模型的“吸收态”
    atom_types = ['H', 'C', 'N', 'O', 'F', 'Absorbing']
    atom_map = {symbol: i for i, symbol in enumerate(atom_types)}

    # 定义了5种边类型：单键、双键、三键、芳香键、无键
    bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
        'NO_BOND'  # 新增，用于表示无键状态
    ]
    bond_map = {bond_type: i for i, bond_type in enumerate(bond_types)}
    
    # 预先计算 'NO_BOND' 的 one-hot 编码，以提高效率
    no_bond_one_hot = [0] * len(bond_types)
    no_bond_one_hot[bond_map['NO_BOND']] = 1

    for mol in tqdm(suppl, desc="处理分子"):
        if mol is None:
            continue

        num_atoms = mol.GetNumAtoms()
        
        # --- 1. 提取原子特征 (One-Hot) ---
        atom_features_list = []
        for atom in mol.GetAtoms():
            one_hot = [0] * len(atom_types)
            one_hot[atom_map[atom.GetSymbol()]] = 1
            atom_features_list.append(one_hot)
        x = torch.tensor(atom_features_list, dtype=torch.float)
        
        # --- 2. 坐标零中心化 ---
        conf = mol.GetConformer()
        pos = conf.GetPositions()
        center = np.mean(pos, axis=0)
        pos = pos - center
        
        # --- 3. 检测环信息 ---
        AllChem.GetSymmSSSR(mol)
        ring_info = mol.GetRingInfo()
        atom_ring_size = [0] * num_atoms
        for ring in ring_info.AtomRings():
            ring_size = len(ring)
            for atom_idx in ring:
                if atom_ring_size[atom_idx] == 0 or ring_size < atom_ring_size[atom_idx]:
                    atom_ring_size[atom_idx] = ring_size
        
        # --- 4. 将环信息转换为独立的张量 ---
        pring_out_1d = (torch.tensor(atom_ring_size, dtype=torch.float) > 0).float()
        pring_out = pring_out_1d.unsqueeze(1)

        # *** 主要修改点: 将分子视为全连接图 ***
        # --- 5. 提取边信息 (全连接图) ---
        
        # 步骤A: 创建一个查找表现有化学键的字典，以提高效率
        # 键是 (atom_idx_1, atom_idx_2) 的排序元组，值是键的one-hot编码
        existing_bonds = {}
        for bond in mol.GetBonds():
            # 获取键的one-hot编码
            rdkit_bond_type = bond.GetBondType()
            bond_one_hot = [0] * len(bond_types)
            bond_one_hot[bond_map[rdkit_bond_type]] = 1
            
            # 使用排序后的元组作为键，以确保(i, j)和(j, i)都指向同一个键属性
            i, j = sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            existing_bonds[(i, j)] = bond_one_hot

        edge_indices = []
        edge_attrs_list = []
        
        # 步骤B: 遍历所有可能的原子对，构建全连接图
        for i in range(num_atoms):
            for j in range(num_atoms):
                if i == j:
                    continue  # 忽略自环

                # 添加边 (i, j) 到边索引列表
                edge_indices.append((i, j))
                
                # 检查是否存在化学键
                key = tuple(sorted((i, j)))
                if key in existing_bonds:
                    # 如果存在化学键，使用其对应的one-hot编码
                    edge_attrs_list.append(existing_bonds[key])
                else:
                    # 如果不存在，使用 'NO_BOND' 的one-hot编码
                    edge_attrs_list.append(no_bond_one_hot)

        if len(edge_indices) > 0:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs_list, dtype=torch.float)
        else:
            # 处理单原子分子（没有边）
            edge_index = torch.empty((2, 0), dtype=torch.long)
            # 注意：这里的维度是 len(bond_types)，即5
            edge_attr = torch.empty((0, len(bond_types)), dtype=torch.float)


        # --- 6. 构建PyG Data对象 ---
        data = Data(
            x=x,                     # 原子特征 [N, 6]
            pos=torch.tensor(pos, dtype=torch.float), # 原子坐标 [N, 3]
            edge_index=edge_index,   # 边连接 [2, N*(N-1)]
            edge_attr=edge_attr,     # 边特征 (One-Hot) [N*(N-1), 5]
            pring_out=pring_out      # 环指导信息 [N, 1]
        )
        
        pyg_data_list.append(data)

    print(f"\n处理完成！共生成 {len(pyg_data_list)} 个分子的PyG数据对象。")
    return pyg_data_list

# --- 主程序 ---
if __name__ == '__main__':
    sdf_file_path = 'qm9_files/gdb9.sdf' 
    
    if not os.path.exists(sdf_file_path):
        print(f"错误: 未找到文件 '{sdf_file_path}'。")
    else:
        dataset = process_gdb9_sdf(sdf_file_path)

        if dataset:
            molecule_to_inspect = None
            # 我们找一个简单的多原子分子来检查，例如乙烷或乙烯
            # gdb_1 是甲烷 (1个C, 4个H, N=5)
            # gdb_2 是乙烷 (2个C, 6个H, N=8)
            if len(dataset) > 1:
                molecule_to_inspect = dataset[1] # 检查第二个分子（通常是乙烷）

            if molecule_to_inspect:
                print("\n--- 检查一个分子的数据示例 (全连接图) ---")
                print(molecule_to_inspect)
                num_atoms_in_sample = molecule_to_inspect.num_nodes
                print(f"分子中的原子数 (N): {num_atoms_in_sample}")

                print(f"\n原子特征张量 (x) shape: {molecule_to_inspect.x.shape}")
                
                # *** 更新检查输出的注释 ***
                print(f"\n边索引张量 (edge_index) shape: {molecule_to_inspect.edge_index.shape}")
                print(f"预期 shape: [2, {num_atoms_in_sample * (num_atoms_in_sample - 1)}]")
                
                print(f"\n边属性张量 (edge_attr) shape: {molecule_to_inspect.edge_attr.shape}")
                print(f"预期 shape: [{num_atoms_in_sample * (num_atoms_in_sample - 1)}, 5]")
                
                print("\n边属性 (edge_attr) 内容示例 (One-Hot编码):")
                # 打印前几条和后几条边属性
                print(molecule_to_inspect.edge_attr[:5])
                print("...")
                # 最后一列为1表示 'NO_BOND'
                print("注意: 边属性的最后一列对应于'NO_BOND'。")
                
            else:
                print("数据集中没有足够的分子以供演示。")


            # --- 保存处理好的数据集 ---
            output_path = 'gdb9_pyg_dataset_fully_connected.pt' # 使用新名字保存
            print(f"\n正在将处理好的数据集保存到 '{output_path}'...")
            torch.save(dataset, output_path)
            print("保存完成！")