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
    5. 将分子转换为PyG.Data对象。
    """
    print("开始处理SDF文件...")
    
    # Chem.SDMolSupplier是RDKit的SDF读取器，用来读取和解析SDF文件，suppl是一个迭代器
    # sanitize=True: (默认开启) 对分子进行化学合理性检查，如检查价态、识别芳香环等。
    #                 如果检查失败（如发现五价碳），则该分子会被跳过（返回None）。
    # removeHs=False: 我们需要保留氢原子
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=True)

    #创建一个空列表，用于存储处理好的Data对象
    pyg_data_list = []
    
    # *** 主要修改点 1: 在原子类型列表中加入'吸收态' ***
    # QM9中可能出现的原子类型，并增加一个用于生成模型的“吸收态”
    atom_types = ['H', 'C', 'N', 'O', 'F', 'Absorbing']
    # 'Absorbing' (吸收态) is added as a special type, often used in generative models.
    
    # 创建原子类型字典，它现在将包含'Absorbing'
    # 例如: {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'Absorbing': 5}
    atom_map = {symbol: i for i, symbol in enumerate(atom_types)}

    # 定义了5种边类型：单键、双键、三键、芳香键、无键 (用于后续的掩码或加噪)
    bond_types = [
        Chem.rdchem.BondType.SINGLE,    # 1.0
        Chem.rdchem.BondType.DOUBLE,    # 2.0
        Chem.rdchem.BondType.TRIPLE,    # 3.0
        Chem.rdchem.BondType.AROMATIC,  # 1.5
        'NO_BOND'                       # 新增，用于表示无键状态
    ]

     # 创建从RDKit键类型到整数索引的映射
    bond_map = {bond_type: i for i, bond_type in enumerate(bond_types)}
    # 注意：'NO_BOND' 不在RDKit的标准类型中，我们给它索引4

    # 使用tqdm显示进度条
    for mol in tqdm(suppl, desc="处理分子"):    #取出一个分子对象
        if mol is None:
            continue     #对分子进行化学合理性检查失败（价态不合理），跳过该分子

        # --- 1. 提取基本信息 ---
        # 获取当前分子对象 (mol) 中所包含的原子的总数量
        num_atoms = mol.GetNumAtoms()
        
        # 原子特征 (使用 one-hot 编码原子类型)
        # 这里的one-hot编码会自动适应新的atom_types列表长度
        atom_features_list = []
        for atom in mol.GetAtoms():
            # One-hot 编码，长度现在是len(atom_types), 即 6
            one_hot = [0] * len(atom_types)
            # 因为我们是从SDF文件中读取真实分子，所以永远不会遇到'Absorbing'类型的原子。
            # atom.GetSymbol() 将返回 'H', 'C'等, atom_map会找到正确的索引。
            # 'Absorbing'对应的索引将保持为0，这是正确的。
            one_hot[atom_map[atom.GetSymbol()]] = 1
            atom_features_list.append(one_hot)

        # *** 结果变化点: x的维度将是 [N, 6] 而不是 [N, 5] ***
        x = torch.tensor(atom_features_list, dtype=torch.float)
        
        # --- 2. 坐标零中心化 ---
        # 获取第一个构象 (QM9每个分子只有一个构象)
        conf = mol.GetConformer()
        # 获取该构象中所有原子的三维坐标
        pos = conf.GetPositions()
        # 计算几何中心并平移
        center = np.mean(pos, axis=0)
        pos = pos - center
        
        # --- 3. 检测环信息 ---
        # RDKit可以找到构成环的原子
        # GetSymmSSSR() 寻找对称化的最小环集 (Symmetrized Smallest Set of Smallest Rings)
        AllChem.GetSymmSSSR(mol)   # 找出mol中构成所有基本环的原子集合
        ring_info = mol.GetRingInfo()   # ring_info是所有环信息的集合
        
        # 为每个原子初始化环标签为0 (不成环)
        atom_ring_size = [0] * num_atoms
        # 遍历所有环
        for ring in ring_info.AtomRings():
            ring_size = len(ring)
            for atom_idx in ring:
                # 如果原子已在更小的环中，则不更新，保证了标签永远是最小环的大小
                if atom_ring_size[atom_idx] == 0 or ring_size < atom_ring_size[atom_idx]:
                    atom_ring_size[atom_idx] = ring_size
        
        # --- 4. 将环信息转换为独立的张量 ---
        # 这里的pring_out_1d是一个布尔张量 (1.0 for in ring, 0.0 for not)
        # 也可以存储环大小：pring_out_1d = torch.tensor(atom_ring_size, dtype=torch.long)
        pring_out_1d = (torch.tensor(atom_ring_size, dtype=torch.float) > 0).float()

        # --- 使用 unsqueeze(1) 将维度从 [N] 变为 [N, 1] ---
        pring_out = pring_out_1d.unsqueeze(1)

        # --- 5. 提取键信息 (边) ---
        edge_indices = []   # 用来临时存储所有边的起点-终点原子索引对，例如 (0, 1), (1, 0)
        edge_attrs_list = []   # 用来临时存储与每条边对应的属性，例如 [1.0]
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            # 获取键类型
            rdkit_bond_type = bond.GetBondType()
            
            # One-Hot 编码边类型
            bond_one_hot = [0] * len(bond_types)
            bond_one_hot[bond_map[rdkit_bond_type]] = 1

            # PyG的edge_index需要双向的边
            edge_indices.append((i, j))
            edge_indices.append((j, i))
            
            # 边属性也需要对应
            edge_attrs_list.append(bond_one_hot)
            edge_attrs_list.append(bond_one_hot)

        if len(edge_indices) > 0:   # 将之前收集在Python列表中的数据转换为PyTorch张量
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs_list, dtype=torch.float)
        else:
            # 处理没有键的单原子分子
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, len(bond_types)), dtype=torch.float)


        # --- 6. 构建PyG Data对象 ---
        data = Data(
            x=x,                     # 原子特征 [N, 6] (维度已改变)
            pos=torch.tensor(pos, dtype=torch.float), # 原子坐标 [N, 3]
            edge_index=edge_index,   # 边连接 [2, E]
            edge_attr=edge_attr,     # 边特征 (One-Hot) [E, num_bond_features]
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
            # --- 检查一个有环的分子，例如苯 (gdb_133880，但索引不确定，找一个靠后的) ---
            # 为了演示，我们遍历找到一个有环的分子
            molecule_to_inspect = None
            for mol_data in reversed(dataset):
                if mol_data.pring_out.sum() > 0:
                    molecule_to_inspect = mol_data
                    break
            
            if molecule_to_inspect:
                print("\n--- 检查一个有环分子的数据示例 ---")
                print(molecule_to_inspect)
                # *** 主要修改点 2: 更新检查输出的注释 ***
                print(f"原子特征张量 (x) shape: {molecule_to_inspect.x.shape}")
                print("原子特征 (x) 内容 (OneHot, 维度已扩展为6):")
                print(molecule_to_inspect.x)
                print("\n注意: 原子特征的最后一列对应于'吸收态'，对于来自SDF的真实分子，该列始终为0。")
                
                print(f"\n边索引张量 (edge_index) shape: {molecule_to_inspect.edge_index.shape}")
                
                print(f"\n边属性张量 (edge_attr) shape: {molecule_to_inspect.edge_attr.shape}")
                print("边属性 (edge_attr) 内容 (One-Hot编码):")
                print(molecule_to_inspect.edge_attr)

                print(f"\n环指导信息张量 (pring_out) shape: {molecule_to_inspect.pring_out.shape}")
                print("环指导信息 (pring_out) 内容 (1.0=在环中, 0.0=不在):")
                print(molecule_to_inspect.pring_out)
            else:
                print("数据集中未找到有环分子以供演示。")


            # --- 保存处理好的数据集 ---
            output_path = 'gdb9_pyg_dataset_with_absorbing_state.pt' # 使用新名字保存
            print(f"\n正在将处理好的数据集保存到 '{output_path}'...")
            torch.save(dataset, output_path)
            print("保存完成！")


            # 将来可以这样加载:
            # loaded_dataset = torch.load(output_path)