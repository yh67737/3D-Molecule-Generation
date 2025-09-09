import torch
import numpy as np
from tqdm import tqdm

def calculate_average_bond_lengths(file_path):
    """
    加载包含分子图数据的文件，并计算不同类型化学键的平均键长。

    Args:
        file_path (str): .pt 文件的路径。

    Returns:
        dict: 包含每种键类型平均键长和数量的字典。
    """
    try:
        # 1. 加载数据文件
        # PyTorch Geometric 等库通常将多个图样本保存为列表
        all_data = torch.load(file_path, map_location=torch.device('cpu'))
        print(f"成功加载文件: {file_path}")
    except FileNotFoundError:
        print(f"错误: 文件未找到 at '{file_path}'")
        return None
    except Exception as e:
        print(f"加载文件时发生错误: {e}")
        return None

    # 如果加载出来不是一个列表而是一个单独的数据对象，将其放入列表中以便统一处理
    if not isinstance(all_data, list):
        all_data = [all_data]

    print(f"文件中包含 {len(all_data)} 个图数据对象。")

    # 2. 初始化用于存储所有键长的字典
    # 键是键类型，值是包含该类型所有键长的列表
    bond_lengths = {
        'single': [],
        'double': [],
        'triple': [],
        'aromatic': []
    }
    
    # 建立从 one-hot 索引到键类型的映射
    # 根据您的描述: 0=单, 1=双, 2=三, 3=芳香, 4=无键
    bond_type_map = {
        0: 'single',
        1: 'double',
        2: 'triple',
        3: 'aromatic'
    }

    # 3. 遍历文件中的每一个图数据对象
    print("开始处理图数据并计算键长...")
    for data in tqdm(all_data, desc="Processing graphs"):
        pos = data.pos
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # 检查数据完整性
        if pos is None or edge_index is None or edge_attr is None:
            print("警告: 跳过一个不完整的数据对象。")
            continue

        # 4. 遍历当前图中的每一条边
        num_edges = edge_index.shape[1]
        for i in range(num_edges):
            # 获取边的属性 (one-hot 编码)
            attr = edge_attr[i]
            # 找到值为1的索引，即边的类型
            bond_type_index = torch.argmax(attr).item()

            # 如果是 "无键" (索引为4)，则跳过
            if bond_type_index not in bond_type_map:
                continue
            
            # 获取键类型名称
            bond_type_name = bond_type_map[bond_type_index]

            # 5. 获取边连接的两个原子的索引
            atom_idx1 = edge_index[0, i].item()
            atom_idx2 = edge_index[1, i].item()

            # 6. 获取这两个原子的三维坐标
            pos1 = pos[atom_idx1]
            pos2 = pos[atom_idx2]

            # 7. 计算它们之间的欧氏距离 (键长)
            # torch.dist 计算 L2 范数，即欧氏距离
            distance = torch.dist(pos1, pos2).item()
            
            # 将键长添加到对应的列表中
            bond_lengths[bond_type_name].append(distance)

    # 8. 计算每种类型的平均键长
    results = {}
    print("\n--- 计算结果 ---")
    for bond_type, lengths in bond_lengths.items():
        count = len(lengths)
        if count > 0:
            average_length = np.mean(lengths)
            std_dev = np.std(lengths) # 计算标准差，可以更好地了解数据分布
            results[bond_type] = {'average': average_length, 'count': count, 'std_dev': std_dev}
            print(f"类型: {bond_type.capitalize():<10} | 数量: {count:<10} | 平均键长: {average_length:.4f} | 标准差: {std_dev:.4f}")
        else:
            results[bond_type] = {'average': 0, 'count': 0, 'std_dev': 0}
            print(f"类型: {bond_type.capitalize():<10} | 未找到该类型的键。")
            
    return results

if __name__ == "__main__":
    # 请将此路径替换为您的实际文件路径
    PT_FILE_PATH = "/root/autodl-tmp/molecular_generation_project/prepared_data/small_fully_connected.pt"
    
    # 执行计算
    average_lengths = calculate_average_bond_lengths(PT_FILE_PATH)

    if average_lengths:
        print("\n计算完成！")