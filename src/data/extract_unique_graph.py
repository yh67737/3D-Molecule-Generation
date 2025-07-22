import torch
import os
import hashlib
from tqdm import tqdm
from torch_geometric.data import Data
import shutil
import json

# =============================================================================
# 1. 配置输入和输出路径
# =============================================================================
# !!! 请修改这里 !!!
# 包含所有.pt子图文件的输入文件夹
INPUT_DIR = 'gdb9_preprocessed_data_bfs_json'
# 用于存放去重后唯一子图的输出文件夹
OUTPUT_DIR = 'gdb9_unique_subgraphs_json'


# =============================================================================
# 2. 定义图内容(从JSON字典)的哈希函数
# =============================================================================
def get_graph_hash_from_dict(graph_dict: dict) -> str:
    """
    为从JSON加载的图字典计算一个唯一的SHA256哈希值。

    Args:
        graph_dict: 一个包含图数据的Python字典。

    Returns:
        一个代表该图内容的十六进制哈希字符串。
    """
    # 定义哪些属性共同决定了一个图的唯一性 (与之前保持一致)
    ATTRIBUTES_TO_HASH = ['x', 'pos', 'edge_index', 'edge_attr', 'is_new_node']

    hasher = hashlib.sha256()

    for attr in ATTRIBUTES_TO_HASH:
        data_list = graph_dict.get(attr)

        # 确保数据列表存在且不为空
        if data_list is not None and len(data_list) > 0:
            # 将Python列表转换为NumPy数组，再转为字节流进行哈希
            # 这确保了无论JSON文件中的格式如何(例如空格)，哈希结果都保持一致
            data_bytes = np.array(data_list).tobytes()
            hasher.update(data_bytes)

    return hasher.hexdigest()


# =============================================================================
# 3. 主去重逻辑
# =============================================================================
def deduplicate_subgraphs(input_dir, output_dir):
    """
    遍历输入目录中的所有.json文件，加载它们，计算哈希值，
    并将唯一的子图文件复制到输出目录。
    """
    print(f"开始处理文件夹: '{input_dir}'")

    os.makedirs(output_dir, exist_ok=True)

    try:
        json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
        if not json_files:
            print("错误：输入文件夹中没有找到任何.json文件。")
            return
    except FileNotFoundError:
        print(f"错误：找不到输入文件夹 '{input_dir}'。请检查路径是否正确。")
        return

    seen_hashes = set()
    total_files = len(json_files)
    unique_files_count = 0

    for filename in tqdm(json_files, desc="正在去重JSON子图"):
        input_path = os.path.join(input_dir, filename)

        try:
            # 使用json库加载子图数据
            with open(input_path, 'r', encoding='utf-8') as f:
                subgraph_dict = json.load(f)

            # 调用新的哈希函数
            graph_hash = get_graph_hash_from_dict(subgraph_dict)

            if graph_hash not in seen_hashes:
                seen_hashes.add(graph_hash)

                output_path = os.path.join(output_dir, filename)
                shutil.copy2(input_path, output_path)

                unique_files_count += 1

        except Exception as e:
            print(f"\n处理文件 '{filename}' 时发生错误: {e}")

    print("\n--- 去重完成 ---")
    print(f"总共处理文件数: {total_files}")
    print(f"发现的唯一文件数: {unique_files_count}")
    print(f"所有唯一的子图已保存至文件夹: '{output_dir}'")


# =============================================================================
# 4. 执行脚本
# =============================================================================
if __name__ == '__main__':
    try:
        from tqdm import tqdm
    except ImportError:
        print("错误: 需要tqdm库来显示进度条。")
        print("请运行 'pip install tqdm' 进行安装。")
        exit()

    # 检查 numpy
    try:
        import numpy as np
    except ImportError:
        print("错误: 需要numpy库来进行哈希计算。")
        print("请运行 'pip install numpy'进行安装。")
        exit()

    deduplicate_subgraphs(INPUT_DIR, OUTPUT_DIR)