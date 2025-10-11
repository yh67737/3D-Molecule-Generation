import torch
from torch_geometric.data import Data
import numpy as np
from collections import deque
from tqdm import tqdm
import os
from torch_geometric.data import Batch
import random
import json

def get_bfs_order(graph_data, start_node = None) -> list:
    """
    在一个PYG图数据对象上执行广度优先搜索(BFS)，返回节点访问顺序。

    Args:
        graph_data (Data): PyG的图数据对象。
        start_node (int, optional): BFS起始节点。如果为None，则随机选择一个。

    Returns:
        list: 包含节点索引的BFS排序列表。
    """
    # 如果未指定起始节点，则随机选择一个
    if start_node is None:
        start_node = np.random.randint(0, graph_data.num_nodes)

    # 构建邻接表以便于遍历
    adj = {i: [] for i in range(graph_data.num_nodes)}
    if graph_data.edge_index is not None and graph_data.edge_attr is not None:
        for i in range(graph_data.edge_index.shape[1]):
            if graph_data.edge_attr[i][-1] != 1:
                u, v = graph_data.edge_index[:, i].tolist()
                adj[u].append(v)
                adj[v].append(u)

    # 执行标准的BFS流程
    order = []
    visited = {start_node}
    queue = deque([start_node])
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in adj[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    # 处理非连通图：确保所有节点都被包含在排序中
    if len(order) < graph_data.num_nodes:
        all_nodes = set(range(graph_data.num_nodes))
        remaining_nodes = list(all_nodes - visited)
        order.extend(remaining_nodes)

    return order


def generate_centered_subgraphs(graph: Data, order: list) -> list:
    """
    根据给定的图和节点排序，生成一个子图序列。
    每个子图的原子坐标都经过几何中心化，并被标记是否为最终子图。
    该函数只保留子图内部的原始边，不创建全连接。

    Args:
        graph (Data): 单个PyG图数据对象。
        order (list): 该图的节点排序列。

    Returns:
        list: 一个列表，包含按顺序生成的、经过处理的子图。
    """
    if len(order) != graph.num_nodes:
        raise ValueError(f"节点排序长度 ({len(order)}) 与图的节点数 ({graph.num_nodes}) 不匹配。")

    graph_cpu = graph.to('cpu')
    subgraph_sequence = []

    # 按顺序，规模从1到N，依次生成子图
    for i in range(1, len(order) + 1):
        subset_nodes = torch.tensor(order[:i], dtype=torch.long)

        # 从原图中提取子图
        subgraph = graph_cpu.subgraph(subset_nodes)

        # 1. 根据“旧”原子对坐标 (pos) 进行几何中心化
        if hasattr(subgraph, 'pos') and subgraph.pos is not None and subgraph.num_nodes > 0:
            # 如果子图包含多个节点，则使用“旧”节点（除最后一个外）计算中心
            # (i > 1)
            if subgraph.num_nodes > 1:
                center = subgraph.pos[:-1].mean(dim=0, keepdim=True)
            # 如果子图只有一个节点（即序列中的第一个），则其自身就是中心
            # (i == 1)
            else:
                center = subgraph.pos.mean(dim=0, keepdim=True)

            # 将计算出的位移应用到所有节点上
            subgraph.pos = subgraph.pos - center

        # 2. 添加条件标志位
        is_new_node_flag = torch.zeros(subgraph.num_nodes, 1, dtype=torch.float)
        # 将最新加入的那个节点（在子图中的索引是 i-1，也就是最后一个）的标志位设为 1
        is_new_node_flag[-1] = 1.0
        subgraph.is_new_node = is_new_node_flag

        # 3. 保存子图节点在原图中的ID，便于追溯
        subgraph.original_node_indices = subset_nodes

        subgraph_sequence.append(subgraph)

    return subgraph_sequence


def generate_single_centered_subgraph(graph: Data, order: list) -> Data:
    """
    根据给定的图和节点排序，从所有可能的子图中随机生成一个。
    生成的单个子图的原子坐标经过几何中心化，并被标记是否为最终子图。

    Args:
        graph (Data): 单个PyG图数据对象。
        order (list): 该图的节点排序列。

    Returns:
        Data: 一个随机生成的、经过处理的子图对象。
    """
    if not order:
        raise ValueError("输入的节点排序 'order' 不能为空。")

    if len(order) != graph.num_nodes:
        raise ValueError(f"节点排序长度 ({len(order)}) 与图的节点数 ({graph.num_nodes}) 不匹配。")

    graph_cpu = graph.to('cpu')


    # 随机选择一个子图的大小 k，范围从1到完整图的大小
    k = random.randint(1, len(order))
    subset_nodes = torch.tensor(order[:k], dtype=torch.long)

    # 从原图中提取子图
    subgraph = graph_cpu.subgraph(subset_nodes)

    # 1. 根据“旧”原子对坐标 (pos) 进行几何中心化
    if hasattr(subgraph, 'pos') and subgraph.pos is not None and subgraph.num_nodes > 0:
        # 如果子图包含多个节点，则使用“旧”节点（除最后一个外）计算中心
        if subgraph.num_nodes > 1:
            center = subgraph.pos[:-1].mean(dim=0, keepdim=True)
        # 如果子图只有一个节点（即新节点），则其自身就是中心
        else:
            center = subgraph.pos.mean(dim=0, keepdim=True)

        subgraph.pos = subgraph.pos - center

    # 2. 添加条件标志位
    is_new_node_flag = torch.zeros(subgraph.num_nodes, 1, dtype=torch.float)
    # 将最新加入的那个节点（在子图中的索引是 k-1，也就是最后一个）的标志位设为 1
    is_new_node_flag[-1] = 1.0
    subgraph.is_new_node = is_new_node_flag

    # 3. 保存子图节点在原图中的ID
    subgraph.original_node_indices = subset_nodes

    # 4. 直接返回这个处理好的单个子图
    return subgraph


if __name__ == '__main__':
    # 定义输入和输出路径
    ORIGINAL_DATASET_PATH = 'gdb9_pyg_dataset_fc_no_aromatic_removed.pt'
    # [修改] 输出文件夹名称，以表明保存的是JSON文件
    PREPROCESSED_DATA_DIR = '../../prepared_data/autodl-tmp/gdb9_unique_subgraphs_json'
    os.makedirs(PREPROCESSED_DATA_DIR, exist_ok=True)

    print(f"正在加载原始数据集: {ORIGINAL_DATASET_PATH}...")
    original_dataset = torch.load(ORIGINAL_DATASET_PATH, weights_only=False)

    if not original_dataset:
        raise ValueError("原始数据集为空!")

    print("开始生成所有子图，并以JSON格式分文件保存...")
    total_subgraphs_count = 0

    # 遍历每个原始图
    for idx, original_graph in enumerate(tqdm(original_dataset, desc="正在预处理图")):
        # 1. 生成节点顺序和子图序列 (这部分不变)
        order = get_bfs_order(original_graph)
        subgraph_sequence = generate_centered_subgraphs(original_graph, order)

        # 2. [修改] 遍历序列中的每一个子图，并单独保存
        for sub_idx, subgraph in enumerate(subgraph_sequence):
            # 创建一个字典来存储可读的数据
            subgraph_dict = {}
            # 遍历Data对象的所有属性 (x, edge_index, pos, etc.)
            for key, value in subgraph.items():
                subgraph_dict[key] = value.tolist()

            # 定义每个子图的保存路径，使用.json后缀
            save_path = os.path.join(PREPROCESSED_DATA_DIR, f'graph_{idx}_subgraph_{sub_idx}.json')

            # 使用json库将字典写入文件，确保使用utf-8编码
            with open(save_path, 'w', encoding='utf-8') as f:
                # indent=4 让JSON文件格式化，更易读
                json.dump(subgraph_dict, f, ensure_ascii=False, indent=4)

        total_subgraphs_count += len(subgraph_sequence)

    print(
        f"\n预处理完成！总共生成了 {total_subgraphs_count} 个子图，已分别以JSON格式保存至文件夹: {PREPROCESSED_DATA_DIR}")
