# 文件名: preprocess_data.py

import torch
from torch_geometric.data import Batch
import numpy as np
from collections import deque
import torch.nn.functional as F
from tqdm import tqdm  # 引入tqdm来显示进度条
import os


# --- 将您原有的 get_bfs_order 和 generate_subgraphs_with_features 函数复制到这里 ---

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
    if graph_data.edge_index is not None:
        for i in range(graph_data.edge_index.shape[1]):
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


def generate_subgraphs_with_features(batch: Batch, orders: list, target_edge_dim: int) -> list:
    """
    根据给定的批次、节点排序和目标维度，为每个图生成一个子图序列。
    该函数会为子图创建全连接的边，并附加上包含距离的、维度统一的边特征。

    Args:
        batch (Batch): PyG的批次对象。
        orders (list): 一个列表，每个元素是对应图的节点排序列。
        target_edge_dim (int): 目标边特征的维度（不包含距离），用于标准化。

    Returns:
        list: 一个列表，每个元素是对应原图的一个子图序列。
    """
    graph_list = batch.to_data_list()
    all_subgraph_sequences = []

    if len(graph_list) != len(orders):
        raise ValueError(f"不匹配：批次含{len(graph_list)}个图, 但提供{len(orders)}个排序。")

    # 遍历批次中的每一个图
    for graph, order in zip(graph_list, orders):
        full_graph_cpu = graph.to('cpu')

        if len(order) != full_graph_cpu.num_nodes:
            raise ValueError(f"不匹配：图有{full_graph_cpu.num_nodes}个节点, 但排序长{len(order)}。")

        single_graph_subgraphs = []
        # 按顺序生成规模从1到N的子图
        for i in range(1, len(order) + 1):
            subset_nodes = torch.tensor(order[:i], dtype=torch.long)
            # 从原图中提取子图
            subgraph = full_graph_cpu.subgraph(subset_nodes)

            # 保存子图节点在原图中的ID，便于追溯
            subgraph.n_id = subset_nodes
            num_sub_nodes = subgraph.num_nodes

            # 仅当子图节点数大于1时，才处理边
            if num_sub_nodes > 1:
                # 构建全连接的边索引
                adj_all = torch.combinations(torch.arange(num_sub_nodes), r=2)
                new_edge_index = torch.cat([adj_all, adj_all.flip(1)], dim=0).t().contiguous()

                # 定义"无化学键"的特征向量，其维度由全局标准决定
                no_bond_vec = torch.zeros(target_edge_dim)
                no_bond_vec[-1] = 1  # 假设最后一维代表无键状态

                # 将原始图中存在的边信息存入字典，便于快速查找
                bond_map = {}
                if subgraph.edge_index.numel() > 0:
                    for j in range(subgraph.edge_index.shape[1]):
                        u, v = subgraph.edge_index[:, j].tolist()
                        bond_map[(u, v)] = subgraph.edge_attr[j]

                # 为所有新创建的全连接边生成特征
                new_edge_attr_list = []
                for j in range(new_edge_index.shape[1]):
                    u, v = new_edge_index[:, j].tolist()

                    # 获取化学键类型：如果边在原图中存在，用原特征；否则用"无键"特征
                    bond_type_attr = bond_map.get((u, v), no_bond_vec)

                    # 维度标准化：如果取出的特征维度不足，进行填充以保证统一
                    if bond_type_attr.shape[0] < target_edge_dim:
                        pad_size = target_edge_dim - bond_type_attr.shape[0]
                        bond_type_attr = F.pad(bond_type_attr, (0, pad_size), "constant", 0)

                    # 计算原子间距离
                    pos_u, pos_v = subgraph.pos[u], subgraph.pos[v]
                    distance = torch.norm(pos_u - pos_v, p=2).unsqueeze(0)

                    # 拼接化学键类型和距离，形成最终的边特征
                    new_attr = torch.cat([bond_type_attr, distance], dim=0)
                    new_edge_attr_list.append(new_attr)

                # 更新子图的边索引和边特征
                subgraph.edge_index = new_edge_index
                subgraph.edge_attr = torch.stack(new_edge_attr_list, dim=0)

            single_graph_subgraphs.append(subgraph)
        all_subgraph_sequences.append(single_graph_subgraphs)

    return all_subgraph_sequences


if __name__ == '__main__':
    # 定义输入和输出路径
    ORIGINAL_DATASET_PATH = 'gdb9_pyg_dataset_with_absorbing_state.pt'
    # [修改] 输出现在是一个文件夹
    PREPROCESSED_DATA_DIR = 'gdb9_preprocessed_data'
    os.makedirs(PREPROCESSED_DATA_DIR, exist_ok=True)  # 创建输出文件夹

    print(f"正在加载原始数据集: {ORIGINAL_DATASET_PATH}...")
    original_dataset = torch.load(ORIGINAL_DATASET_PATH, weights_only=False)

    if not original_dataset:
        raise ValueError("原始数据集为空!")
    TARGET_EDGE_DIM_BEFORE_DIST = original_dataset[0].edge_attr.shape[1]

    print("开始生成所有子图，并分文件保存...")
    total_subgraphs_count = 0

    # 遍历每个原始图，并立即保存其子图序列
    for idx, original_graph in enumerate(tqdm(original_dataset, desc="正在预处理图")):
        batch = Batch.from_data_list([original_graph])
        orders = [get_bfs_order(original_graph)]

        subgraph_sequences = generate_subgraphs_with_features(batch, orders, TARGET_EDGE_DIM_BEFORE_DIST)

        if subgraph_sequences and subgraph_sequences[0]:
            # 获取当前图生成的所有子图
            subgraphs_for_this_graph = subgraph_sequences[0]

            # 为当前图的子图序列创建一个单独的保存文件
            save_path = os.path.join(PREPROCESSED_DATA_DIR, f'graph_{idx}.pt')
            torch.save(subgraphs_for_this_graph, save_path)

            total_subgraphs_count += len(subgraphs_for_this_graph)

    print(f"\n预处理完成！总共生成了 {total_subgraphs_count} 个子图，已分别保存至文件夹: {PREPROCESSED_DATA_DIR}")