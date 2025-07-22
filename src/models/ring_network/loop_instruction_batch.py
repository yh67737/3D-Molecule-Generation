import os
import torch
import numpy as np
from collections import deque
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
from torch_geometric.data.batch import Batch
from sklearn.metrics import accuracy_score, roc_auc_score


# --- 节点排序与子图生成函数 ---

def get_bfs_order(graph_data: Data, start_node: int = None) -> list:
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


# --- 环指导网络 ---
class RingPredictor(torch.nn.Module):
    """
    图注意力网络（GAT），用于预测每个节点（原子）是否属于环结构。
    """

    def __init__(self, node_feature_dim: int, edge_feature_dim: int, num_ring_classes: int = 1, hidden_dim: int = 64,
                 num_heads: int = 4):
        super(RingPredictor, self).__init__()
        self.num_ring_classes = num_ring_classes
        # GAT卷积层，edge_dim参数使其能利用边特征（如距离）
        self.conv1 = GATConv(node_feature_dim, hidden_dim, heads=num_heads, concat=True, dropout=0.1,
                             edge_dim=edge_feature_dim)
        self.bn1 = BatchNorm1d(hidden_dim * num_heads)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, concat=True, dropout=0.1,
                             edge_dim=edge_feature_dim)
        self.bn2 = BatchNorm1d(hidden_dim * num_heads)
        # 最后的线性输出层，用于分类
        self.out = Linear(hidden_dim * num_heads, num_ring_classes)

    def forward(self, data: Batch) -> torch.Tensor:
        """模型的前向传播"""
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # GAT层的前向传播，同时传入节点、边索引和边特征
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(self.bn1(x))
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(self.bn2(x))

        # 输出原始的logits
        logits = self.out(x)
        return logits


def train_epoch(model, loader, optimizer, criterion, device, target_edge_dim):
    """对模型进行一个周期的训练"""
    model.train()  # 设置为训练模式
    total_loss = 0
    # 遍历训练数据加载器
    for data_batch in loader:
        # 为批次中的每个图生成节点排序
        orders = [get_bfs_order(g) for g in data_batch.to_data_list()]
        # 生成子图序列，并传入统一的边特征维度标准
        subgraph_sequences = generate_subgraphs_with_features(data_batch, orders, target_edge_dim)

        optimizer.zero_grad()  # 清空梯度
        batch_loss = 0
        max_len = max(len(s) for s in subgraph_sequences if s)  # 获取最长的子图序列长度

        # 按生成步骤（时间步）进行迭代
        for step in range(max_len):
            # 提取出当前步骤的所有子图
            subgraphs_at_step_k = [seq[step] for seq in subgraph_sequences if len(seq) > step]
            if not subgraphs_at_step_k: continue

            # 过滤掉没有边的子图，因为它们无法通过GATConv的边注意力机制
            subgraphs_with_edges = [s for s in subgraphs_at_step_k if
                                    s.edge_index is not None and s.edge_index.numel() > 0]
            if not subgraphs_with_edges: continue

            # 将子图列表打包成一个批次并移到GPU
            gpu_subgraphs = [s.to(device) for s in subgraphs_with_edges]
            subgraph_batch = Batch.from_data_list(gpu_subgraphs)

            # 模型前向传播，计算损失
            predictions = model(subgraph_batch)
            ground_truth = subgraph_batch.pring_out
            step_loss = criterion(predictions, ground_truth)
            batch_loss = batch_loss + step_loss

        # 对整个批次的累积损失进行反向传播和优化
        if isinstance(batch_loss, torch.Tensor):
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, criterion, device, target_edge_dim):
    """在验证集或测试集上评估模型"""
    model.eval()  # 设置为评估模式
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():  # 在评估阶段不计算梯度
        for data_batch in loader:
            # 数据处理流程与训练时相同
            orders = [get_bfs_order(g) for g in data_batch.to_data_list()]
            subgraph_sequences = generate_subgraphs_with_features(data_batch, orders, target_edge_dim)
            max_len = max(len(s) for s in subgraph_sequences if s)

            for step in range(max_len):
                subgraphs_at_step_k = [seq[step] for seq in subgraph_sequences if len(seq) > step]
                if not subgraphs_at_step_k: continue

                subgraphs_with_edges = [s for s in subgraphs_at_step_k if
                                        s.edge_index is not None and s.edge_index.numel() > 0]
                if not subgraphs_with_edges: continue

                gpu_subgraphs = [s.to(device) for s in subgraphs_with_edges]
                subgraph_batch = Batch.from_data_list(gpu_subgraphs)

                predictions = model(subgraph_batch)
                ground_truth = subgraph_batch.pring_out
                loss = criterion(predictions, ground_truth)
                total_loss += loss.item()

                # 收集预测和标签以计算准确率
                preds = (torch.sigmoid(predictions) > 0.5).cpu().numpy()
                labels = ground_truth.cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)

    accuracy = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), accuracy


if __name__ == '__main__':
    # --- 设置全局参数和计算设备 ---
    DATASET_PATH = 'gdb9_pyg_dataset_with_absorbing_state.pt'
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    EPOCHS = 50  # 训练总轮数
    SAVE_INTERVAL = 1  # 每隔多少个epoch保存一次

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"将使用设备: {device}")

    # --- 加载和划分数据集 ---
    full_dataset = torch.load(DATASET_PATH, weights_only=False)
    np.random.shuffle(full_dataset)
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    train_dataset = full_dataset[:train_size]
    val_dataset = full_dataset[train_size: train_size + val_size]
    test_dataset = full_dataset[train_size + val_size:]

    print("\n--- 数据集信息 ---")
    print(f"总样本数: {len(full_dataset)}")
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")

    # --- 创建数据加载器 ---
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 实例化模型、损失函数和优化器 ---
    TARGET_EDGE_DIM_BEFORE_DIST = train_dataset[0].edge_attr.shape[1]

    NODE_FEATURE_DIM = train_dataset[0].x.shape[1]
    EDGE_FEATURE_DIM = TARGET_EDGE_DIM_BEFORE_DIST + 1
    NUM_RING_CLASSES = 1

    model = RingPredictor(
        node_feature_dim=NODE_FEATURE_DIM,
        edge_feature_dim=EDGE_FEATURE_DIM,
        num_ring_classes=NUM_RING_CLASSES
    ).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 执行训练和验证循环 ---
    print("\n--- 开始训练 ---")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, TARGET_EDGE_DIM_BEFORE_DIST)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device, TARGET_EDGE_DIM_BEFORE_DIST)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")

        # 每隔 SAVE_INTERVAL 个 epoch 保存一次模型
        # 同时在最后一个 epoch 也保存，确保最终模型被保存
        if epoch % SAVE_INTERVAL == 0 or epoch == EPOCHS:
            # 创建一个包含 epoch 编号的文件名，避免覆盖
            epoch_save_path = f'ring_predictor_epoch_{epoch}.pt'
            torch.save(model.state_dict(), epoch_save_path)
            print(f"  -> 已到达第 {epoch} 个 epoch，模型已保存至 {epoch_save_path}")

    # --- 在测试集上进行最终评估 ---
    print("\n--- 训练完成，开始在测试集上评估 ---")

    # 加载最后一个 epoch 保存的模型进行测试
    last_epoch_save_path = f'ring_predictor_epoch_{EPOCHS}.pt'
    if os.path.exists(last_epoch_save_path):
        model.load_state_dict(torch.load(last_epoch_save_path))
        print(f"已加载最后一个epoch的模型 ({last_epoch_save_path}) 进行最终测试。")

        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device, TARGET_EDGE_DIM_BEFORE_DIST)
        print(f"最终测试结果 | Test Loss: {test_loss:.4f} | Test Acc: {test_accuracy:.4f}")
    else:
        print(f"未找到最后一个epoch的模型文件 ({last_epoch_save_path})，跳过最终测试。")