# 文件名: train_preprocessed.py

import os
import torch
import numpy as np
from e3nn.nn import Dropout
from torch.nn import Linear, BatchNorm1d
import torch.nn.functional as F
from torch_geometric.data import Batch, DataLoader
from torch_geometric.nn import GATConv
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# --- RingPredictor 模型定义 (保持不变) ---
class RingPredictor(torch.nn.Module):
    """
    图注意力网络（GAT），增加了更强的正则化来预测环结构。
    """

    def __init__(self, node_feature_dim: int, edge_feature_dim: int, num_ring_classes: int = 1, hidden_dim: int = 64,
                 num_heads: int = 4, gat_dropout: float = 0.1, dropout_rate: float = 0.5):  # <--- 新增 dropout_rate 参数
        """
        Args:
            ...
            gat_dropout (float): GAT层中注意力权重的dropout率。
            dropout_rate (float): 应用于节点特征的dropout率。
        """
        super(RingPredictor, self).__init__()
        self.num_ring_classes = num_ring_classes

        # GAT卷积层，edge_dim使其能利用边特征
        # GATConv内部的dropout作用于注意力系数，防止模型过度依赖少数邻居
        self.conv1 = GATConv(node_feature_dim, hidden_dim, heads=num_heads, concat=True, dropout=gat_dropout,
                             edge_dim=edge_feature_dim)
        self.bn1 = BatchNorm1d(hidden_dim * num_heads)

        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, concat=True, dropout=gat_dropout,
                             edge_dim=edge_feature_dim)
        self.bn2 = BatchNorm1d(hidden_dim * num_heads)

        # --- 新增的正则化层 ---
        # 这是一个标准的Dropout层，它将作用于整个节点的嵌入表示
        self.dropout = Dropout(p=dropout_rate)  # <--- 在此实例化

        # 最后的线性输出层
        self.out = Linear(hidden_dim * num_heads, num_ring_classes)

    def forward(self, data: Batch) -> torch.Tensor:
        """模型的前向传播"""
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # --- 第1层 ---
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(self.bn1(x))
        x = self.dropout(x)  # <--- 在激活函数后应用Dropout

        # --- 第2层 ---
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(self.bn2(x))
        x = self.dropout(x)  # <--- 再次应用Dropout

        # 输出原始的logits
        logits = self.out(x)
        return logits


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for subgraph_batch in loader:
        # 数据加载器直接提供可以直接使用的、带有正确特征的子图批次
        subgraph_batch = subgraph_batch.to(device)

        # 过滤掉批次中可能存在的没有边的子图
        if subgraph_batch.edge_index is None or subgraph_batch.edge_index.numel() == 0:
            continue

        optimizer.zero_grad()
        predictions = model(subgraph_batch)
        ground_truth = subgraph_batch.pring_out
        loss = criterion(predictions, ground_truth)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for subgraph_batch in loader:
            subgraph_batch = subgraph_batch.to(device)
            if subgraph_batch.edge_index is None or subgraph_batch.edge_index.numel() == 0:
                continue

            predictions = model(subgraph_batch)
            ground_truth = subgraph_batch.pring_out
            loss = criterion(predictions, ground_truth)
            total_loss += loss.item()
            preds = (torch.sigmoid(predictions) > 0.5).cpu().numpy()
            labels = ground_truth.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
    accuracy = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), accuracy


if __name__ == '__main__':
    # 使用新的预处理数据集 ---
    PREPROCESSED_DATA_DIR = 'gdb9_preprocessed_data'
    BATCH_SIZE = 1024
    LEARNING_RATE = 1e-4
    EPOCHS = 50
    SAVE_INTERVAL = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"将使用设备: {device}")

    print(f"正在从文件夹加载预处理过的数据: {PREPROCESSED_DATA_DIR}...")

    all_subgraph_files = [os.path.join(PREPROCESSED_DATA_DIR, f) for f in os.listdir(PREPROCESSED_DATA_DIR) if
                          f.endswith('.pt')]

    full_dataset = []
    for f_path in tqdm(all_subgraph_files, desc="正在加载数据文件"):
        full_dataset.extend(torch.load(f_path))

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
    # 可以使用更大的 num_workers，因为现在IO是主要任务
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # --- 实例化模型 ---
    # 直接从预处理好的数据中获取维度
    sample_data = train_dataset[0]
    NODE_FEATURE_DIM = sample_data.x.shape[1]
    EDGE_FEATURE_DIM = sample_data.edge_attr.shape[1]
    NUM_RING_CLASSES = 1

    model = RingPredictor(NODE_FEATURE_DIM, EDGE_FEATURE_DIM, NUM_RING_CLASSES).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 执行训练和验证循环 ---
    print("\n--- 开始在预处理数据上进行高效训练 ---")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
        if epoch % SAVE_INTERVAL == 0 or epoch == EPOCHS:
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

        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        print(f"最终测试结果 | Test Loss: {test_loss:.4f} | Test Acc: {test_accuracy:.4f}")
    else:
        print(f"未找到最后一个epoch的模型文件 ({last_epoch_save_path})，跳过最终测试。")