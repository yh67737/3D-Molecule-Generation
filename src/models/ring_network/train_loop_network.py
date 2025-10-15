# 文件名: train_preprocessed.py

import os
import torch
import numpy as np
# from e3nn.nn import Dropout
from torch.nn import Linear, BatchNorm1d, Dropout
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from .egnn_new import EGNN 

# --- RingPredictor 模型定义 (保持不变) ---
class RingPredictor(torch.nn.Module):
    """
    一个基于EGNN的E(3)等变的环结构预测器。
    它同时利用节点/边特征和3D坐标信息。
    """

    def __init__(self, node_feature_dim: int, edge_feature_dim: int, num_ring_classes: int = 1, 
                 hidden_nf: int = 64, n_layers: int = 4, attention: bool = True,
                 dropout_rate: float = 0.5):
        """
        Args:
            node_feature_dim (int): 输入节点特征的维度。
            edge_feature_dim (int): 输入边特征的维度。(注意：EGNN不直接使用edge_attr，但我们可以在forward中处理)
            num_ring_classes (int): 输出类别的数量 (对于二分类，为1)。
            hidden_nf (int): EGNN隐藏层的维度。
            n_layers (int): EGNN中EquivariantBlock的数量。
            attention (bool): EGNN中是否使用注意力机制。
            dropout_rate (float): 应用于最终输出前的dropout率。
        """
        super(RingPredictor, self).__init__()

        # 1. 实例化 EGNN 作为模型的主干
        #    in_node_nf: 输入节点特征维度
        #    hidden_nf: 隐藏层维度
        #    out_node_nf: EGNN输出的节点特征维度，我们让它等于隐藏层维度
        #    in_edge_nf: EGNN计算距离后拼接的额外边特征维度。
        #                原始EGNN只用了距离(2维: radial, sin_embedding)，
        #                我们这里暂时设置为0，在forward里动态处理。
        self.egnn = EGNN(
            in_node_nf=node_feature_dim,
            hidden_nf=hidden_nf,
            out_node_nf=hidden_nf, 
            in_edge_nf=edge_feature_dim, # <-- 关键：把边特征维度传给EGNN
            n_layers=n_layers,
            attention=attention
        )

        # 2. 定义一个分类头 (Classification Head)
        #    这个MLP将EGNN输出的节点表征映射到最终的预测logits
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_nf, hidden_nf),
            torch.nn.ReLU(),
            Dropout(p=dropout_rate), # 使用标准的Dropout
            torch.nn.Linear(hidden_nf, num_ring_classes)
        )

    def forward(self, data: Batch) -> torch.Tensor:
        """模型的前向传播"""
        # 从data对象中解包出节点特征(h), 坐标(x), 和边索引
        h, x, edge_index, edge_attr = data.x, data.pos, data.edge_index, data.edge_attr # <-- 增加 edge_attr

        # 检查是否有边，如果没有边，EGNN无法运行
        if edge_index is None or edge_index.numel() == 0:
            # 对于没有边的图（单个原子），直接返回0的logits
            return torch.zeros((h.size(0), 1), device=h.device)

        # 1. 通过EGNN主干网络处理图，得到更新后的节点特征和坐标
        #    注意：这个版本的EGNN不直接接收edge_attr，它内部通过coord2diff计算距离作为边信息。
        #    如果想把edge_attr也利用起来，需要修改EGNN内部或在这里做特征拼接，
        #    但对于环预测，几何信息可能更重要。我们先从简化版开始。
        h_final, _ = self.egnn(h=h, x=x, edge_index=edge_index, edge_attr=edge_attr)

        # 2. 将最终的节点特征通过分类头得到预测logits
        logits = self.head(h_final)
        
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
    PREPROCESSED_DATA_DIR = 'src/models/ring_network/gdb9_pt_data_for_ring_predictor'
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
    
    print("\n--- Model Dimensionality ---")
    print(f"Node feature dimension: {NODE_FEATURE_DIM}")
    print(f"Edge feature dimension: {EDGE_FEATURE_DIM}") # <--- 核心打印语句
    print("--------------------------\n")

    # 实例化新的 EGNN-based RingPredictor
    model = RingPredictor(
        node_feature_dim=NODE_FEATURE_DIM,
        edge_feature_dim=EDGE_FEATURE_DIM, 
        num_ring_classes=NUM_RING_CLASSES,
        hidden_nf=64,   # EGNN隐藏层维度
        n_layers=4      # EGNN层数
    ).to(device)
    
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