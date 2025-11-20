import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GraphConv, global_mean_pool, MessagePassing
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, Pad
import torch.nn.functional as F
from rdkit import Chem
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler # 归一化(标准化)
from sklearn.preprocessing import MinMaxScaler # 归一化(最大最小归一化)


# 数据集构建

# 构建Data数据
def smiles_to_graph(smiles, y):
    # 解析SMILES字符串并构建图结构
    mol = Chem.MolFromSmiles(smiles)
    atoms = mol.GetAtoms()
    edges = mol.GetBonds()

    # 创建节点特征
    node_features = []
    for atom in atoms:
        # 增加更多的原子特征
        hybridization = atom.GetHybridization()
        is_aromatic = atom.GetIsAromatic()
        num_neighbors = len(atom.GetNeighbors())
        atom_type = atom.GetSymbol()  # 收集原子类型，C、N、O等

        node_features.append([
            atom.GetAtomicNum(),  # 原子序数
            atom.GetTotalNumHs(),  # 氢原子数量
            atom.GetFormalCharge(),  # 正式电荷
            hybridization,  # 杂化状态
            1 if is_aromatic else 0,  # 是否为芳香族原子
            num_neighbors  # 邻近原子数量
        ])

    # 创建边特征
    edge_features = []
    for bond in edges:
        bond_type = bond.GetBondTypeAsDouble()
        bond_stereo = bond.GetStereo()
        is_aromatic_bond = 1 if bond.GetIsAromatic() else 0  # 芳香性键

        edge_features.append([
            bond_type,  # 键类型
            bond_stereo,  # 立体化学
            is_aromatic_bond  # 是否为芳香性键
        ])

    num_nodes = len(atoms)
    num_edges = len(edges)
    x = torch.tensor(node_features, dtype=torch.float)
    edge_attr = torch.tensor(edge_features, dtype=torch.float)

    edge_index = []
    for bond in edges:
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.append([begin, end])
        edge_index.append([end, begin])
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_index = edge_index.t()

    # 创建Data对象
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, num_nodes=num_nodes, num_edges=num_edges)
    return data


class SmilesToGraphDataset(Dataset):
    '''自定义Dataset类，用于实现图结点填充'''

    def __init__(self, data_list, transform=None):
        super(SmilesToGraphDataset, self).__init__()
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        if self.transform:
            data = self.transform(data)
        return data


# CSV文件路径
csv_file_path = '250k_rndm_zinc_drugs_clean.csv'

# 利用pandas读取文件
file = pd.read_csv(csv_file_path)
# 提取第一列的 SMILES 信息
smiles_list = file.iloc[:, 0].tolist()  # 列表
y = file.iloc[:, 1:4].values  # 二维数组

# 对y归一化(标准化方法)
# scaler = StandardScaler()
# y = scaler.fit_transform(y)

# 对y归一化(最大最小方法)
min_max_scaler = MinMaxScaler()
y = min_max_scaler.fit_transform(y)

# 转换为numpy数组
smiles_array = np.array(smiles_list)
y_array = np.array(y)

# 生成随机的索引
indices = np.random.permutation(len(smiles_array))

# 根据随机索引打乱数据
shuffled_smiles = smiles_array[indices]
shuffled_y = y_array[indices]

# 构建训练集和测试集
total = len(shuffled_smiles)
split_ratio = 4 / 5
train_idx = int(total * split_ratio)
train_smiles = shuffled_smiles[:train_idx].tolist()
train_y = shuffled_y[:train_idx]
test_smiles = shuffled_smiles[train_idx:].tolist()
test_y = shuffled_y[train_idx:]

# 输出训练和测试集的大小
print(f"Training samples: {len(train_smiles)}, Testing samples: {len(test_smiles)}")

# 创建train_DataLoader
train_datalist = []
for i in range(len(train_smiles)):
    # 将smiles和y的对应行数据传递给smiles_to_graph函数
    train_datalist.append(smiles_to_graph(train_smiles[i], train_y[i]))

# 填充结点
# 计算最大节点数量
max_train_nodes = max([data.num_nodes for data in train_datalist])
# 应用填充转换
train_transform = Compose([Pad(max_num_nodes=max_train_nodes)])
# 实例化数据集
train_dataset = SmilesToGraphDataset(train_datalist, transform=train_transform)
'''
for data in train_dataset:
    print(f"Data shape in dimension 0 (number of nodes): {data.x.size(1)}")
'''
train_data_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
''' 小数据集batch_size = 32'''

# 创建test_DataLoader
test_datalist = []
for i in range(len(test_smiles)):
    # 将smiles和y的对应行数据传递给smiles_to_graph函数
    test_datalist.append(smiles_to_graph(test_smiles[i], test_y[i]))

# 填充结点
# 计算最大节点数量
max_test_nodes = max([data.num_nodes for data in test_datalist])
# 应用填充转换
test_transform = Compose([Pad(max_num_nodes=max_test_nodes)])
# 实例化数据集
test_dataset = SmilesToGraphDataset(test_datalist, transform=test_transform)
test_data_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)
''' 小数据集batch_size = 32'''
print(f'Data preprocess is finished.')

# 模型训练
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 300
print(device)

# 搭建图神经网络
class GCN(nn.Module):
    def __init__(self, dim_in, dim_out, dim_h1, dim_h2, dim_h3, dim_h4, dim_fc, device):
        # dim_in, dim_out, dim_h1, dim_h2, dim_h3, dim_h4分别为每层输入输出前后单个结点对应的特征向量维数
        super(GCN, self).__init__()
        self.device = device
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.conv1 = GraphConv(self.dim_in, dim_h1)
        self.conv2 = GraphConv(dim_h1, dim_h2)
        self.conv3 = GraphConv(dim_h2, dim_h3)
        self.conv4 = GraphConv(dim_h3, dim_h4)
        self.dropout = nn.Dropout(0.3)
        # 定义全连接层
        self.fc = nn.Linear(dim_h4, dim_fc)  # 新的全连接层
        self.mul_regressor = nn.Linear(dim_fc, self.dim_out)

    def forward(self, data):
        # 确保 x、edge_index 和 batch 在正确的设备上
        x, edge_index, batch = data.x.to(self.device), data.edge_index.to(self.device), data.batch.to(self.device)
        x_h = self.conv1(x, edge_index)
        x_h = F.relu(x_h)
        x_h = self.conv2(x_h, edge_index)
        x_h = F.relu(x_h)
        x_h = self.conv3(x_h, edge_index)
        x_h = F.relu(x_h)
        x_h = self.conv4(x_h, edge_index)
        x_h = F.relu(x_h)
        x_h = global_mean_pool(x_h, batch)
        x_h = self.dropout(x_h)  # 使用之前定义的 dropout
        x_h = self.fc(x_h)
        x_out = self.mul_regressor(x_h)
        return x_out


class MyMessagePassingLayer(MessagePassing):
    def __init__(self, dimention_in, dimention_out):
        super(MyMessagePassingLayer, self).__init__(aggr='mean')  # 聚合方式可以选择'mean', 'sum'等
        self.lin = torch.nn.Linear(dimention_in, dimention_out)

    def forward(self, x, edge_index):
        # Step 1: 通过线性层转化特征
        x = self.lin(x)

        # Step 2: 进行消息传递
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        # Step 3: 定义消息传递的方式
        return x_j

    def update(self, aggr_out):
        # Step 4: 更新节点特征
        return torch.relu(aggr_out)


class MyGNN(torch.nn.Module):
    def __init__(self, dim_in, dim_out, dim_h1, dim_h2, dim_h3, dim_h4, dim_fc,device):
        super(MyGNN, self).__init__()
        self.device = device
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.conv1 = MyMessagePassingLayer(self.dim_in, dim_h1)  # 第1层
        self.conv2 = MyMessagePassingLayer(dim_h1, dim_h2)  # 第2层
        self.conv3 = MyMessagePassingLayer(dim_h2, dim_h3)  # 第3层
        self.conv4 = MyMessagePassingLayer(dim_h3, dim_h4)  # 第4层
        self.dropout = nn.Dropout(0.5)
        self.fc = torch.nn.Linear(dim_h4, dim_fc)  # 全连接层
        self.mul_regressor = torch.nn.Linear(dim_fc, self.dim_out)  # 输出层

    def forward(self, data):
        x, edge_index, batch = data.x.to(self.device), data.edge_index.to(self.device), data.batch.to(self.device)

        # 4层消息传递网络
        x_h = self.conv1(x, edge_index)
        x_h = F.tanh(x_h)

        x_h = self.conv2(x_h, edge_index)
        x_h = F.tanh(x_h)

        x_h = self.conv3(x_h, edge_index)
        x_h = F.tanh(x_h)

        x_h = self.conv4(x_h, edge_index)
        x_h = F.tanh(x_h)

        # 池化操作
        x_h = global_mean_pool(x_h, batch)
        x_h = self.dropout(x_h)
        # 通过全连接层
        x_h = self.fc(x_h)
        x_h = F.relu(x_h)

        # 三项特征预测层
        x_out = self.mul_regressor(x_h)
        return x_out

    # 训练模型：前向传播、计算损失、反向传播和参数更新
def train(model, data, device, optimizer, loss_func):
    model.train()  # 将模型设置为训练模式
    data = data.to(device)  # 将数据转移到指定的设备上
    optimizer.zero_grad()  # 清空优化器的梯度
    out = model(data)
    data_y = np.array(data.y)  # 将列表转换为 NumPy 数组
    targets = torch.tensor(data_y, dtype=torch.float).to(device)  # 转换为 PyTorch 张量
    loss = loss_func(out, targets)  # 确保 y 能够与出具有匹配的形状)
    loss.backward()  # 反向传播，计算梯度
    optimizer.step()  # 更新模型参数
    return loss


# 模型评估函数
def evaluate(model, data_loader):
    model.eval()  # 将模型设置为评估模式
    predictions = []
    targets = []
    with torch.no_grad():  # 设置为无梯度计算模式, 提高计算效率
        for data in data_loader:
            data = data.to(device)
            out = model(data)
            predictions.append(out.cpu())  # 将预测结果移回 CPU
            # 处理 targets，直接将 data.y 转换为张量
            target_tensor = torch.tensor(data.y, dtype=torch.float).cpu()  # 将目标转换为张量并移回 CPU
            targets.append(target_tensor)  # 添加目标到列表中
    predictions = torch.cat(predictions, dim=0)  # 合并预测结果
    targets = torch.cat(targets, dim=0)
    return predictions, targets


def calculate_accuracy(predictions, targets):
    """
    计算每个指标的准确率，并合并成一个三维向量
    :param predictions: 预测值张量，大小为 (data_nums, 3)
    :param targets: 真实值张量，大小为 (data_nums, 3)
    :return: 三个指标准确率组成的张量，大小为 (3,)
    """
    accuracies = []
    for i in range(3):  # 遍历每个指标
        # 计算预测结果和真实值的匹配程度，认为误差在0.005以内匹配
        correct_predictions = 0
        if i == 0:
            for j in range(predictions.shape[0]):
                if (abs(predictions[j, i] - targets[j, i]) <= 0.05):
                    correct_predictions += 1
        else:
            for j in range(predictions.shape[0]):
                if (abs(predictions[j, i] - targets[j, i]) <= 0.3):
                    correct_predictions += 1
        data_nums = predictions.shape[0]
        accuracy = float(correct_predictions / data_nums)
        accuracies.append(accuracy)

    accuracy_vector = torch.tensor(accuracies)
    return accuracy_vector


def show_R2(predictions, targets):
    '''
    使用r2计算每个指标的准确率，并返回R方值
    param predictions: 预测值张量，大小为 (data_nums, 3)
    param targets: 真实值张量，大小为 (data_nums, 3)
    return：R方值的列表
    '''
    predictions = np.array(predictions)
    targets = np.array(targets)

    r2_values = []

    for i in range(3):
        line = LinearRegression()
        line.fit(targets[:, i].reshape(-1, 1), predictions[:, i].reshape(-1, 1))
        r2 = r2_score(targets[:, i], predictions[:, i])
        r2_values.append(r2)

    return r2_values  # 返回R²值列表

def save_accuracy_to_file(file_path, logP_accuracy, qed_accuracy, SAS_accuracy, logP_r2, qed_r2, SAS_r2):
    """
    将准确度和R²值保存到文本文件
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"logP: {logP_accuracy:.4f}\n")
            f.write(f"qed: {qed_accuracy:.4f}\n")
            f.write(f"SAS: {SAS_accuracy:.4f}\n")
            f.write(f"\nTraining-set R²: logP = {logP_r2:.4f}, qed = {qed_r2:.4f}, SAS = {SAS_r2:.4f}\n")
    except IOError as e:
        print(f"Error writing to file {file_path}: {e}")

# 声明模型
model = GCN(dim_in=6, dim_out=3, dim_h1=128, dim_h2=256, dim_h3=128, dim_h4=64, dim_fc=32, device=device).to(device)

# 损失函数：均方差
loss_func = nn.MSELoss()
# 优化器：Adam
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

for epoch in range(num_epochs):
    print(f'training epoch {epoch + 1}...')
    # 每个 epoch 的损失累积
    epoch_loss = 0.0
    for data in train_data_loader:
        loss = train(model, data, device, optimizer, loss_func)
        epoch_loss += loss.item()

        # 计算当前 epoch 的平均损失
    average_epoch_loss = epoch_loss / len(train_data_loader)
    scheduler.step(average_epoch_loss)

    if (epoch % 20) == 0:
        # 在每20个epoch后进行训练数据评估
        train_predictions, train_targets = evaluate(model, train_data_loader)
        train_predictions = train_predictions.cpu()
        train_targets = train_targets.cpu()

        # 计算准确度和R²值
        train_accuracy = calculate_accuracy(train_predictions, train_targets)
        train_r2_values = show_R2(train_predictions, train_targets)

        # 计算MSE和MAE
        mse = nn.MSELoss()(train_predictions, train_targets).item()
        mae = torch.mean(torch.abs(train_predictions - train_targets)).item()

        print(
            f"Training-set R² at {epoch} epoch: logP = {train_r2_values[0]:.4f}, qed = {train_r2_values[1]:.4f}, SAS = {train_r2_values[2]:.4f}")
        print(
            f"Training-set accuracy at {epoch} epoch: logP - {train_accuracy[0]:.4f}, qed - {train_accuracy[1]:.4f}, SAS - {train_accuracy[2]:.4f}")
        print(f"Training-set MSE: {mse:.4f}, MAE: {mae:.4f}")

    if (epoch % 20 == 0):
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
        # 每10个epoch保存一次模型权重
        model_path = f'model1_weights/model1_weights_{epoch + 1}.pth'
        torch.save(model.state_dict(), model_path)

    # 在最后一个 epoch 时更新最终损失
final_loss = average_epoch_loss
model_path = f'model1_weights/model1_weights_{epoch + 1}.pth'
torch.save(model.state_dict(), model_path)

# 训练数据评估
train_predictions, train_targets = evaluate(model, train_data_loader)
train_predictions = train_predictions.cpu()
train_targets = train_targets.cpu()

# 计算准确度和R²值
train_accuracy = calculate_accuracy(train_predictions, train_targets)
train_r2_values = show_R2(train_predictions, train_targets)

# 计算最终的MSE和MAE
final_mse = nn.MSELoss()(train_predictions, train_targets).item()
final_mae = torch.mean(torch.abs(train_predictions - train_targets)).item()

print(f"Final loss: {final_loss:.4f}")
print(
    f"Training-set R²: logP = {train_r2_values[0]:.4f}, qed = {train_r2_values[1]:.4f}, SAS = {train_r2_values[2]:.4f}")
print(
    f"Training-set accuracy: logP - {train_accuracy[0]:.4f}, qed - {train_accuracy[1]:.4f}, SAS - {train_accuracy[2]:.4f}")
print(f"Final Training-set MSE: {final_mse:.4f}, MAE: {final_mae:.4f}")

# 保存训练准确度到文本文件
train_data_path = 'train_accuracy.txt'
save_accuracy_to_file(train_data_path, train_accuracy[0], train_accuracy[1], train_accuracy[2],
                      train_r2_values[0], train_r2_values[1], train_r2_values[2])

# 测试数据评估
test_predictions, test_targets = evaluate(model, test_data_loader)
test_predictions = test_predictions.cpu()  # 确保在CPU上
test_targets = test_targets.cpu()  # 确保在CPU上

# 计算测试准确度和R²值
test_accuracy = calculate_accuracy(test_predictions, test_targets)
test_r2_values = show_R2(test_predictions, test_targets)  # 获取R²值

# 计算测试集的MSE和MAE
test_mse = nn.MSELoss()(test_predictions, test_targets).item()
test_mae = torch.mean(torch.abs(test_predictions - test_targets)).item()

print(f"Testing-set R²: logP = {test_r2_values[0]:.4f}, qed = {test_r2_values[1]:.4f}, SAS = {test_r2_values[2]:.4f}")
print(f"Testing-set accuracy: logP - {test_accuracy[0]:.4f}, qed - {test_accuracy[1]:.4f}, SAS - {test_accuracy[2]:.4f}")
print(f"Testing-set MSE: {test_mse:.4f}, MAE: {test_mae:.4f}")

# 保存测试准确度到文本文件
test_data_path = 'test_accuracy.txt'
save_accuracy_to_file(test_data_path, test_accuracy[0], test_accuracy[1], test_accuracy[2],
                      test_r2_values[0], test_r2_values[1], test_r2_values[2])

# 保存最终损失到文本文件
final_loss_path = 'final_loss.txt'
try:
    with open(final_loss_path, 'w') as f:
        f.write(f"Final Loss: {final_loss:.4f}\n")
        f.write(f"Final Training-set MSE: {final_mse:.4f}, MAE: {final_mae:.4f}\n")
        f.write(f"Final Testing-set MSE: {test_mse:.4f}, MAE: {test_mae:.4f}\n")
except IOError as e:
    print(f"Error writing to file {final_loss_path}: {e}")
