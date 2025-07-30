import torch
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
import os
# from torch.optim.lr_scheduler import ReduceLROnPlateau # <-- 新增: 导入学习率调度器
from torch.optim.lr_scheduler import CosineAnnealingLR  # 导入 CosineAnnealingLR
from torch_geometric.utils import scatter

# 假设模型和调度器已经定义好
# from e_dit_network import E_DiT_Network
# from scheduler import HierarchicalDiffusionScheduler

# ==============================================================================
# 1. 辅助函数 (Helper Functions)
# ==============================================================================

def scale_to_unit_sphere(pos: torch.Tensor, batch_map: torch.Tensor) -> torch.Tensor:
    """
    将批次中每个图的坐标独立地缩放到单位球内。
    此版本确保了设备一致性。

    Args:
        pos (torch.Tensor): 批次中所有节点的坐标张量, shape [N, 3]。
        batch_map (torch.Tensor): 将每个节点映射到其所属图的向量, shape [N]。

    Returns:
        torch.Tensor: 缩放后的坐标，与输入张量在同一设备上。
    """
    # PyG 的 scatter 函数可以高效地按组求和/求均值

    # 计算每个节点到其质心的距离
    distances = torch.linalg.norm(pos, dim=1)

    # 按图分组，计算每个图中的最大距离
    # scatter 函数会保持设备一致性
    max_distances = scatter(distances, batch_map, dim=0, reduce='max')

    # 计算每个图的缩放因子，加上一个小的 epsilon 防止除以零
    # 索引操作同样会保持设备一致性
    scale_factors = max_distances[batch_map].unsqueeze(1) + 1e-8

    # 缩放坐标，最终结果与输入 pos 在同一个设备上
    return pos / scale_factors


def noise_discrete_features(
    features_0: torch.Tensor,
    Q_bar: torch.Tensor,
    t_per_item: torch.Tensor
) -> torch.Tensor:
    """
    对 one-hot 编码的离散特征（如原子类型、边类型）进行加噪。

    Args:
        features_0 (torch.Tensor): 干净的 one-hot 特征, shape [M, K] (M个项目, K个类别)。
        Q_bar (torch.Tensor):      转移矩阵集合, shape [T, K, K] (T个时间步)。
        t_per_item (torch.Tensor): 每个项目对应的时间步, shape [M]。

    Returns:
        torch.Tensor: 加噪后的 one-hot 特征。
    """
    # 1. 根据每个项目的时间步 t，从 Q_bar 中选出对应的转移矩阵
    # Q_bar_t 的 shape 为 [M, K, K]
    Q_bar_t = Q_bar[t_per_item]
    Q_bar_t = Q_bar_t.to(features_0.device)
    
    # 2. 计算加噪后的概率分布
    # features_0.unsqueeze(1) -> [M, 1, K]   在 features_0 的第1个维度上增加一个维度
    # Q_bar_t                 -> [M, K, K]
    # prob_t                  -> [M, 1, K]
    # 执行批量矩阵乘法
    prob_t = torch.bmm(features_0.unsqueeze(1), Q_bar_t).squeeze(1) # shape: [M, K]
    
    # 3. 根据概率分布进行采样，得到新的类别索引
    # torch.multinomial 要求输入是概率，我们这里已经是概率了
    # multinomial 会把每一行都看作是一个独立的“骰子”的概率设置
    # num_samples=1 指每行采样一次，这时输出维度为[M, 1]
    # .squeeze(-1) 作用为移除最后一个维度（dim=-1）上大小为1的维度
    sampled_indices = torch.multinomial(prob_t, num_samples=1).squeeze(-1) # shape: [M]
    
    # 4. 将采样出的索引转换回 one-hot 编码
    num_classes = features_0.shape[1]  # 获取类别的总数
    # 将整数类别索引转换成 One-Hot 编码向量
    features_t = torch.nn.functional.one_hot(sampled_indices, num_classes=num_classes).float()
    
    return features_t

# ==============================================================================
# 2. 损失函数框架 (Loss Function Skeletons)
# ==============================================================================

# 2.1 原子类型损失
# 推荐使用的、更简洁的损失函数
def calculate_atom_type_loss(
    pred_logits: torch.Tensor,   # 模型对 x0 的预测 logits, shape [M, C]
    true_x0_indices: torch.Tensor, # 真实的 x0 类别索引, shape [M]
    lambda_aux: float = 0.001,     # D3PM论文建议的小值
) -> torch.Tensor:
    """
    计算基于 D3PM 混合损失 L_λ 的简化版原子类型损失。
    这本质上是一个加权的交叉熵损失。

    Args:
        pred_logits: 模型的 logits 输出。
        true_x0_indices: 真实的类别索引。
        t: 每个项目的时间步。
        lambda_aux: 辅助损失的权重。
        T: 噪声过程的总步长。

    Returns:
        torch.Tensor: 该批次的平均损失。
    """

    ### --- 修正：为保证数值稳定性，将 logits 转换为 float32 --- ###
    # F.cross_entropy 在 float16 下可能不稳定
    stable_logits = pred_logits.to(torch.float32)

    # 1. 计算标准的交叉熵损失 (对应于 L_vlb 的主要部分和 L_aux)
    # reduction='none' 表示我们为批次中的每个项目计算一个损失值
    loss = F.cross_entropy(stable_logits, true_x0_indices, reduction='none')
    
    # 2. 根据 D3PM 混合损失 L_λ 的思想，应用权重
    # L_λ = L_vlb + λ * [-log p(x0|xt)]
    # L_vlb 的 KL 项在 t>1 时权重为1，在 t=1 时权重为1(重建项)。
    # L_aux 在所有 t 上权重都为 λ。
    # 所以，总权重为 1 + λ，除了 t=0 的情况（我们不处理）。
    
    # 一个非常常见的简化实现是直接应用权重
    # 另一种来自其他论文的思路是给低t的损失更高的权重
    # 这里我们采用 L_λ 的精神：一个基础损失 + 一个小的辅助损失
    
    # 权重为 1(来自L_vlb) + lambda_aux (来自L_aux)
    final_loss = (1 + lambda_aux) * loss
    
    # 对于 t=1 的特殊情况，VLB中只有重建项，可以认为权重不同
    # 但D3PM的混合损失简化了这一点，我们在此也采用简化
    
    return final_loss.mean()

# 2.2 原子坐标损失
def calculate_coordinate_loss_wrapper(
    predicted_r0: torch.Tensor,      # 模型预测的干净坐标 (t=0), shape [M, 3]
    true_noise: torch.Tensor,        # 用于生成 r_t 的真实高斯噪声, shape [M, 3]
    r_t: torch.Tensor,               # 输入到模型的加噪坐标 (t>0), shape [M, 3]
    t: torch.Tensor,                 # 每个坐标对应的时间步, shape [M]
    scheduler,                       # HierarchicalDiffusionScheduler 实例
    schedule_type: str               # 使用的调度类型, 'alpha' 或 'delta'
) -> torch.Tensor:
    """
    计算原子坐标的损失。

    这是一个包装函数，它接收模型预测的 r0，使用调度器将其转换为
    预测的噪声 epsilon，然后计算与真实噪声的 L2 损失。

    Args:
        predicted_r0: 模型预测的干净坐标。
        true_noise: 真实的噪声。
        r_t: 加噪后的坐标。
        t: 时间步。
        scheduler: 噪声调度器实例。
        schedule_type: 使用的调度类型 ('alpha' 或 'delta')。

    Returns:
        torch.Tensor: 计算出的标量损失值。
    """
    # 1. 检查输入是否为空。
    # 在策略II中，如果目标原子被某种方式移除了（虽然不太可能），这可以防止出错。
    if predicted_r0.shape[0] == 0:
        return torch.tensor(0.0, device=predicted_r0.device)

    # 2. 调用调度器的核心方法，从 predicted_r0 反推出 predicted_noise
    predicted_noise = scheduler.get_predicted_noise_from_r0(
        r_t=r_t,
        t=t,
        predicted_r0=predicted_r0,
        schedule_type=schedule_type
    )

    # 3. 计算预测噪声和真实噪声之间的 L2 损失 (均方误差, Mean Squared Error)
    # F.mse_loss(A, B) 会计算 (A - B)^2 的所有元素的平均值。
    loss = F.mse_loss(predicted_noise, true_noise)
    
    return loss

# 2.3 边类型损失
def calculate_bond_type_loss(
    pred_logits: torch.Tensor,      # 模型对干净边类型的预测 logits, shape [M_edges, C_bonds]
    true_b0_indices: torch.Tensor, # 真实的干净边类型索引, shape [M_edges]
    lambda_aux: float = 0.001,     # 辅助损失的权重
) -> torch.Tensor:

    """
    计算基于 D3PM 混合损失 L_λ 的简化版边类型损失。
    这本质上是一个加权的交叉熵损失，与原子类型损失的逻辑完全相同。

    Args:
        pred_logits: 模型的 logits 输出。
        true_b0_indices: 真实的边类别索引。
        t: 每个边对应的时间步。
        lambda_aux: 辅助损失的权重。
        T: 噪声过程的总步长 (当前未使用，为未来扩展保留)。

    Returns:
        torch.Tensor: 该批次的平均损失。
    """

    print("--- DEBUG: Bond Logits ---")
    print(f"Shape: {pred_logits.shape}, Dtype: {pred_logits.dtype}")
    if pred_logits.numel() > 0:
        print(f"Min: {pred_logits.min().item()}, Max: {pred_logits.max().item()}, Mean: {pred_logits.mean().item()}")
    print(f"Contains NaN: {torch.isnan(pred_logits).any().item()}")
    print(f"Contains Inf: {torch.isinf(pred_logits).any().item()}")
    print("--------------------------")
    # 1. 检查输入是否为空。如果一个批次中没有需要预测的边，则损失为0。
    # 这在策略II中，如果目标原子是孤立点时可能发生。
    if pred_logits.shape[0] == 0:
        return torch.tensor(0.0, device=pred_logits.device)

    stable_logits = pred_logits.to(torch.float32)
        
    # 2. 计算标准的交叉熵损失
    # reduction='none' 表示我们为批次中的每条边计算一个损失值
    loss = F.cross_entropy(stable_logits, true_b0_indices, reduction='none')
    
    # 3. 应用 D3PM 混合损失的简化权重
    # 总权重 = 1 (来自 L_vlb) + lambda_aux (来自 L_aux)
    final_loss = (1 + lambda_aux) * loss
    
    # 4. 对批次中的所有边的损失求平均，得到最终的标量损失
    return final_loss.mean()


# ==============================================================================
# 3. 验证函数 (Validation Function)
# ==============================================================================
@torch.no_grad() # 装饰器，表示该函数内所有 torch 计算都不需要记录梯度
def validate(val_loader, model, s_model, scheduler, args, amp_autocast, fragmenter):
    """
    在验证集上评估模型损失。
    此函数的前向传播和损失计算逻辑与训练过程完全一致。
    """
    device = args.device
    model.eval() # 将模型设置为评估模式
    total_val_loss = 0.0
    total_s_val_loss_epoch = 0.0
    pbar_val = tqdm(val_loader, desc=f"Validating", leave=False)

    for batch_non_fc, batch_fc in pbar_val:
        batch_fc = batch_fc.to(device)
        batch_non_fc = batch_non_fc.to(device)

        with amp_autocast():
            # 将非全连接的 batch 送入排序网络
            orders, log_prob_orders = s_model(batch_non_fc)

            clean_batch = seg(batch_fc, orders, fragmenter)

            # --- [逻辑与训练循环完全相同] ---

            # --- 0. 准备工作 ---
            num_graphs, num_nodes, num_edges = clean_batch.num_graphs, clean_batch.num_nodes, clean_batch.num_edges
            scaled_pos = scale_to_unit_sphere(clean_batch.pos, clean_batch.batch)
            t1 = torch.randint(1, scheduler.T1 + 1, (num_graphs,), device=device)
            t2 = torch.randint(1, scheduler.T2 + 1, (num_graphs,), device=device)
            noise1, noise2 = torch.randn_like(scaled_pos), torch.randn_like(scaled_pos)
            t1_per_node, t1_per_edge = t1[clean_batch.batch], t1[clean_batch.batch[clean_batch.edge_index[0]]]

            # --- 策略 I: 全局去噪 ---
            noised_pos_I = scheduler.q_sample(scaled_pos, t1_per_node, noise1, 'alpha')
            noised_x_I = noise_discrete_features(clean_batch.x, scheduler.Q_bar_alpha_a, t1_per_node)
            noised_edge_attr_I = noise_discrete_features(clean_batch.edge_attr, scheduler.Q_bar_alpha_b, t1_per_edge)
            noised_data_I = clean_batch.clone(); noised_data_I.pos, noised_data_I.x, noised_data_I.edge_attr = noised_pos_I, noised_x_I, noised_edge_attr_I
            
            target_node_mask_I = torch.ones(num_nodes, dtype=torch.bool, device=device)
            target_edge_mask_I = torch.ones(num_edges, dtype=torch.bool, device=device)
            
            predictions_I = model(noised_data_I, t1, target_node_mask_I, target_edge_mask_I)
            
            lossI_a = calculate_atom_type_loss(predictions_I['atom_type_logits'], clean_batch.x.argmax(dim=-1), t1_per_node, args.lambda_aux, scheduler.T_full)
            lossI_r = calculate_coordinate_loss_wrapper(predictions_I['predicted_r0'], noise1, noised_pos_I, t1_per_node, scheduler, 'alpha')
            lossI_b = calculate_bond_type_loss(predictions_I['bond_logits'], clean_batch.edge_attr.argmax(dim=-1), t1_per_edge, args.lambda_aux, scheduler.T_full)
            loss_I = args.w_a * lossI_a + args.w_r * lossI_r + args.w_b * lossI_b
    
            # --- 策略 II: 局部生成 ---
            target_node_mask_II = clean_batch.is_new_node.squeeze().bool()
            context_node_mask_II = ~target_node_mask_II
            target_edge_mask = (target_node_mask_II[clean_batch.edge_index[0]] | target_node_mask_II[clean_batch.edge_index[1]])
            context_edge_mask = ~target_edge_mask
            
            t_T1_per_node, t_T1_per_edge = torch.full_like(t1_per_node, scheduler.T1), torch.full_like(t1_per_edge, scheduler.T1)
            t2_per_node, t2_per_edge = t2[clean_batch.batch], t2[clean_batch.batch[clean_batch.edge_index[0]]]
    
            noised_pos_context = scheduler.q_sample(scaled_pos[context_node_mask_II], t_T1_per_node[context_node_mask_II], noise2[context_node_mask_II], 'alpha')
            noised_pos_target = scheduler.q_sample(scaled_pos[target_node_mask_II], t2_per_node[target_node_mask_II], noise2[target_node_mask_II], 'delta')
            noised_pos_II = torch.zeros_like(scaled_pos); noised_pos_II[context_node_mask_II], noised_pos_II[target_node_mask_II] = noised_pos_context, noised_pos_target
            
            noised_x_context = noise_discrete_features(clean_batch.x[context_node_mask_II], scheduler.Q_bar_alpha_a, t_T1_per_node[context_node_mask_II])
            noised_x_target = noise_discrete_features(clean_batch.x[target_node_mask_II], scheduler.Q_bar_gamma_a, t2_per_node[target_node_mask_II])
            noised_x_II = torch.zeros_like(clean_batch.x); noised_x_II[context_node_mask_II], noised_x_II[target_node_mask_II] = noised_x_context, noised_x_target
        
            noised_edge_attr_context = noise_discrete_features(clean_batch.edge_attr[context_edge_mask], scheduler.Q_bar_alpha_b, t_T1_per_edge[context_edge_mask])
            noised_edge_attr_target = noise_discrete_features(clean_batch.edge_attr[target_edge_mask], scheduler.Q_bar_gamma_b, t2_per_edge[target_edge_mask])
            noised_edge_attr_II = torch.zeros_like(clean_batch.edge_attr); noised_edge_attr_II[context_edge_mask], noised_edge_attr_II[target_edge_mask] = noised_edge_attr_context, noised_edge_attr_target
        
            noised_data_II = clean_batch.clone(); noised_data_II.pos, noised_data_II.x, noised_data_II.edge_attr = noised_pos_II, noised_x_II, noised_edge_attr_II
        
            predictions_II = model(noised_data_II, t2, target_node_mask_II, target_edge_mask)

            lossII_a = calculate_atom_type_loss(predictions_II['atom_type_logits'], clean_batch.x[target_node_mask_II].argmax(dim=-1), t2_per_node[target_node_mask_II], args.lambda_aux, scheduler.T_full)
            lossII_r = calculate_coordinate_loss_wrapper(predictions_II['predicted_r0'], noise2[target_node_mask_II], noised_pos_target, t2_per_node[target_node_mask_II], scheduler, 'delta')
            lossII_b = calculate_bond_type_loss(predictions_II['bond_logits'], clean_batch.edge_attr[target_edge_mask].argmax(dim=-1), t2_per_edge[target_edge_mask], args.lambda_aux, scheduler.T_full)
            loss_II = args.w_a * lossII_a + args.w_r * lossII_r + args.w_b * lossII_b

            # --- 总验证损失 ---
            total_loss = scheduler.T1 * loss_I + scheduler.T2 * loss_II 

            reward = (-total_loss - log_prob_orders).detach()  # 奖励必须从计算图中分离，不带梯度

            # 策略损失 L_policy = -R(π) * log q_φ(π|G)
            loss_s_model = (-reward * log_prob_orders).mean()
        
        total_val_loss += total_loss.item()
        total_s_val_loss_epoch += loss_s_model.item()
        pbar_val.set_postfix({
            'loss_G': f"{total_loss.item():.2f}", 
            'loss_S': f"{loss_s_model.item():.2f}"
        })
    
    avg_val_loss = total_val_loss / len(val_loader)
    avg_s_val_loss = total_s_val_loss_epoch / len(val_loader)
    return avg_val_loss, avg_s_val_loss

def seg(batch_fc, orders, fragmenter):
    """
    从批次分子图和对应的排序列表生成子图批次
    
    Args:
        batch_fc: 从DataLoader获取的批次分子图（PyG的Batch对象）
        orders: 与批次对应的排序列表集合，每个元素是一个分子图的节点排序
        
    Returns:
        clean_batch: 由子图组成的新批次（PyG的Batch对象）
    """
    # 确保输入批次大小与排序数量一致
    assert len(orders) == batch_fc.num_graphs, \
        f"排序数量({len(orders)})与批次中分子图数量({batch_fc.num_graphs})不匹配"
    
    # 存储所有生成的子图
    subgraphs = []
    
    # 拆分批次为单个分子图（使用PyG的unbind方法）
    single_graphs = batch_fc.to_data_list()
    
    # 循环处理每个分子图及其排序
    for graph, order in zip(single_graphs, orders):
        # 生成子图
        subgraph = fragmenter(graph, order)
        subgraphs.append(subgraph)
    
    # 将所有子图组合成新的批次（使用PyG的Batch.from_data_list方法）
    from torch_geometric.data import Batch
    clean_batch = Batch.from_data_list(subgraphs)
    
    return clean_batch.to(batch_fc.x.device)


# ==============================================================================
# 4. 主训练函数 (Main Training Function)
# ==============================================================================

def train(
    args,
    logger,
    train_loader, # 训练数据
    val_loader, # 测试数据
    s_model, # 排序网络实例
    model, # E_DiT_Network 实例
    scheduler, # HierarchicalDiffusionScheduler 实例
    subgraph_generator_fn,
    amp_autocast,
    loss_scaler
):
    """
    主训练函数。
    """
    device = args.device
    model.to(device)
    s_model.to(device)

    # 创建优化器
    optimizer_model = optim.Adam(model.parameters(), lr=args.learning_rate)
    optimizer_s_model = optim.Adam(s_model.parameters(), lr=args.s_learning_rate) # s_model有独立的学习率

    # 创建 CosineAnnealingLR 调度器
    T_max = args.epochs  # 最大迭代次数，通常设置为总 epoch 数
    lr_min_factor = args.lr_min_factor
    scheduler_model = CosineAnnealingLR(optimizer_model, T_max=T_max, eta_min=lr_min_factor * args.learning_rate)
    scheduler_s_model = CosineAnnealingLR(optimizer_s_model, T_max=T_max, eta_min=lr_min_factor * args.s_learning_rate)

    best_val_loss = float('inf')
    best_epoch = 0

    logger.info(f"模型检查点将保存在: {args.checkpoints_dir}")
    logger.info("开始训练...")

    for epoch in range(1, args.epochs + 1):
        # 将模型设置为“训练模式”
        # 它会通知模型中所有具有不同训练/评估行为的层（主要是 Dropout 层和 BatchNorm 层）切换到它们的训练状态。
        # Dropout 层在训练时会随机“丢弃”一些神经元，以防止过拟合；在评估时则不会丢弃，会使用所有神经元。
        # BatchNorm 层在训练时会使用当前批次的均值和方差进行归一化，并更新其内部的全局统计量；在评估时则会使用已学习到的全局统计量。
        model.train()  
        s_model.train()

        total_loss_epoch = 0.0
        total_s_loss_epoch = 0.0
        
        # 创建数据迭代器
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for step, (batch_non_fc, batch_fc) in enumerate(pbar_train):
            batch_fc = batch_fc.to(device)
            batch_non_fc = batch_non_fc.to(device) # 将当前批次的数据（包括所有张量，如 pos, x, edge_index 等）一次性地移动到之前定义好的目标设备 device 上
             
            # 在每个 step 开始时清零梯度
            optimizer_model.zero_grad()
            optimizer_s_model.zero_grad()
            with torch.autograd.set_detect_anomaly(True):

                # --- 使用 amp_autocast 上下文管理器 ---
                # 将所有的前向传播和损失计算包裹在内
                with amp_autocast():

                    # 将非全连接的 batch 送入排序网络，返回排序和损失
                    orders, log_prob_orders = s_model(batch_non_fc)

                    # 得到切分后的子图片段
                    clean_batch = seg(batch_fc, orders, subgraph_generator_fn)

                    # --- 0. 准备工作 ---
                    num_graphs = clean_batch.num_graphs # 批次中包含的独立图的数量（等于 batch_size）。用于采样图级别的变量，如时间步 t
                    num_nodes = clean_batch.num_nodes #  批次中所有图的节点总数
                    num_edges = clean_batch.num_edges # 批次中所有图的边总数

                    # a. 坐标缩放
                    scaled_pos = scale_to_unit_sphere(clean_batch.pos, clean_batch.batch)

                    # b. 采样时间步和高斯噪声
                    # 为批次中的每一个图随机采样一个时间步 t1
                    # t1 是一个形状为 [batch_size] 的张量，例如 tensor([18, 98, 21, ...])
                    t1 = torch.randint(1, scheduler.T1 + 1, (num_graphs,), device=device)
                    t2 = torch.randint(1, scheduler.T2 + 1, (num_graphs,), device=device)
                    # noise1 是一个形状为 [N, 3] 的张量，其中每个元素都是一个随机数（均值为0，方差为1）。noise1[i] 就是要加到第 i 个原子坐标上的噪声向量。
                    noise1 = torch.randn_like(scaled_pos)
                    noise2 = torch.randn_like(scaled_pos)

                    # c. 将 per-graph 的时间步扩展到 per-node 和 per-edge
                    # t1: 一个形状为 [num_graphs] 的张量。假设 batch_size=4，t1可能长这样：tensor([18, 98, 21, 76])
                    # clean_batch.batch: 一个形状为 [num_nodes] 的张量，记录了每个节点属于哪个图。
                    # 它可能长这样（假设4个图分别有3, 2, 4, 3个节点）：tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3])。
                    t1_per_node = t1[clean_batch.batch] # 形状为[num_nodes]
                    t1_per_edge = t1[clean_batch.batch[clean_batch.edge_index[0]]] # 形状为[num_edges]


                    # --- 策略 I: 全局去噪 (生成噪声图 Ⅰ) ---

                    # a. 加噪坐标
                    noised_pos_I = scheduler.q_sample(scaled_pos, t1_per_node, noise1, schedule_type='alpha')

                    # b. 加噪原子类型
                    noised_x_I = noise_discrete_features(clean_batch.x, scheduler.Q_bar_alpha_a, t1_per_node)

                    # c. 加噪边属性
                    noised_edge_attr_I = noise_discrete_features(clean_batch.edge_attr, scheduler.Q_bar_alpha_b, t1_per_edge)

                    # d. 构建加噪后的数据对象 Ⅰ
                    # 复制干净的数据，更改加噪的部分
                    noised_data_I = clean_batch.clone()
                    noised_data_I.pos = noised_pos_I
                    noised_data_I.x = noised_x_I
                    noised_data_I.edge_attr = noised_edge_attr_I

                    # e. 准备模型输入
                    # 创建一个长度为当前批次中所有原子的总数，内容全为 True 的向量。
                    target_node_mask_I = torch.ones(num_nodes, dtype=torch.bool, device=device)
                    # 处理并输出所有边的预测结果
                    target_edge_mask_I = torch.ones(num_edges, dtype=torch.bool, device=device)

                    # f. 模型前向传播
                    predictions_I = model(noised_data_I, t1, target_node_mask_I, target_edge_mask_I)

                    # g. 计算损失 Ⅰ
                    lossI_a = calculate_atom_type_loss(
                        predictions_I['atom_type_logits'],
                        clean_batch.x.argmax(dim=-1),  # 从 One-Hot 编码的特征张量中，提取出每个项目对应的类别索引 (class index)
                        args.lambda_aux
                    )
                    lossI_r = calculate_coordinate_loss_wrapper(
                        predicted_r0=predictions_I['predicted_r0'],
                        true_noise=noise1,
                        r_t=noised_pos_I,
                        t=t1_per_node,
                        scheduler=scheduler,
                        schedule_type='alpha'
                    )
                    lossI_b = calculate_bond_type_loss(
                        pred_logits=predictions_I['bond_logits'],
                        true_b0_indices=clean_batch.edge_attr.argmax(dim=-1),
                        lambda_aux=args.lambda_aux
                    )
                    loss_I = args.w_a * lossI_a + args.w_r * lossI_r + args.w_b * lossI_b


                    # --- 策略 II: 局部生成 (生成噪声图 Ⅱ) ---

                    # a. 识别上下文和目标
                    # 标识哪些节点是我们的预测目标
                    target_node_mask_II = clean_batch.is_new_node.squeeze() # is_new_node 就是我们的目标mask，维度压缩为[num_nodes]
                    target_node_mask_II = target_node_mask_II.to(torch.bool)
                    # 标识哪些节点是上下文节点，用于对上下文节点加噪
                    context_node_mask_II = ~target_node_mask_II
                    # 标识哪些边是与预测目标节点相关的边
                    # 对于第 i 条边，如果它的起点是目标节点或者它的终点是目标节点，那么它就是需要被预测的边
                    target_edge_mask = (target_node_mask_II[clean_batch.edge_index[0]] | target_node_mask_II[clean_batch.edge_index[1]])
                    # 用于对上下文边加噪
                    context_edge_mask = ~target_edge_mask

                    # b. 准备时间步 (T1 和 t2)
                    # 创建一个与给定张量形状相同、类型相同、设备相同的新张量，并将所有元素填充为T1
                    t_T1_per_node = torch.full_like(t1_per_node, fill_value=scheduler.T1)
                    t_T1_per_edge = torch.full_like(t1_per_edge, fill_value=scheduler.T1)
                    t2_per_node = t2[clean_batch.batch]
                    t2_per_edge = t2[clean_batch.batch[clean_batch.edge_index[0]]]

                    # c. 对上下文和目标分别加噪
                    # 坐标
                    # 计算出所有上下文原子的加噪后坐标
                    noised_pos_context = scheduler.q_sample(scaled_pos[context_node_mask_II], t_T1_per_node[context_node_mask_II], noise2[context_node_mask_II], 'alpha')
                    # 计算出所有目标原子的加噪后坐标
                    noised_pos_target = scheduler.q_sample(scaled_pos[target_node_mask_II], t2_per_node[target_node_mask_II], noise2[target_node_mask_II], 'delta')
                    # 创建一个空的“画布”
                    noised_pos_II = torch.zeros_like(scaled_pos)
                    # 将计算好的上下文坐标填充到画布的相应位置
                    noised_pos_II[context_node_mask_II] = noised_pos_context
                    # 将计算好的目标坐标填充到画布的相应位置
                    noised_pos_II[target_node_mask_II] = noised_pos_target

                    # 原子类型
                    noised_x_context = noise_discrete_features(clean_batch.x[context_node_mask_II], scheduler.Q_bar_alpha_a, t_T1_per_node[context_node_mask_II])
                    noised_x_target = noise_discrete_features(clean_batch.x[target_node_mask_II], scheduler.Q_bar_gamma_a, t2_per_node[target_node_mask_II])
                    noised_x_II = torch.zeros_like(clean_batch.x)
                    noised_x_II[context_node_mask_II] = noised_x_context
                    noised_x_II[target_node_mask_II] = noised_x_target

                    # 边属性
                    noised_edge_attr_context = noise_discrete_features(clean_batch.edge_attr[context_edge_mask], scheduler.Q_bar_alpha_b, t_T1_per_edge[context_edge_mask])
                    noised_edge_attr_target = noise_discrete_features(clean_batch.edge_attr[target_edge_mask], scheduler.Q_bar_gamma_b, t2_per_edge[target_edge_mask])
                    noised_edge_attr_II = torch.zeros_like(clean_batch.edge_attr)
                    noised_edge_attr_II[context_edge_mask] = noised_edge_attr_context
                    noised_edge_attr_II[target_edge_mask] = noised_edge_attr_target

                    # d. 构建加噪后的数据对象 Ⅱ
                    noised_data_II = clean_batch.clone()
                    noised_data_II.pos = noised_pos_II
                    noised_data_II.x = noised_x_II
                    noised_data_II.edge_attr = noised_edge_attr_II


                    # f. 模型前向传播 (注意时间步传入的是 t2)
                    predictions_II = model(noised_data_II, t2, target_node_mask_II, target_edge_mask)

                    # g. 计算损失 Ⅱ
                    # 注意：这里的真实标签和噪声都需要根据 mask 进行筛选
                    lossII_a = calculate_atom_type_loss(
                        predictions_II['atom_type_logits'],
                        clean_batch.x[target_node_mask_II].argmax(dim=-1),
                        args.lambda_aux
                    )
                    lossII_r = calculate_coordinate_loss_wrapper(
                        predicted_r0=predictions_II['predicted_r0'],
                        true_noise=noise2[target_node_mask_II],
                        r_t=noised_pos_target,
                        t=t2_per_node[target_node_mask_II],
                        scheduler=scheduler,
                        schedule_type='delta'
                    )
                    lossII_b = calculate_bond_type_loss(
                        pred_logits=predictions_II['bond_logits'],
                        true_b0_indices=clean_batch.edge_attr[target_edge_mask].argmax(dim=-1),
                        lambda_aux=args.lambda_aux
                    )
                    loss_II = args.w_a * lossII_a + args.w_r * lossII_r + args.w_b * lossII_b


                    # --- 总损失与反向传播 ---
                    total_loss = scheduler.T1 * loss_I + scheduler.T2 * loss_II

                     # 根据 OM 论文, 奖励 R(π) = log p_θ(G|π) - log q_φ(π|G)
                    # log p_θ(G|π) 由生成模型的负损失 -total_loss 近似
                    # log q_φ(π|G) 是排序网络输出的 log_prob_orders
                    # 我们希望最大化 E[R(π)], 所以将 R(π) 作为 REINFORCE 的奖励
                    reward = (-total_loss - log_prob_orders).detach()  # 奖励必须从计算图中分离，不带梯度

                    # 策略损失 L_policy = -R(π) * log q_φ(π|G)
                    loss_s_model = (-reward * log_prob_orders).mean()


                # --- 4. 反向传播与参数更新 ---
                # 使用 loss_scaler 分别缩放两个模型的损失
                # backward() 会累加梯度，而我们已经在 step 开始时清零了梯度

                if loss_scaler is not None:
                    # --- AMP 启用模式 ---
                    # 1. 分别对两个损失进行缩放和反向传播
                    loss_scaler._scaler.scale(total_loss).backward()
                    loss_scaler._scaler.scale(loss_s_model).backward()

                    # 2. 对 model 的梯度进行反缩放，然后裁剪
                    loss_scaler._scaler.unscale_(optimizer_model)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    # 3. 对 s_model 的梯度进行反缩放，然后裁剪
                    loss_scaler._scaler.unscale_(optimizer_s_model)
                    torch.nn.utils.clip_grad_norm_(s_model.parameters(), max_norm=1.0)

                    # 4. 更新两个优化器
                    loss_scaler._scaler.step(optimizer_model)
                    loss_scaler._scaler.step(optimizer_s_model)

                    # 5. 更新缩放器
                    loss_scaler._scaler.update()
                else:
                    # --- AMP 禁用模式 (标准 float32 训练) ---
                    # 1. 直接对两个损失进行反向传播
                    total_loss.backward()
                    loss_s_model.backward()

                    # 2. 直接对两个模型的梯度进行裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(s_model.parameters(), max_norm=1.0)

                    # 3. 更新两个优化器
                    optimizer_model.step()
                    optimizer_s_model.step()

            
            total_loss_epoch += total_loss.item()
            total_s_loss_epoch += loss_s_model.item()
            pbar_train.set_postfix({
                'loss_G': f"{total_loss.item():.2f}", 
                'loss_S': f"{loss_s_model.item():.2f}"
            })
            
        avg_train_loss = total_loss_epoch / len(train_loader)
        avg_s_train_loss = total_s_loss_epoch / len(train_loader)
        logger.info(f"Epoch {epoch} [Train] 完成, 平均生成损失: {avg_train_loss:.4f}, 平均排序损失: {avg_s_train_loss:.4f}")


        # --- 验证阶段 ---
        if epoch >= args.val_thre and (epoch % args.val_log_freq == 0):
            avg_val_loss, avg_s_val_loss = validate(val_loader, model, s_model, scheduler, args, amp_autocast, subgraph_generator_fn)
            logger.info(f"Epoch {epoch} [Validation] 完成, 平均损失: {avg_val_loss:.4f}")

            # 保存周期性检查点 
            logger.info(f"在 Epoch {epoch} 保存周期性检查点及其验证损失...")

            checkpoint_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                's_model_state_dict': s_model.state_dict(),
                'optimizer_model_state_dict': optimizer_model.state_dict(),
                'optimizer_s_model_state_dict': optimizer_s_model.state_dict(),
                'scheduler_model_state_dict': scheduler_model.state_dict(),    
                'scheduler_s_model_state_dict': scheduler_s_model.state_dict(), 
                'loss_scaler_state_dict': loss_scaler.state_dict(),
                'validation_loss': avg_val_loss, 
                's_validation_loss': avg_s_val_loss,
                'args': args
            }
            # 使用包含 epoch 编号的唯一文件名
            checkpoint_path = os.path.join(args.checkpoints_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint_state, checkpoint_path)

            # 检查并保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                logger.info(f"🎉 新的最佳验证损失: {best_val_loss:.4f}。保存最佳模型...")
                
                # 为最佳模型创建一个单独的保存状态
                best_model_state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    's_model_state_dict': s_model.state_dict(),
                    'best_val_loss': best_val_loss,
                    'args': args
                }
                
                best_model_path = os.path.join(args.checkpoints_dir, 'best_model.pth')
                torch.save(best_model_state, best_model_path)

            # 更新学习率调度器
            scheduler_model.step()
            scheduler_s_model.step()

    logger.info("训练完成。")
    logger.info(f"最终，最佳模型发现在 Epoch {best_epoch}，验证损失为: {best_val_loss:.4f}")
