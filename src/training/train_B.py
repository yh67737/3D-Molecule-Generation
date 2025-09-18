import torch
#torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR  # 导入 CosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts # 需要导入新的类
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import collections

# <--- 为SAM优化器添加实现 --->
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the sharpest point

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual descent

        if zero_grad: self.zero_grad()

    def step(self, closure=None):
        raise NotImplementedError("SAM doesn't work like vanilla optimizers, please use 'first_step' and 'second_step' explicitly.")

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm
# <--- [结束] --->

def log_timestep_distribution(stats_dict, epoch, strategy_name, t_bin_size, logger):
    """
    处理并记录一个 epoch 内时间步的采样分布和对应的平均损失。
    """
    if not stats_dict:
        logger.info(f"--- 在 Epoch {epoch} 未收集到策略 {strategy_name} 的时间步统计信息 ---")
        return

    logger.info(f"--- Epoch {epoch} | 策略 {strategy_name} | 时间步/损失分布 ---")
    
    total_samples = sum(data['count'] for data in stats_dict.values())
    if total_samples == 0:
        logger.info("  -> (无有效样本)")
        return

    # 按时间步的 bin 排序
    sorted_bins = sorted(stats_dict.keys())
    
    for t_bin_start in sorted_bins:
        data = stats_dict[t_bin_start]
        count = data['count']
        losses = data['losses']
        
        if count == 0:
            continue
            
        frequency = (count / total_samples) * 100
        avg_loss = sum(losses) / len(losses)
        
        t_bin_end = t_bin_start + t_bin_size - 1
        log_str = (
            f"  - t 范围 [{t_bin_start:04d}-{t_bin_end:04d}]: "
            f"采样频率={frequency:5.1f}% ({count:>{len(str(total_samples))}}/{total_samples}), "
            f"平均损失={avg_loss:.4f}"
        )
        logger.info(log_str)

def check_tensors(step_name, tensor_dict):
    """一个辅助函数，用于检查字典中的张量是否存在 NaN 或 Inf。"""
    for name, tensor in tensor_dict.items():
        if tensor is not None and torch.isinf(tensor).any():
            print(f"!!!!!!!!!!!!!! 在 {step_name} 处，张量 '{name}' 中检测到 Inf !!!!!!!!!!")
            # 在检测到问题的第一个地方就强制退出，以便检查
            raise RuntimeError(f"Inf detected in tensor '{name}' at step '{step_name}'")
        if tensor is not None and torch.isnan(tensor).any():
            print(f"!!!!!!!!!!!!!! 在 {step_name} 处，张量 '{name}' 中检测到 NaN !!!!!!!!!!")
            raise RuntimeError(f"NaN detected in tensor '{name}' at step '{step_name}'")

# ==============================================================================
# 1. 辅助函数 (Helper Functions)
# ==============================================================================

# def scale_to_unit_sphere(pos: torch.Tensor, batch_map: torch.Tensor) -> torch.Tensor:
#     """
#     将批次中每个图的坐标独立地缩放到单位球内。
    
#     Args:
#         pos (torch.Tensor): 批次中所有节点的坐标张量, shape [N, 3]。
#         batch_map (torch.Tensor): 将每个节点映射到其所属图的向量, shape [N]。
    
#     Returns:
#         torch.Tensor: 缩放后的坐标。
#     """
#     # 假设我们的输入 pos 是一个 [N, 3] 的张量，代表批次中所有 N 个原子的坐标；
#     # batch_map 是一个 [N] 的张量，告诉我们每个原子属于哪个分子图（0, 0, 0, 1, 1, ...）。
#     # PyG 的 scatter 函数可以高效地按组求和/求均值
#     from torch_geometric.utils import scatter
    
#     # 计算每个节点到其质心的距离
#     # 输入 pos 的维度是 [N, 3]
#     # e.g. 对于第 0 行 [x_0, y_0, z_0]，它计算 sqrt(x_0² + y_0² + z_0²)
#     # torch.linalg.norm 沿着 dim=1 进行操作，这个维度在计算后会消失
#     # 输出 distances 的维度是 [N]
#     distances = torch.linalg.norm(pos, dim=1)
    
#     # 按图分组，计算每个图中的最大距离
#     max_distances = scatter(distances, batch_map, dim=0, reduce='max') # shape: [num_graphs]
    
#     # 计算每个图的缩放因子，加上一个小的 epsilon 防止除以零
#     scale_factors = max_distances[batch_map].unsqueeze(1) + 1e-8

    
#     # 缩放坐标
#     return pos / scale_factors


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

    # --- [新] 调试检查 ---
    if torch.isnan(pred_logits).any() or torch.isinf(pred_logits).any():
        print("!!! 警告: 在计算损失之前，在 pred_logits 中检测到 NaN 或 Inf !!!")
        # 可选：保存有问题的张量，以便后续分析
        # torch.save(pred_logits, "problematic_logits.pt")
        
    # 同时检查目标索引是否在有效范围内
    num_classes = pred_logits.shape[-1]
    if true_x0_indices.max() >= num_classes or true_x0_indices.min() < 0:
        print(f"!!! 警告: 目标索引越界！最大索引: {true_x0_indices.max()}，类别总数: {num_classes}")
    # --- 调试检查结束 ---

    # 1. 计算标准的交叉熵损失 (对应于 L_vlb 的主要部分和 L_aux)
    # reduction='none' 表示我们为批次中的每个项目计算一个损失值
    loss = F.cross_entropy(pred_logits, true_x0_indices, reduction='none')

    # [新] 增加对损失输出本身的检查
    if torch.isnan(loss).any():
        print("!!! 警告: 经过 cross_entropy 计算后，损失值立刻变成了 NaN !!!")

    
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
# def calculate_coordinate_loss_wrapper(
#     predicted_r0: torch.Tensor,      # 模型预测的干净坐标 (t=0), shape [M, 3]
#     true_noise: torch.Tensor,        # 用于生成 r_t 的真实高斯噪声, shape [M, 3]
#     r_t: torch.Tensor,               # 输入到模型的加噪坐标 (t>0), shape [M, 3]
#     t: torch.Tensor,                 # 每个坐标对应的时间步, shape [M]
#     scheduler,                       # HierarchicalDiffusionScheduler 实例
#     schedule_type: str               # 使用的调度类型, 'alpha' 或 'delta'
# ) -> torch.Tensor:
#     """
#     计算原子坐标的损失。

#     这是一个包装函数，它接收模型预测的 r0，使用调度器将其转换为
#     预测的噪声 epsilon，然后计算与真实噪声的 L2 损失。

#     Args:
#         predicted_r0: 模型预测的干净坐标。
#         true_noise: 真实的噪声。
#         r_t: 加噪后的坐标。
#         t: 时间步。
#         scheduler: 噪声调度器实例。
#         schedule_type: 使用的调度类型 ('alpha' 或 'delta')。

#     Returns:
#         torch.Tensor: 计算出的标量损失值。
#     """
#     # 1. 检查输入是否为空。
#     # 在策略II中，如果目标原子被某种方式移除了（虽然不太可能），这可以防止出错。
#     if predicted_r0.shape[0] == 0:
#         return torch.tensor(0.0, device=predicted_r0.device)

#     # 2. 调用调度器的核心方法，从 predicted_r0 反推出 predicted_noise
#     predicted_noise = scheduler.get_predicted_noise_from_r0(
#         r_t=r_t,
#         t=t,
#         predicted_r0=predicted_r0,
#         schedule_type=schedule_type
#     )

#     # 3. 计算预测噪声和真实噪声之间的 L2 损失 (均方误差, Mean Squared Error)
#     # F.mse_loss(A, B) 会计算 (A - B)^2 的所有元素的平均值。
#     loss = F.mse_loss(predicted_noise, true_noise)

def calculate_coordinate_loss_wrapper(
    predicted_x0: torch.Tensor,      # 模型预测的干净坐标 (t=0), shape [M, 3]
    true_x0: torch.Tensor,           # 真实的干净坐标 (t=0), shape [M, 3]
    r_t: torch.Tensor,               # 输入到模型的加噪坐标 (t>0), shape [M, 3]
    t: torch.Tensor,                 # 每个坐标对应的时间步, shape [M]
    scheduler,                       # HierarchicalDiffusionScheduler 实例
    schedule_type: str               # 使用的调度类型, 'alpha' 或 'delta'
) -> torch.Tensor:
    """
    计算原子坐标的损失。

    这是一个包装函数，它接收模型预测的噪声，然后计算与真实噪声的 L2 损失。

    Args:
        predicted_x0: 模型预测的干净坐标 x0。
        true_x0: 真实的干净坐标 x0。

    Returns:
        torch.Tensor: 计算出的标量损失值。
    """
    # 1. 检查输入是否为空。
    if predicted_x0.shape[0] == 0:
        return torch.tensor(0.0, device=predicted_x0.device)

    # 2. 计算预测的 x0 和真实的 x0 之间的 L2 损失 (均方误差, Mean Squared Error)
    # <---  将 MSE Loss 更换为 Huber Loss --->
    # 原始代码: loss = F.mse_loss(predicted_x0, true_x0)
    # Huber Loss 对异常值更鲁棒。delta=1.0是常用默认值。
    loss = F.huber_loss(predicted_x0, true_x0, delta=1.0)
    # <--- [结束] --->

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
    # 1. 检查输入是否为空。如果一个批次中没有需要预测的边，则损失为0。
    # 这在策略II中，如果目标原子是孤立点时可能发生。
    if pred_logits.shape[0] == 0:
        return torch.tensor(0.0, device=pred_logits.device)
        
    # 2. 计算标准的交叉熵损失
    # reduction='none' 表示我们为批次中的每条边计算一个损失值
    loss = F.cross_entropy(pred_logits, true_b0_indices, reduction='none')
    
    # 3. 应用 D3PM 混合损失的简化权重
    # 总权重 = 1 (来自 L_vlb) + lambda_aux (来自 L_aux)
    final_loss = (1 + lambda_aux) * loss
    
    # 4. 对批次中的所有边的损失求平均，得到最终的标量损失
    return final_loss.mean()


# ==============================================================================
# 3. 验证函数 (Validation Function) - [修改后版本]
# ==============================================================================
@torch.no_grad() # 装饰器，表示该函数内所有 torch 计算都不需要记录梯度
def validate(val_loader, model, scheduler, args, amp_autocast):
    """
    在验证集上评估模型损失。
    此函数的前向传播和损失计算逻辑与训练过程完全一致。
    """
    device = args.device
    model.eval() # 将模型设置为评估模式
    total_val_loss = 0.0
    pbar_val = tqdm(val_loader, desc=f"Validating", leave=False)

    for clean_batch in pbar_val:
        clean_batch = clean_batch.to(device)

        # with amp_autocast():
        # --- [逻辑与训练循环完全相同] ---

        # --- 0. 准备工作 ---
        num_graphs, num_nodes, num_edges = clean_batch.num_graphs, clean_batch.num_nodes, clean_batch.num_edges
        # scaled_pos = scale_to_unit_sphere(clean_batch.pos, clean_batch.batch)
        scaled_pos = clean_batch.pos # 不进行坐标缩放
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
        
        lossI_a = calculate_atom_type_loss(predictions_I['atom_type_logits'], clean_batch.x.argmax(dim=-1), args.lambda_aux)
        lossI_r = calculate_coordinate_loss_wrapper(predictions_I['predicted_r0'], scaled_pos, noised_pos_I, t1_per_node, scheduler, 'alpha')
        lossI_b = calculate_bond_type_loss(predictions_I['bond_logits'], clean_batch.edge_attr.argmax(dim=-1), args.lambda_aux)
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

        lossII_a = calculate_atom_type_loss(predictions_II['atom_type_logits'], clean_batch.x[target_node_mask_II].argmax(dim=-1), args.lambda_aux)
        lossII_r = calculate_coordinate_loss_wrapper(predictions_II['predicted_r0'], scaled_pos[target_node_mask_II], noised_pos_target, t2_per_node[target_node_mask_II], scheduler, 'delta')
        lossII_b = calculate_bond_type_loss(predictions_II['bond_logits'], clean_batch.edge_attr[target_edge_mask].argmax(dim=-1), args.lambda_aux)
        loss_II = args.w_a * lossII_a + args.w_r * lossII_r + args.w_b * lossII_b

        # --- 总验证损失 ---
        # total_loss = scheduler.T1 * loss_I + scheduler.T2 * loss_II 
        # 计算分母
        denominator = scheduler.T1 + scheduler.T2
        # 计算加权损失
        total_loss = (scheduler.T1 / denominator) * loss_I + (scheduler.T2 / denominator) * loss_II
        
        total_val_loss += total_loss.item()
        pbar_val.set_postfix({
            'val_loss': total_loss.item(),
            'val_loss_I': loss_I.item(),
            'val_loss_II': loss_II.item()
        })
    
    avg_val_loss = total_val_loss / len(val_loader)
    return avg_val_loss

# ==============================================================================
# 4. 主训练函数 (Main Training Function)
# ==============================================================================

def train(
    args, # 包含超参数的对象，如学习率、epoch数等
    logger,
    train_loader, # 训练数据
    val_loader, # 测试数据
    model: nn.Module, # E_DiT_Network 实例
    scheduler, # HierarchicalDiffusionScheduler 实例
    amp_autocast,
    loss_scaler,
    train_sampler
):
    """
    主训练函数。
    """
    device = args.device
    model.to(device)

    # <--- 优化器创建逻辑 --->
    weight_decay = getattr(args, 'weight_decay', 1e-4)
    if args.optimizer.lower() == 'adamw':
        optimizer = AdamW(
            model.parameters(), 
            lr=args.learning_rate,
            weight_decay=weight_decay
        )
        logger.info(f"优化器已设置为 AdamW，学习率: {args.learning_rate}, 权重衰减: {weight_decay}")
    elif args.optimizer.lower() == 'sam':
        # SAM 需要一个基础优化器
        base_optimizer = AdamW
        optimizer = SAM(
            model.parameters(), 
            base_optimizer, 
            rho=0.05, # SAM的邻域大小参数，可以调整
            adaptive=False,
            lr=args.learning_rate,
            weight_decay=weight_decay
        )
        logger.info(f"优化器已设置为 SAM (AdamW base)，学习率: {args.learning_rate}, 权重衰减: {weight_decay}")
    else:
        raise ValueError(f"不支持的优化器: {args.optimizer}")
    # <--- [结束] --->

    T_max = args.epochs  # 最大迭代次数，通常设置为总 epoch 数
    lr_min_factor = args.lr_min_factor
    # lr_scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=lr_min_factor * args.learning_rate)
    # 方案 B: 周期逐渐变长的重启 (更常见)
    lr_scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=50,             # 第一个周期是 50 个 epoch
        T_mult=2,           # 下一个周期是上一个的 2 倍长 (50, 100, 200...)
        eta_min=lr_min_factor * args.learning_rate
    )
    # 使用 ReduceLROnPlateau 调度器
    # lr_factor=0.5
    # lr_patience=3

    # lr_scheduler = ReduceLROnPlateau(
    #     optimizer,
    #     mode='min',          # 监控的指标(lossII_r)需要减小
    #     factor=lr_factor,          # 当指标不再改善时，学习率乘以 0.5
    #     patience=lr_patience,          # 容忍 3 个 epoch 指标不改善
    #     verbose=True,        # 当学习率更新时在控制台打印消息
    #     threshold=0.001,     # 只有当新旧指标差异大于此阈值时才认为有改善
    #     min_lr=args.lr_min_factor * args.learning_rate # 学习率的下限
    # )
    # logger.info(f"学习率调度器已设置为 ReduceLROnPlateau，监控指标: 'avg_lossII_r', 耐心值: {lr_patience}, 衰减因子: {lr_factor}")

    start_epoch = 1
    best_val_loss = float('inf')
    best_epoch = 0
    
    # 计算 Warmup 的总步数
    # len(train_loader) 是每个 epoch 的迭代（step）次数
    num_train_steps_per_epoch = len(train_loader)
    total_warmup_steps = args.warmup_epochs_for_EDiT * num_train_steps_per_epoch
    base_lr = args.learning_rate # 保存基础学习率
    initial_warmup_lr = base_lr * args.warmup_factor
    
    logger.info(f"学习率预热已启用，将持续 {args.warmup_epochs_for_EDiT} 个 epochs ({total_warmup_steps} 步)。")
    logger.info(f"学习率将从 {initial_warmup_lr:.2e} 线性增长到 {base_lr:.2e}。")

    if args.resume_ckpt and os.path.isfile(args.resume_ckpt):
        logger.info(f"正在从检查点恢复训练: {args.resume_ckpt}")
        
        # 加载 checkpoint 文件到 CPU，以避免 GPU 内存冲突
        checkpoint = torch.load(args.resume_ckpt, map_location='cpu', weights_only=False)

        # 1. 恢复模型权重
        #    处理分布式（DDP）和非分布式模型保存的 state_dict 差异
        model_state_dict = checkpoint['model_state_dict']
        if args.distributed:
            # DDP 模型保存的键会带有 'module.' 前缀，我们需要确保加载时匹配
            # 如果当前是 DDP 模型，而保存的不是，则需要添加前缀
            # 通常更简单的方法是直接加载到 model.module
            model.module.load_state_dict(model_state_dict)
        else:
            # 如果当前不是 DDP，但保存的是 DDP 模型权重，需要移除 'module.' 前缀
            # from collections import OrderedDict
            # new_state_dict = OrderedDict()
            # for k, v in model_state_dict.items():
            #     if k.startswith('module.'):
            #         name = k[7:] # remove `module.`
            #         new_state_dict[name] = v
            #     else:
            #         new_state_dict[k] = v
            # model.load_state_dict(new_state_dict)
            model.load_state_dict(model_state_dict)

        # 2. 恢复优化器状态
        if 'optimizer_model_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_model_state_dict'])
            logger.info("优化器状态已恢复。")

        # 3. 恢复学习率调度器状态
        if 'scheduler_model_state_dict' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['scheduler_model_state_dict'])
            logger.info("学习率调度器状态已恢复。")

        # 4. 恢复训练周期和最佳验证损失
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            logger.info(f"将从 Epoch {start_epoch} 开始训练。")
        
        # 从 best_model.pth 恢复时用这个
        if 'best_val_loss' in checkpoint:
             best_val_loss = checkpoint['best_val_loss']
             logger.info(f"已恢复之前的最佳验证损失: {best_val_loss:.4f}")
        # 从 checkpoint_epoch_xx.pth 恢复时用这个
        elif 'validation_loss' in checkpoint:
            best_val_loss = checkpoint.get('best_val_loss', checkpoint['validation_loss']) # 兼容两种保存方式
            logger.info(f"已恢复之前的最佳验证损失: {best_val_loss:.4f}")

        # 5. 恢复 AMP loss scaler 的状态 (如果使用)
        if loss_scaler is not None and 'loss_scaler_state_dict' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['loss_scaler_state_dict'])
            logger.info("AMP loss scaler 状态已恢复。")
            
        logger.info("检查点加载完成。")
    
    else:
        if args.resume_ckpt:
            logger.warning(f"指定的检查点文件未找到: {args.resume_ckpt}。将从头开始训练。")
        else:
            logger.info("未指定检查点，将从头开始训练。")
        # 这些变量的初始化移到这里，确保逻辑清晰
        best_val_loss = float('inf')
        best_epoch = 0
    
    logger.info(f"模型检查点将保存在: {args.checkpoints_dir}")
    logger.info("开始训练...")

    accumulation_steps = args.accumulation_steps

    # ==================== 修改处: 调整 bin 大小 ====================
    # 您可以根据 T1 和 T2 的总步长调整这些值以获得合适的粒度
    # 例如，T1=100, bin=10 会产生10个条目。T2=900, bin=100 会产生9个条目。
    t_bin_size_I = 10
    t_bin_size_II = 100
    # ==========================================================

    for epoch in range(start_epoch, args.epochs + 1):
        # 将模型设置为“训练模式”
        # 它会通知模型中所有具有不同训练/评估行为的层（主要是 Dropout 层和 BatchNorm 层）切换到它们的训练状态。
        # Dropout 层在训练时会随机“丢弃”一些神经元，以防止过拟合；在评估时则不会丢弃，会使用所有神经元。
        # BatchNorm 层在训练时会使用当前批次的均值和方差进行归一化，并更新其内部的全局统计量；在评估时则会使用已学习到的全局统计量。
        model.train()  

        if args.distributed and hasattr(train_sampler, 'set_epoch'):
            train_sampler.set_epoch(epoch)

        # 初始化一个变量，用于累加当前这个 epoch 内所有批次的损失值
        # 在每个 epoch 结束时，我们可以用 total_loss_epoch 除以批次的总数，来计算并打印出这个 epoch 的平均损失，以此来监控训练的进展。
        total_loss_epoch = 0.0
        total_loss_I_epoch = 0.0
        total_loss_II_epoch = 0.0
        total_lossI_a_epoch = 0.0
        total_lossI_r_epoch = 0.0
        total_lossI_b_epoch = 0.0
        total_lossII_a_epoch = 0.0
        total_lossII_r_epoch = 0.0
        total_lossII_b_epoch = 0.0

        # ==================== 初始化 epoch 统计容器 ====================
        # 使用 defaultdict 可以让代码更简洁
        t1_stats = collections.defaultdict(lambda: {'count': 0, 'losses': []})
        t2_stats = collections.defaultdict(lambda: {'count': 0, 'losses': []})
        # =================================================================

        # 使用 tqdm 创建进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")  # desc=...: 设置进度条左侧的描述性文字，例如 Epoch 1/100

        # ==================== 初始化用于显示的梯度范数变量 ====================
        grad_norm_to_display = 0.0
        # =======================================================================
       
        optimizer.zero_grad()
        for i, clean_batch in enumerate(pbar):
            # 在每个训练步中，手动调整学习率
            # (epoch 从1开始, i 从0开始)
            current_step = (epoch - 1) * num_train_steps_per_epoch + i
            if current_step < total_warmup_steps:
                # 计算当前预热阶段的学习率乘子
                # +1 是为了确保 lr_scale 从一个很小的值开始而不是0
                lr_scale = (current_step + 1) / total_warmup_steps
                # 线性插值
                current_lr = initial_warmup_lr + lr_scale * (base_lr - initial_warmup_lr)
                
                # 直接将计算出的学习率应用到优化器
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr

            clean_batch = clean_batch.to(device)

            is_sync_step = ((i + 1) % accumulation_steps == 0) or (i + 1 == len(train_loader))

            from contextlib import suppress
            context = model.no_sync() if (args.distributed and not is_sync_step) else suppress()

            # --- 定义一个计算损失的闭包函数，方便SAM调用 ---
            def compute_loss():
                with context:
                    # --- 0. 准备工作 ---
                    num_graphs = clean_batch.num_graphs # 批次中包含的独立图的数量（等于 batch_size）。用于采样图级别的变量，如时间步 t
                    num_nodes = clean_batch.num_nodes #  批次中所有图的节点总数
                    num_edges = clean_batch.num_edges # 批次中所有图的边总数
    
                    # a. 坐标缩放
                    # scaled_pos = scale_to_unit_sphere(clean_batch.pos, clean_batch.batch)
                    scaled_pos = clean_batch.pos # 不进行坐标缩放
            
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
                    noised_pos_I = scheduler.q_sample(scaled_pos, t1_per_node, noise1, schedule_type='alpha')  # 阅读标记
            
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

                    pos_noise_I = torch.randn_like(scaled_pos) * args.pos_noise_std

                    lossI_r = calculate_coordinate_loss_wrapper(
                        predicted_x0=predictions_I['predicted_r0'], 
                        true_x0=scaled_pos + pos_noise_I, 
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

                    pos_noise_II = torch.randn_like(scaled_pos[target_node_mask_II]) * args.pos_noise_std

                    lossII_r = calculate_coordinate_loss_wrapper(
                        predicted_x0=predictions_II['predicted_r0'], 
                        true_x0=scaled_pos[target_node_mask_II] + pos_noise_II, 
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
                    # total_loss = scheduler.T1 * loss_I + scheduler.T2 * loss_II
                    # 计算分母
                    denominator = scheduler.T1 + scheduler.T2
                    # 计算加权损失
                    total_loss = (scheduler.T1 / denominator) * loss_I + (scheduler.T2 / denominator) * loss_II

                    # total_loss = total_loss / accumulation_steps
                    # total_loss.backward()

                # 返回所有需要的损失项用于记录
                return total_loss, loss_I, loss_II, lossI_a, lossI_r, lossI_b, lossII_a, lossII_r, lossII_b, t1, t2, predictions_I, predictions_II, scaled_pos, target_node_mask_II

            # --- 步骤 1: 计算第一步的损失和梯度 ---
            # AdamW和SAM都需要执行这一步
            total_loss, loss_I, loss_II, lossI_a, lossI_r, lossI_b, lossII_a, lossII_r, lossII_b, t1, t2, predictions_I, predictions_II, scaled_pos, target_node_mask_II = compute_loss()
            (total_loss / accumulation_steps).backward()
            
            # --- 步骤 2: 如果是SAM并且是同步步骤，则执行第二步 ---
            if args.optimizer.lower() == 'sam' and is_sync_step:
                optimizer.first_step(zero_grad=True) # 上升并清零梯度

                # 在扰动后的位置，重新计算损失并进行反向传播
                total_loss_sam, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = compute_loss()
                (total_loss_sam / accumulation_steps).backward() # 计算第二步的梯度

            if is_sync_step:
                total_grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=50.0)
                grad_norm_to_display = total_grad_norm_before_clip.item()
                
                # --- 步骤 3: 执行优化器更新 ---
                if args.optimizer.lower() == 'sam':
                    optimizer.second_step(zero_grad=True) # 使用第二步的梯度更新并清零
                else: # AdamW
                    optimizer.step()
                    optimizer.zero_grad()

            # ==================== 收集当前批次的统计数据 ====================
            # 策略 I
            lossI_r_item = lossI_r.item()
            for t_val in t1.cpu().numpy(): # t1 是 per-graph 的
                t_bin = (t_val // t_bin_size_I) * t_bin_size_I
                t1_stats[t_bin]['count'] += 1
                t1_stats[t_bin]['losses'].append(lossI_r_item)

            # 策略 II
            lossII_r_item = lossII_r.item()
            for t_val in t2.cpu().numpy(): # t2 也是 per-graph 的
                t_bin = (t_val // t_bin_size_II) * t_bin_size_II
                t2_stats[t_bin]['count'] += 1
                t2_stats[t_bin]['losses'].append(lossII_r_item)
            # =====================================================================

            with torch.no_grad(): # 确保这部分不计算梯度
                # --- 监控策略 I (全局微调) ---
                # 1. 获取模型预测的x0
                predicted_x0_I = predictions_I['predicted_r0']
                    
                # # 2. 根据预测的噪声，反推出模型预测的干净坐标 x_hat_0
                # alpha_bar_t_I = scheduler.alpha_bars[t1_per_node]
                # sqrt_alpha_bar_t_I = torch.sqrt(alpha_bar_t_I).unsqueeze(1)
                # sqrt_one_minus_alpha_bar_t_I = torch.sqrt(1.0 - alpha_bar_t_I).unsqueeze(1)
                # predicted_x0_I = (noised_pos_I - sqrt_one_minus_alpha_bar_t_I * predicted_noise_I) / sqrt_alpha_bar_t_I
                    
                # 3. 计算并记录关键指标的模长 (Norm)
                # 我们关心的是平均范数，而不是总范数
                # norm_true_noise_I = torch.linalg.norm(noise1, dim=-1).mean()
                # norm_predicted_noise_I = torch.linalg.norm(predicted_noise_I, dim=-1).mean()
                norm_predicted_x0_I = torch.linalg.norm(predicted_x0_I, dim=-1).mean().item()
                norm_true_x0_I = torch.linalg.norm(scaled_pos, dim=-1).mean().item() # scaled_pos 是真实的干净坐标

                # --- 监控策略 II (局部生成) ---
                predicted_x0_II = predictions_II['predicted_r0']
                    
                # alpha_bar_t_II = scheduler.delta_bars[t2_per_node[target_node_mask_II]]
                # sqrt_alpha_bar_t_II = torch.sqrt(alpha_bar_t_II).unsqueeze(1)
                # sqrt_one_minus_alpha_bar_t_II = torch.sqrt(1.0 - alpha_bar_t_II).unsqueeze(1)
                # predicted_x0_II = (noised_pos_target - sqrt_one_minus_alpha_bar_t_II * predicted_noise_II) / sqrt_alpha_bar_t_II

                # norm_true_noise_II = torch.linalg.norm(noise2[target_node_mask_II], dim=-1).mean()
                # norm_predicted_noise_II = torch.linalg.norm(predicted_noise_II, dim=-1).mean()
                norm_predicted_x0_II = torch.linalg.norm(predicted_x0_II, dim=-1).mean().item()
                norm_true_x0_II = torch.linalg.norm(scaled_pos[target_node_mask_II], dim=-1).mean().item()

                # --- 监控 `first_pred_r0` (这个逻辑也放在这里) ---
                if predicted_x0_II.shape[0] > 0:
                    first_pred_r0_sample = predicted_x0_II[0].detach().cpu().numpy()
                    pred_r0_str = f"[{first_pred_r0_sample[0]:>7.3f}, {first_pred_r0_sample[1]:>7.3f}, {first_pred_r0_sample[2]:>7.3f}]"
                else:
                    pred_r0_str = "N/A"
                
            total_loss_epoch += total_loss.item()
            total_loss_I_epoch += loss_I.item()
            total_loss_II_epoch += loss_II.item()
            total_lossI_a_epoch += lossI_a.item()
            total_lossI_r_epoch += lossI_r.item()
            total_lossI_b_epoch += lossI_b.item()
            total_lossII_a_epoch += lossII_a.item()
            total_lossII_r_epoch += lossII_r.item()
            total_lossII_b_epoch += lossII_b.item()
            pbar.set_postfix({
                # 'loss': total_loss.item() * accumulation_steps, # 将当前批次的损失乘以 accumulation_steps，得到可比较的真实损失
                'loss_I': loss_I.item(), # 将 loss_I 的数值添加到进度条
                'loss_II': loss_II.item(), # 将 loss_II 的数值添加到进度条
                # 'lossI_a': lossI_a.item(), # 原子类型损失 Ⅰ
                'lossI_r': lossI_r.item(), # 坐标损失 Ⅰ
                # 'lossI_b': lossI_b.item(), # 边类型损失 Ⅰ
                # 'lossII_a': lossII_a.item(), # 原子类型损失 Ⅱ
                'lossII_r': lossII_r.item(), # 坐标损失 Ⅱ
                # 'lossII_b': lossII_b.item(),  # 边类型损失 Ⅱ
                # 'p_noise_I_norm': norm_predicted_noise_I.item(),
                'grad_norm': grad_norm_to_display, # 显示当前的梯度范数
                # ==================== 新增的监控项 ====================
                'p_x0_I_norm': norm_predicted_x0_I,    # 策略 I 预测 x0 的范数
                't_x0_I_norm': norm_true_x0_I,      # 策略 I 真实 x0 的范数
                'p_x0_II_norm': norm_predicted_x0_II,   # 策略 II 预测 x0 的范数
                't_x0_II_norm': norm_true_x0_II,     # 策略 II 真实 x0 的范数
                'first_pred_r0': pred_r0_str       # 策略 II 第一个预测 x0 的值
                # =======================================================
            })
            
        avg_real_train_loss = total_loss_epoch / len(train_loader)
        logger.info(f"Epoch {epoch} [Train] 完成, 平均损失: {avg_real_train_loss:.4f}")
        num_batches = len(train_loader)
        avg_loss_I = total_loss_I_epoch / num_batches
        avg_loss_II = total_loss_II_epoch / num_batches
        avg_lossI_a = total_lossI_a_epoch / num_batches
        avg_lossI_r = total_lossI_r_epoch / num_batches
        avg_lossI_b = total_lossI_b_epoch / num_batches
        avg_lossII_a = total_lossII_a_epoch / num_batches
        avg_lossII_r = total_lossII_r_epoch / num_batches
        avg_lossII_b = total_lossII_b_epoch / num_batches

        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        # 构建日志字符串
        log_str = (
            f"  -> Loss Details: loss={avg_real_train_loss:.2e}, " # 使用科学计数法
            f"loss_I={avg_loss_I:.2f}, loss_II={avg_loss_II:.2f}, "
            f"lossI_a={avg_lossI_a:.2f}, lossI_r={avg_lossI_r:.2f}, lossI_b={avg_lossI_b:.2f}, "
            f"lossII_a={avg_lossII_a:.2f}, lossII_r={avg_lossII_r:.2f}, lossII_b={avg_lossII_b:.2f},"
            f"lr={current_lr:.2e}"  # 在日志中添加学习率
        )
        logger.info(log_str)

        # ==================== 在 epoch 结束时记录分布 ====================
        # log_timestep_distribution(t1_stats, epoch, "I (全局微调 - 坐标损失)", t_bin_size_I, logger)
        # log_timestep_distribution(t2_stats, epoch, "II (局部生成 - 坐标损失)", t_bin_size_II, logger)
        # ====================================================================

        # --- 验证阶段 ---
        if epoch >= args.val_thre and (epoch % args.val_log_freq == 0):
            avg_val_loss = validate(val_loader, model, scheduler, args, amp_autocast)
            logger.info(f"Epoch {epoch} [Validation] 完成, 平均损失: {avg_val_loss:.4f}")

            # 保存周期性检查点 
            logger.info(f"在 Epoch {epoch} 保存周期性检查点及其验证损失...")
            if args.distributed:
                model_state_to_save = model.module.state_dict()
            else:
                model_state_to_save = model.state_dict()

            checkpoint_state = {
                'epoch': epoch,
                'model_state_dict': model_state_to_save,
                'optimizer_model_state_dict': optimizer.state_dict(),
                'scheduler_model_state_dict': lr_scheduler.state_dict(),
                'validation_loss': avg_val_loss, # <-- 明确保存当前 epoch 的验证损失
                'args': args
            }
            if loss_scaler is not None:
                checkpoint_state['loss_scaler_state_dict'] = loss_scaler.state_dict()
            # 使用包含 epoch 编号的唯一文件名
            checkpoint_path = os.path.join(args.checkpoints_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint_state, checkpoint_path)

            # 检查并保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                logger.info(f"🎉 新的最佳验证损失: {best_val_loss:.4f}。保存最佳模型...")
                
                if args.distributed:
                    model_state_to_save = model.module.state_dict()
                else:
                    model_state_to_save = model.state_dict()

                # 为最佳模型创建一个单独的保存状态
                best_model_state = {
                    'epoch': epoch,
                    'model_state_dict': model_state_to_save,
                    'optimizer_model_state_dict': optimizer.state_dict(),
                    'scheduler_model_state_dict': lr_scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'args': args
                }
                
                best_model_path = os.path.join(args.checkpoints_dir, 'best_model.pth')
                torch.save(best_model_state, best_model_path)
        # 更新学习率调度器
        # lr_scheduler.step(avg_lossII_r)
        if epoch >= args.warmup_epochs_for_EDiT:
            lr_scheduler.step()
            
    logger.info("训练完成。")
    logger.info(f"最终，最佳模型发现在 Epoch {best_epoch}，验证损失为: {best_val_loss:.4f}")