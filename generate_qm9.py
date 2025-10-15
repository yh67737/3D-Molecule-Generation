import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import torch.nn as nn
from tqdm import trange
from src.training.scheduler import HierarchicalDiffusionScheduler
import numpy as np

# ==============================================================================
# 辅助函数
# ==============================================================================

# 新增辅助函数：对称化
# ==============================================================================
def symmetrize_bond_logits(edge_index, bond_logits):
    """
    通过平均相反方向边的 logits 来强制对称性。
    
    Args:
        edge_index (Tensor): [2, num_edges]
        bond_logits (Tensor): [num_edges, num_bond_types]
        
    Returns:
        symmetrized_logits (Tensor): [num_edges, num_bond_types]
    """
    num_edges = edge_index.shape[1]
    symmetrized_logits = torch.zeros_like(bond_logits)
    processed_edges = torch.zeros(num_edges, dtype=torch.bool, device=edge_index.device)

    for i in range(num_edges):
        if processed_edges[i]:
            continue

        u, v = edge_index[0, i], edge_index[1, i]

        # 寻找反向边 (v, u)
        # 注意：这在稠密图上效率不高，但在小分子上是可行的
        mask_rev = (edge_index[0] == v) & (edge_index[1] == u)
        
        # 应该只找到一条反向边
        if mask_rev.sum() == 1:
            j = torch.where(mask_rev)[0].item()
            
            # 平均 logits
            avg_logits = (bond_logits[i] + bond_logits[j]) / 2.0
            
            symmetrized_logits[i] = avg_logits
            symmetrized_logits[j] = avg_logits
            
            processed_edges[i] = True
            processed_edges[j] = True
        else:
            # 如果没有反向边（例如自环），则直接使用原始 logits
            symmetrized_logits[i] = bond_logits[i]
            processed_edges[i] = True
            
    return symmetrized_logits

def sample_symmetric_bonds(edge_index, x_t_bonds, pred_logits_b, scheduler, t_current_val, t_prev_val, schedule_type):
    """
    对给定的边子集进行对称采样。
    
    Args:
        edge_index (Tensor): 边的索引 [2, num_sub_edges]。
        x_t_bonds (Tensor): 当前时间步的边属性 [num_sub_edges, num_bond_types]。
        pred_logits_b (Tensor): 模型预测的边 logits [num_sub_edges, num_bond_types]。
        scheduler: 扩散调度器。
        t_gen (int): 当前去噪时间步。
        schedule_type (str): 'alpha' 或 'delta'。
        
    Returns:
        Tensor: 对称采样后的新边属性 [num_sub_edges, num_bond_types]。
    """
    # 1. 对称化 logits，确保概率分布相同
    symmetrized_logits = symmetrize_bond_logits(edge_index, pred_logits_b)
    
    # 2. 联动采样
    new_edge_attr_one_hot = torch.zeros_like(x_t_bonds)
    processed_edges = torch.zeros(edge_index.shape[1], dtype=torch.bool, device=edge_index.device)

    for i in range(edge_index.shape[1]):
        if processed_edges[i]:
            continue

        u, v = edge_index[0, i], edge_index[1, i]
        
        # 在给定的 edge_index 中寻找反向边
        mask_rev = (edge_index[0] == v) & (edge_index[1] == u)
        
        if mask_rev.sum() == 1:
            j = torch.where(mask_rev)[0].item()
            
            # 对边对只采样一次
            sampled_bond = scheduler.compute_discrete_jump_step(
                x_t=x_t_bonds[i].unsqueeze(0),
                pred_x0_logits=symmetrized_logits[i].unsqueeze(0),
                t_current=t_current_val,
                t_previous=t_prev_val,
                schedule_type=schedule_type,
                is_atom=False
            )
            
            # 将同一样本应用到两个方向
            new_edge_attr_one_hot[i] = sampled_bond
            new_edge_attr_one_hot[j] = sampled_bond
            
            processed_edges[i] = True
            processed_edges[j] = True
        else:
            # 处理无反向边的情况 (例如自环)
            sampled_bond = scheduler.compute_discrete_jump_step(
                x_t=x_t_bonds[i].unsqueeze(0),
                pred_x_logits=symmetrized_logits[i].unsqueeze(0),
                t_current=t_current_val,
                t_previous=t_prev_val,
                schedule_type=schedule_type,
                is_atom=False
            )
            new_edge_attr_one_hot[i] = sampled_bond
            processed_edges[i] = True
            
    return new_edge_attr_one_hot


def get_ring_guidance(p_model, fragment: Data, threshold=0.5) -> Data:
    """
    直接在输入的图上运行环结构预测（要求图已经包含 x, edge_index, edge_attr）。

    Args:
        graph (Data): PyG 图对象，需包含 x、edge_index、edge_attr。
        model (nn.Module): 已训练好的 RingPredictor 模型。
        threshold (float): 判定为“在环上”的概率阈值，默认 0.5。

    Returns:
        probs (Tensor): 每个节点属于环的概率，[num_nodes]
        pred_mask (Tensor[bool]): 每个节点是否被预测为“在环上”
    """
    device = next(p_model.parameters()).device
    fragment = fragment.to(device)

    p_model.eval()
    with torch.no_grad():
        logits = p_model(fragment)
        probs = torch.sigmoid(logits).squeeze(-1)
        pred_mask = probs > threshold

    fragment.pring_out = pred_mask.long().unsqueeze(1)  
    return fragment  

def check_connectivity(new_atom_idx: int, fragment: Data) -> bool:
    """检查新原子是否与现有片段有连接。"""
    # 这是一个简化检查，只看是否有非“无键”的边
    if fragment.num_edges == 0:
        return False
        
    edge_index = fragment.edge_index
    edge_attr = fragment.edge_attr
    
    # 无键类别是最后一个类别
    no_bond_idx = edge_attr.shape[-1] - 1
    
    # 找到连接到新原子的边
    connected_mask = (edge_index[0] == new_atom_idx) | (edge_index[1] == new_atom_idx)
    
    if not connected_mask.any():
        return False
        
    # 检查这些边是否都不是“无键”
    connected_bonds = edge_attr[connected_mask]
    is_connected = (connected_bonds.argmax(dim=-1) != no_bond_idx).any()
    
    return is_connected.item()


# ==============================================================================
# 主生成函数
# ==============================================================================
@torch.no_grad()
def generate_molecule(
    model: nn.Module,
    p_model: nn.Module,
    scheduler: HierarchicalDiffusionScheduler,
    args
):
    """
    自回归地、使用双阶段扩散模型生成一个完整的分子。
    """
    device = args.device
    model.eval() 
    p_model.eval() # 评估模式 (Evaluation Mode)

    ATOM_MAP = ['H', 'C', 'N', 'O', 'F', 'Absorb']
    BOND_MAP = ['Single', 'Double', 'Triple', 'No Bond']

    # --- 1. 从一个原子开始 ---
    print("步骤 1: 随机采样第一个原子")
    # a. 随机原子类型 (H,C,N,O,F)
    # atom_type_idx_tensor = torch.randint(0, 4, (1,), device=device)
    atom_type_idx_tensor = torch.tensor([0], device=device, dtype=torch.long)
    atom_type_idx = atom_type_idx_tensor.item() # 获取 python int
    atom_symbol = ATOM_MAP[atom_type_idx]      # 从映射中查找符号
    atom_type = F.one_hot(atom_type_idx_tensor, num_classes=6).float() # 6类，最后一类是吸收态
    
    # b. 设置坐标为原点
    pos = torch.zeros(1, 3, device=device)
    
    # c. 创建 PyG Data 对象
    fragment = Data(x=atom_type, pos=pos)
    fragment.is_new_node = torch.tensor([[False]], device=device) # is_new_node设置为False
    
    # 确认子图只有一个原子
    assert fragment.num_nodes == 1, "子图节点数不为1"

    # 为单原子子图添加 pring_out 属性，形状为 [1, 1]，值为0
    fragment.pring_out = torch.zeros(
        (1, 1),  # 固定形状 [1, 1]（单个节点，单个输出维度）
        dtype=torch.float,  # 与环指导信息的标准类型一致（0/1的float类型）
        device=fragment.x.device if hasattr(fragment, 'x') else torch.device('cpu')
    )

    print(f"  -> 第一个原子类型: {atom_symbol} (索引: {atom_type_idx})")
    print(f"  -> 初始片段信息: {fragment}")
    print(f"  -> [坐标] 初始坐标: Shape={fragment.pos.shape}\n{fragment.pos}")

    # --- 2. 自回归生成循环 ---
    for num_existing_atoms in range(1, args.max_atoms):
        print(f"\n--- [主循环] 当前原子数: {num_existing_atoms}, 开始生成第 {num_existing_atoms + 1} 个原子 ---")
        
        # a. 添加带噪的新原子
        print("步骤 2: 添加带噪的新原子")
        # i. 新原子类型为吸收态
        absorbing_state_idx = 5
        new_atom_type_idx = torch.tensor([absorbing_state_idx], device=device)
        new_atom_type = F.one_hot(new_atom_type_idx, num_classes=6).float()
        
        # ii. 随机新原子坐标
        new_pos = torch.randn(1, 3, device=device)
        
        # iii. 随机新原子环信息
        new_pring_out = (torch.rand(1, 1, device=device) < 0.3).float()
        
        # iv. 更新数据对象
        current_data = fragment.clone() # 复制现有片段

        # 更新节点相关的属性
        current_data.x = torch.cat([fragment.x, new_atom_type], dim=0)
        current_data.pos = torch.cat([fragment.pos, new_pos], dim=0)
        current_data.pring_out = torch.cat([fragment.pring_out, new_pring_out], dim=0)
        current_data.is_new_node = torch.cat([
            torch.zeros_like(fragment.is_new_node), 
            torch.tensor([[True]], device=device)
        ], dim=0)
        
        
        # v. 更新边 (全连接图)
        # 获取旧的节点数和新原子的索引
        num_old_nodes = fragment.num_nodes
        new_atom_idx = num_old_nodes # 新原子的索引就是旧的节点数，比如4

        # 新边是所有旧原子与新原子之间的双向连接
        old_atom_indices = torch.arange(num_old_nodes, device=device) # 构建长为num_old_nodes的等差数列，步长为1 e.g.[0, 1, 2, 3]

        # 旧原子 -> 新原子
        edges_to_new = torch.stack([
            old_atom_indices, # [0, 1, 2, 3]
            torch.full_like(old_atom_indices, new_atom_idx) # [4, 4, 4, 4]
        ], dim=0)
        # 结果是 [[0, 1, 2, 3], [4, 4, 4, 4]]，代表了边 0->4, 1->4, 2->4, 3->4

        # 新原子 -> 旧原子
        edges_from_new = torch.stack([
            torch.full_like(old_atom_indices, new_atom_idx), # [4, 4, 4, 4]
            old_atom_indices # [0, 1, 2, 3]
        ], dim=0)
        # 结果是 [[4, 4, 4, 4], [0, 1, 2, 3]]，代表了边 4->0, 4->1, 4->2, 4->3

        # 合并新创建的边
        new_edge_index = torch.cat([edges_to_new, edges_from_new], dim=1)
        # [[0, 1, 2],      [[3, 3, 3],      [[0, 1, 2, 3, 3, 3],
        #  [3, 3, 3]]  cat  [0, 1, 2]]  =    [3, 3, 3, 0, 1, 2]]
        num_new_edges = new_edge_index.shape[1] # 获取新边的总数

        # b. 为新边创建 "无键" 属性
        no_bond_idx = 3
        new_edge_attr = F.one_hot(
            torch.full((num_new_edges,), no_bond_idx, device=device), # 指定形状为num_new_edges，填充值为no_bond_idx的一维向量，如tensor([4, 4, 4, 4, 4, 4])
            num_classes = 4
        ).float() # 将上一步的索引列表转换为 one-hot 编码
        # e.g.
        # tensor([[0., 0., 0., 0., 1.],
        #         [0., 0., 0., 0., 1.],
        #         [0., 0., 0., 0., 1.],
        #         [0., 0., 0., 0., 1.],
        #         [0., 0., 0., 0., 1.],
        #         [0., 0., 0., 0., 1.]])

         # c. 将新的边信息与旧的边信息合并
        # 如果旧的 fragment 没有边，edge_attr 可能不存在
        if hasattr(fragment, 'edge_index') and fragment.edge_index is not None:
            final_edge_index = torch.cat([fragment.edge_index, new_edge_index], dim=1)
            final_edge_attr = torch.cat([fragment.edge_attr, new_edge_attr], dim=0)
        else: # 这是处理第一个原子之后，添加第二个原子的情况
            final_edge_index = new_edge_index
            final_edge_attr = new_edge_attr
        
        current_data.edge_index = final_edge_index
        current_data.edge_attr = final_edge_attr

        print(f"  -> 添加新原子后, 'current_data' 准备进入去噪阶段:")
        print(f"     - is_new_node 标记: {current_data.is_new_node.squeeze().tolist()}")
        print(f"  -> [坐标] 阶段一去噪前 (含带噪新原子): Shape={current_data.pos.shape}\n{current_data.pos}")

        # --- 3. 阶段一去噪 (T2 循环) ---
        print(f"步骤 3: 阶段一去噪 (T2={scheduler.T2} -> 1)")

        # ✅ DDIM MODIFICATION: 创建时间步序列
        if args.sampler_type == 'ddim':
            # 创建稀疏的时间步序列，例如 [T2, T2-skip, T2-2*skip, ..., 0]
            # 我们需要包含 0，因为 DDIM 公式需要 t-1
            time_steps_T2 = np.linspace(0, scheduler.T2, args.ddim_steps_T2 + 1).astype(int)
            time_steps_T2 = np.flip(time_steps_T2) # 翻转为 [T2, ..., 0]
        else: # ddpm
            time_steps_T2 = np.arange(scheduler.T2, -1, -1)
        print(f"  -> Phase 1 Timestep Sequence (first 5): {time_steps_T2[:5]}...")
        
        # 这是去噪过程中的数据，我们会不断更新它
        denoising_data = current_data.clone()
        
        for i, _ in enumerate(trange(len(time_steps_T2) - 1, desc="  Phase 1 Denoising")):
            t_current_val = time_steps_T2[i]
            t_prev_val = time_steps_T2[i+1]

            t = torch.tensor([t_current_val], device=device) # 将 t_gen 转换成一个PyTorch 张量 (Tensor)

            # 准备模型输入
            # 这里的 t 是 per-graph 的，模型内部会处理
            target_node_mask = denoising_data.is_new_node.squeeze().bool()
            target_edge_mask = (target_node_mask[denoising_data.edge_index[0]] | target_node_mask[denoising_data.edge_index[1]])
            # edge_index_to_predict = denoising_data.edge_index[:, target_edge_mask]

            if i == 0:
                num_target_edges = target_edge_mask.sum().item()
                print(f"\n  -> [阶段一] 开始去噪。将对 {num_target_edges} 条目标边进行预测。")
            
            # 模型前向传播，只预测目标
            predictions = model(denoising_data, t, target_node_mask, target_edge_mask)
            pred_r0 = predictions['predicted_r0']
            pred_logits_a = predictions['atom_type_logits']
            pred_logits_b = predictions['bond_logits']

            if i == 0:
                print(f"  -> [阶段一] 模型输出 pred_logits_b 的维度: {pred_logits_b.shape}")
                # 验证维度是否匹配
                assert pred_logits_b.shape[0] == num_target_edges, \
                    f"预测的边数 ({pred_logits_b.shape[0]}) 与目标边数 ({num_target_edges}) 不匹配!"

            # if t_gen % 2 == 0:
            #     with torch.no_grad():
            #         pos_norm = torch.linalg.norm(denoising_data.pos[target_node_mask], dim=-1).mean()
            #         pred_r0_norm = torch.linalg.norm(pred_r0, dim=-1).mean()
            #         print(f"\n  [T2={t_gen:04d}] Pos Norm: {pos_norm:.4f}, Predicted R0 Norm: {pred_r0_norm:.4f}")

            # # 进行一步采样
            # # a. 坐标采样
            # pred_noise = scheduler.get_predicted_noise_from_r0(
            #     r_t=denoising_data.pos[target_node_mask],
            #     t=t.expand(target_node_mask.sum()), # 扩展t到节点数
            #     predicted_r0=pred_r0,
            #     schedule_type='delta'
            # )
            
            # # DDPM 采样公式
            # c1 = 1.0 / torch.sqrt(scheduler.gammas[t])
            # c2 = (1.0 - scheduler.gammas[t]) / torch.sqrt(1.0 - scheduler.delta_bars[t])
            
            # pos_mean = c1 * (denoising_data.pos[target_node_mask] - c2 * pred_noise)
            
            # if t_gen > 1:
            #     noise = torch.randn_like(pos_mean)
            #     sigma_t = torch.sqrt(scheduler.posterior_variance_delta[t])
            #     pos_t_minus_1 = pos_mean + sigma_t * noise
            # else:
            #     pos_t_minus_1 = pos_mean

            # #DDIM采样实现
            # alpha_bar_t = scheduler.delta_bars[t]
            # alpha_bar_t_minus_1 = scheduler.delta_bars[t-1] if t_gen > 1 else torch.tensor(1.0, device=device)
            
            # x_t = denoising_data.pos[target_node_mask]
            
            # # 第一步: 计算预测的 x0
            # c1 = torch.sqrt(1.0 - alpha_bar_t)
            # c2 = torch.sqrt(alpha_bar_t)
            # predicted_x0 = (x_t - c1 * pred_noise) / c2
            
            # # 第二步: 计算 x_{t-1}
            # c3 = torch.sqrt(alpha_bar_t_minus_1)
            # c4 = torch.sqrt(1.0 - alpha_bar_t_minus_1)
            # pos_t_minus_1 = c3 * predicted_x0 + c4 * pred_noise

            # 采用 DDPM 后验采样：mu + sigma * eps，其中 x_recon = 第0步坐标
            x_t = denoising_data.pos[target_node_mask]
            # DDIM MODIFICATION: 根据采样器类型选择更新规则
            if args.sampler_type == 'ddim':
                alpha_bar_t = scheduler.delta_bars[t_current_val]
                alpha_bar_prev = scheduler.delta_bars[t_prev_val]
                
                # 1. 计算预测的噪声
                pred_noise = (x_t - torch.sqrt(alpha_bar_t) * pred_r0) / torch.sqrt(1. - alpha_bar_t)
                
                # 2. 计算 x_{t-1}
                # 注意：这里我们使用 eta=0 的确定性 DDIM
                pos_t_minus_1 = (torch.sqrt(alpha_bar_prev) * pred_r0 + 
                                 torch.sqrt(1. - alpha_bar_prev) * pred_noise)

            else: # DDPM (原始代码)
                x_recon = pred_r0
                t_batch = t.expand(target_node_mask.sum()).long()
                
                coef_x0 = scheduler.coef_x0_delta[t_batch].unsqueeze(-1)
                coef_xt = scheduler.coef_xt_delta[t_batch].unsqueeze(-1)
                sigma   = scheduler.std_delta[t_batch].unsqueeze(-1)

                mu = coef_x0 * x_recon + coef_xt * x_t
                if t_current_val > 1:
                    eps = torch.randn_like(mu)
                    pos_t_minus_1 = mu + sigma * eps
                else:
                    pos_t_minus_1 = mu

            # b. 原子类型和边属性采样
            atom_type_t_minus_1 = scheduler.compute_discrete_jump_step(
                x_t=denoising_data.x[target_node_mask],
                pred_x0_logits=pred_logits_a,
                t_current=t_current_val,
                t_previous=t_prev_val,
                schedule_type='delta',
                is_atom=True
            )
            # bond_attr_t_minus_1 = scheduler.compute_discrete_t_minus_1(
            #     x_t=denoising_data.edge_attr[target_edge_mask],
            #     pred_x0_logits=symmetrized_logits_b, # <-- 使用 symmetrized_logits_b
            #     t=t_gen,
            #     schedule_type='delta',
            #     is_atom=False
            # )

            bond_attr_t_minus_1_subset = sample_symmetric_bonds(
                edge_index=denoising_data.edge_index[:, target_edge_mask],
                x_t_bonds=denoising_data.edge_attr[target_edge_mask],
                pred_logits_b=pred_logits_b,
                scheduler=scheduler,
                t_current_val=t_current_val,
                t_prev_val=t_prev_val,
                schedule_type='delta'
            )
            
            # c. 更新 denoising_data 的目标部分
            denoising_data.pos[target_node_mask] = pos_t_minus_1
            denoising_data.x[target_node_mask] = atom_type_t_minus_1
            denoising_data.edge_attr[target_edge_mask] = bond_attr_t_minus_1_subset

        # T2 循环结束，denoising_data 现在是阶段一去噪的结果
        fragment = denoising_data

        newly_denoised_atom_type_idx = fragment.x[-1].argmax().item()
        predicted_symbol = ATOM_MAP[newly_denoised_atom_type_idx]
        print(f"  -> 阶段一去噪完成。新原子类型被预测为: {predicted_symbol} (索引: {newly_denoised_atom_type_idx})")

        print("  -> 新原子与现有片段的连接预测:")
        # 找到所有从新原子出发的边 (这样可以避免打印重复的边)
        source_is_new_mask = (fragment.edge_index[0] == new_atom_idx)
        
        if not source_is_new_mask.any():
            print("    - (未检测到从新原子出发的边)")
        else:
            # 获取这些边的目标节点和属性
            edges_from_new = fragment.edge_index[:, source_is_new_mask]
            bonds_from_new = fragment.edge_attr[source_is_new_mask]
            
            # 获取预测的键类型索引
            bond_indices = bonds_from_new.argmax(dim=1)
            
            # 遍历并打印每条新边
            for i in range(edges_from_new.shape[1]):
                target_atom_idx = edges_from_new[1, i].item()
                bond_idx = bond_indices[i].item()
                bond_symbol = BOND_MAP[bond_idx]
                print(f"    - Atom {new_atom_idx} <--> Atom {target_atom_idx}: {bond_symbol} (索引: {bond_idx})")

        print(f"  -> [坐标] 阶段一去噪后: Shape={fragment.pos.shape}\n{fragment.pos}")
        
        # 更新环指导信息
        print(f"  -> 更新环指导信息...")
        fragment = get_ring_guidance(p_model, fragment)
        print(f"  -> 环预测结果 (pring_out): {fragment.pring_out.squeeze().tolist()}")

        # --- 4. 阶段二去噪 (T1 循环) ---
        print(f"步骤 4: 阶段二去噪 (T1={scheduler.T1} -> 1)")
        # ✅ DDIM MODIFICATION: 为 T1 创建时间步序列
        if args.sampler_type == 'ddim':
            time_steps_T1 = np.linspace(0, scheduler.T1, args.ddim_steps_T1 + 1).astype(int)
            time_steps_T1 = np.flip(time_steps_T1)
        else: # ddpm
            time_steps_T1 = np.arange(scheduler.T1, -1, -1)
        print(f"  -> Phase 2 Timestep Sequence (first 5): {time_steps_T1[:5]}...")

        # 这里的 fragment 是上一步的结果，我们继续对它进行微调
        for i, _ in enumerate(trange(len(time_steps_T1) - 1, desc="  Phase 2 Denoising")):
            t_current_val = time_steps_T1[i]
            t_prev_val = time_steps_T1[i+1]
            
            t = torch.tensor([t_current_val], device=device)
            
            # 准备模型输入，这次是全局预测
            target_node_mask = torch.ones(fragment.num_nodes, dtype=torch.bool, device=device)
            target_edge_mask = torch.ones(fragment.num_edges, dtype=torch.bool, device=device)

            if i == 0:
                num_target_edges = target_edge_mask.sum().item()
                # 注意这里的 num_target_edges 应该等于 fragment.num_edges
                assert num_target_edges == fragment.num_edges
                print(f"\n  -> [阶段二] 开始全局微调。将对全部 {num_target_edges} 条边进行预测。")
            
            predictions = model(fragment, t, target_node_mask, target_edge_mask)
            pred_r0 = predictions['predicted_r0']
            pred_logits_a = predictions['atom_type_logits']
            pred_logits_b = predictions['bond_logits']

            if i == 0:
                print(f"  -> [阶段二] 模型输出 pred_logits_b 的维度: {pred_logits_b.shape}")
                # 验证维度是否匹配
                assert pred_logits_b.shape[0] == num_target_edges, \
                    f"预测的边数 ({pred_logits_b.shape[0]}) 与目标边数 ({num_target_edges}) 不匹配!"

            # if t_gen % 2 == 0:
            #     with torch.no_grad():
            #         pos_norm = torch.linalg.norm(fragment.pos, dim=-1).mean()
            #         pred_r0_norm = torch.linalg.norm(pred_r0, dim=-1).mean()
            #         print(f"\n  [T1={t_gen:04d}] Pos Norm: {pos_norm:.4f}, Predicted R0 Norm: {pred_r0_norm:.4f}")

            # # 全局采样
            # pred_noise = scheduler.get_predicted_noise_from_r0(fragment.pos, t.expand(fragment.num_nodes), pred_r0, 'alpha')
            
            # c1 = 1.0 / torch.sqrt(scheduler.alphas[t])
            # c2 = (1.0 - scheduler.alphas[t]) / torch.sqrt(1.0 - scheduler.alpha_bars[t])
            # pos_mean = c1 * (fragment.pos - c2 * pred_noise)
            
            # if t_gen > 1:
            #     noise = torch.randn_like(pos_mean)
            #     sigma_t = torch.sqrt(scheduler.posterior_variance_alpha[t])
            #     fragment.pos = pos_mean + sigma_t * noise
            # else:
            #     fragment.pos = pos_mean

            # #DDIM采样实现
            # alpha_bar_t = scheduler.alpha_bars[t]
            # alpha_bar_t_minus_1 = scheduler.alpha_bars[t-1] if t_gen > 1 else torch.tensor(1.0, device=device)
            
            # x_t = fragment.pos
            
            # c1 = torch.sqrt(1.0 - alpha_bar_t)
            # c2 = torch.sqrt(alpha_bar_t)
            # predicted_x0 = (x_t - c1 * pred_noise) / c2
            
            # c3 = torch.sqrt(alpha_bar_t_minus_1)
            # c4 = torch.sqrt(1.0 - alpha_bar_t_minus_1)
            # pos_t_minus_1 = c3 * predicted_x0 + c4 * pred_noise

            # 采用 DDPM 后验采样：mu + sigma * eps，其中 x_recon = 第0步坐标
            # ✅ DDIM MODIFICATION: 再次应用采样器逻辑
            if args.sampler_type == 'ddim':
                alpha_bar_t = scheduler.alpha_bars[t_current_val]
                alpha_bar_prev = scheduler.alpha_bars[t_prev_val]
                
                pred_noise = (x_t - torch.sqrt(alpha_bar_t) * pred_r0) / torch.sqrt(1. - alpha_bar_t)
                
                pos_t_minus_1 = (torch.sqrt(alpha_bar_prev) * pred_r0 + 
                                 torch.sqrt(1. - alpha_bar_prev) * pred_noise)
            else: # DDPM
                x_recon = pred_r0
                t_batch = t.expand(fragment.num_nodes).long()
                
                coef_x0 = scheduler.coef_x0_alpha[t_batch].unsqueeze(-1)
                coef_xt = scheduler.coef_xt_alpha[t_batch].unsqueeze(-1)
                sigma   = scheduler.std_alpha[t_batch].unsqueeze(-1)

                mu = coef_x0 * x_recon + coef_xt * x_t
                if t_current_val > 1:
                    eps = torch.randn_like(mu)
                    pos_t_minus_1 = mu + sigma * eps
                else:
                    pos_t_minus_1 = mu
            
            fragment.pos = pos_t_minus_1
            
            fragment.x = scheduler.compute_discrete_jump_step(            
                x_t=fragment.x[target_node_mask],
                pred_x0_logits=pred_logits_a,
                t_current=t_current_val,
                t_previous=t_prev_val,
                schedule_type='alpha',
                is_atom=True
            )
            # fragment.edge_attr = scheduler.compute_discrete_t_minus_1(
            #     x_t=fragment.edge_attr[target_edge_mask],
            #     pred_x0_logits=symmetrized_logits_b_global, # <-- 使用 symmetrized_logits_b_global
            #     t=t_gen,
            #     schedule_type='alpha',
            #     is_atom=False
            # )

            fragment.edge_attr = sample_symmetric_bonds(
                edge_index=fragment.edge_index[:, target_edge_mask],
                x_t_bonds=fragment.edge_attr[target_edge_mask], # 传入更新前的状态
                pred_logits_b=pred_logits_b,
                scheduler=scheduler,
                t_current_val=t_current_val,
                t_prev_val=t_prev_val,
                schedule_type='alpha'
            )

        print(f"  -> [坐标] 阶段二微调后 (中心化前): Shape={fragment.pos.shape}\n{fragment.pos}")

            
        # T1 循环结束，我们得到了一个新增了原子并微调过的分子片段
        
        # 更新环指导信息
        print(f"  -> 再次更新环指导信息...")
        fragment = get_ring_guidance(p_model, fragment)
        print(f"  -> 环预测结果 (pring_out): {fragment.pring_out.squeeze().tolist()}")

        # 将坐标零中心化
        print("  -> 将分子坐标零中心化。")
        fragment.pos = fragment.pos - fragment.pos.mean(dim=0, keepdim=True)
        print(f"  -> [坐标] 零中心化后: Shape={fragment.pos.shape}\n{fragment.pos}")
        
        # --- 5. 检查停止条件 ---
        print("步骤 5: 检查停止条件")
        new_atom_idx = num_existing_atoms
        is_connected = check_connectivity(new_atom_idx, fragment)
        
        if not is_connected and num_existing_atoms >= args.min_atoms:
            print(f"  -> 新原子未连接到片段，生成终止。")
            # 终止前，移除最后一个未连接的原子
            # (这是一个可选的清理步骤)
            # --- 关键修改：添加 device 参数 ---
            final_mask = torch.ones(fragment.num_nodes, 
                                    dtype=torch.bool, 
                                    device=fragment.x.device) # <-- 和 fragment 在同一设备上
            final_mask[new_atom_idx] = False
            fragment = fragment.subgraph(final_mask)
            break
            
    print("\n--- 分子生成完成 ---")
    num_bonds = fragment.num_edges // 2 if hasattr(fragment, 'num_edges') else 0
    print(f"最终分子包含 {fragment.num_nodes} 个原子和 {num_bonds} 条化学键。")
    return fragment