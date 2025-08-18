import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import torch.nn as nn
from tqdm import trange
from src.training.scheduler import HierarchicalDiffusionScheduler

# ==============================================================================
# 辅助函数
# ==============================================================================

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

    # --- 1. 从一个原子开始 ---
    print("步骤 1: 随机采样第一个原子")
    # a. 随机原子类型 (H,C,N,O,F)
    atom_type_idx = torch.randint(0, 5, (1,), device=device) # 输出单个元素的一维张量
    atom_type = F.one_hot(atom_type_idx, num_classes=6).float() # 6类，最后一类是吸收态
    
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

    # --- 2. 自回归生成循环 ---
    for num_existing_atoms in range(1, args.max_atoms):
        print(f"\n--- [主循环] 当前原子数: {num_existing_atoms}, 开始生成第 {num_existing_atoms + 1} 个原子 ---")
        
        # a. 添加带噪的新原子
        print("步骤 2a: 添加带噪的新原子")
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
        no_bond_idx = 4
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

        # --- 3. 阶段一去噪 (T2 循环) ---
        print(f"步骤 3: 阶段一去噪 (T2={scheduler.T2} -> 1)")
        
        # 这是去噪过程中的数据，我们会不断更新它
        denoising_data = current_data.clone()
        
        for t_gen in trange(scheduler.T2, 0, -1, desc="  Phase 1 Denoising"):
            t = torch.tensor([t_gen], device=device) # 将 t_gen 转换成一个PyTorch 张量 (Tensor)
            
            # 准备模型输入
            # 这里的 t 是 per-graph 的，模型内部会处理
            target_node_mask = denoising_data.is_new_node.squeeze().bool()
            target_edge_mask = (target_node_mask[denoising_data.edge_index[0]] | target_node_mask[denoising_data.edge_index[1]])
            edge_index_to_predict = denoising_data.edge_index[:, target_edge_mask]
            
            # 模型前向传播，只预测目标
            predictions = model(denoising_data, t, target_node_mask, edge_index_to_predict)
            pred_noise = predictions['predicted_r0']
            pred_logits_a = predictions['atom_type_logits']
            pred_logits_b = predictions['bond_logits']

            # # 进行一步采样
            # # a. 坐标采样
            # pred_noise = scheduler.get_predicted_noise_from_r0(
            #     r_t=denoising_data.pos[target_node_mask],
            #     t=t.expand(target_node_mask.sum()), # 扩展t到节点数
            #     predicted_r0=pred_r0,
            #     schedule_type='delta'
            # )
            
            # DDPM 采样公式
            c1 = 1.0 / torch.sqrt(scheduler.gammas[t])
            c2 = (1.0 - scheduler.gammas[t]) / torch.sqrt(1.0 - scheduler.delta_bars[t])
            
            pos_mean = c1 * (denoising_data.pos[target_node_mask] - c2 * pred_noise)
            
            if t_gen > 1:
                noise = torch.randn_like(pos_mean)
                sigma_t = torch.sqrt(scheduler.posterior_variance_delta[t])
                pos_t_minus_1 = pos_mean + sigma_t * noise
            else:
                pos_t_minus_1 = pos_mean

            # b. 原子类型和边属性采样
            atom_type_t_minus_1 = scheduler.compute_discrete_t_minus_1(
                x_t=denoising_data.x[target_node_mask],
                pred_x0_logits=pred_logits_a,
                t=t_gen,
                schedule_type='delta',
                is_atom=True
            )
            bond_attr_t_minus_1 = scheduler.compute_discrete_t_minus_1(
                x_t=denoising_data.edge_attr[target_edge_mask],
                pred_x0_logits=pred_logits_b,
                t=t_gen,
                schedule_type='delta',
                is_atom=False
            )
            
            # c. 更新 denoising_data 的目标部分
            denoising_data.pos[target_node_mask] = pos_t_minus_1
            denoising_data.x[target_node_mask] = atom_type_t_minus_1
            denoising_data.edge_attr[target_edge_mask] = bond_attr_t_minus_1

        # T2 循环结束，denoising_data 现在是阶段一去噪的结果
        fragment = denoising_data
        
        # 更新环指导信息
        fragment = get_ring_guidance(p_model, fragment)

        # --- 4. 阶段二去噪 (T1 循环) ---
        print(f"步骤 4: 阶段二去噪 (T1={scheduler.T1} -> 1)")
        # 这里的 fragment 是上一步的结果，我们继续对它进行微调
        for t_gen in trange(scheduler.T1, 0, -1, desc="  Phase 2 Denoising"):
            t = torch.tensor([t_gen], device=device) # 将 t_gen 转换成一个PyTorch 张量 (Tensor)
            
            # 准备模型输入，这次是全局预测
            target_node_mask = torch.ones(fragment.num_nodes, dtype=torch.bool, device=device)
            edge_index_to_predict = torch.ones(fragment.num_edges, dtype=torch.bool, device=device)
            
            predictions = model(fragment, t, target_node_mask, edge_index_to_predict)
            pred_noise = predictions['predicted_r0']
            pred_logits_a = predictions['atom_type_logits']
            pred_logits_b = predictions['bond_logits']
            
            # # 全局采样
            # pred_noise = scheduler.get_predicted_noise_from_r0(fragment.pos, t.expand(fragment.num_nodes), pred_r0, 'alpha')
            c1 = 1.0 / torch.sqrt(scheduler.alphas[t])
            c2 = (1.0 - scheduler.alphas[t]) / torch.sqrt(1.0 - scheduler.alpha_bars[t])
            pos_mean = c1 * (fragment.pos - c2 * pred_noise)
            
            if t_gen > 1:
                noise = torch.randn_like(pos_mean)
                sigma_t = torch.sqrt(scheduler.posterior_variance_alpha[t])
                fragment.pos = pos_mean + sigma_t * noise
            else:
                fragment.pos = pos_mean
            
            fragment.x = scheduler.compute_discrete_t_minus_1(            
                x_t=fragment.x[target_node_mask],
                pred_x0_logits=pred_logits_a,
                t=t_gen,
                schedule_type='alpha',
                is_atom=True
            )
            fragment.edge_attr = scheduler.compute_discrete_t_minus_1(
                x_t=fragment.edge_attr[target_edge_mask],
                pred_x0_logits=pred_logits_b,
                t=t_gen,
                schedule_type='alpha',
                is_atom=False
            )
            
        # T1 循环结束，我们得到了一个新增了原子并微调过的分子片段
        
        # 更新环指导信息
        fragment = get_ring_guidance(p_model, fragment)

        # 将坐标零中心化
        fragment.pos = fragment.pos - fragment.pos.mean(dim=0, keepdim=True)
        
        # --- 5. 检查停止条件 ---
        print("步骤 5: 检查停止条件")
        new_atom_idx = num_existing_atoms
        is_connected = check_connectivity(new_atom_idx, fragment)
        
        if not is_connected and num_existing_atoms > args.min_atoms:
            print(f"  -> 新原子未连接到片段，生成终止。")
            # 终止前，移除最后一个未连接的原子
            # (这是一个可选的清理步骤)
            final_mask = torch.ones(fragment.num_nodes, dtype=torch.bool)
            final_mask[new_atom_idx] = False
            fragment = fragment.subgraph(final_mask)
            break
            
    print("\n--- 分子生成完成 ---")
    return fragment