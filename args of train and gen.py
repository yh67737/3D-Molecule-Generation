# args 参数列表
# 1. 设备配置 (Device Configuration)

# args.device
# 作用: 指定用于训练的计算设备。
# 代码中的使用: device = args.device, model.to(device), clean_batch.to(device) 等。
# 建议类型: torch.device 或 str。
# 示例值: "cuda", "cpu", "mps"。

# 2. 训练流程控制 (Training Flow Control)

# args.epochs
# 作用: 定义训练的总周期数。
# 代码中的使用: for epoch in range(1, args.epochs + 1):。
# 建议类型: int。
# 示例值: 100, 500, 1000。

# args.batch_size
# 作用: 定义每个批次包含的图（分子片段）的数量。
# 代码中的使用: dataloader_train = DataLoader(..., batch_size=args.batch_size), num_val_samples = int(args.batch_size * args.batch_ratio)。
# 建议类型: int。
# 示例值: 32, 64, 128。

# args.learning_rate
# 作用: 设置优化器（Adam）的学习率。
# 代码中的使用: optimizer = optim.Adam(..., lr=args.learning_rate)。
# 建议类型: float。
# 示例值: 1e-3, 2e-4, 5e-5。

# 3. 损失函数权重 (Loss Weights)

# args.w_a
# 作用: 原子类型损失（loss_a）的权重。
# 代码中的使用: loss_I = args.w_a * lossI_a + ..., loss_II = args.w_a * lossII_a + ...。
# 建议类型: float。
# 示例值: 1.0。

# args.w_r
# 作用: 原子坐标损失（loss_r）的权重。
# 代码中的使用: loss_I = ... + args.w_r * lossI_r + ..., loss_II = ... + args.w_r * lossII_r + ...。
# 建议类型: float。
# 示例值: 1.0, 0.1。

# args.w_b
# 作用: 边类型损失（loss_b）的权重。
# 代码中的使用: loss_I = ... + args.w_b * lossI_b, loss_II = ... + args.w_b * lossII_b。
# 建议类型: float。
# 示例值: 1.0。

# args.lambda_aux
# 作用: D3PM 混合损失中辅助项的权重 λ。
# 代码中的使用: calculate_atom_type_loss(..., lambda_aux=args.lambda_aux), calculate_bond_type_loss(..., lambda_aux=args.lambda_aux)。
# 建议类型: float。
# 示例值: 0.001, 0.01 (根据 D3PM 论文建议)。

# 4. 验证集配置 (Validation Set Configuration)

# args.batch_ratio_val
# 作用: 用于计算在验证阶段每个 epoch 要采样的总样本数。num_samples = batch_size * batch_ratio_val。
# 代码中的使用: num_val_samples = int(args.batch_size * args.batch_ratio_val)。
# 建议类型: int 或 float。
# 示例值: 10 (表示每个验证 epoch 采样 10 个批次的数据量)。

# 5. 模型保存路径 (Model Saving Paths)

# args.final_model_path
# 作用: 指定训练完成后最终模型的保存路径。
# 代码中的使用: final_model_path = args.final_model_path, torch.save(..., final_model_path)。
# 建议类型: str。
# 示例值: './checkpoints/final_model.pt'。

# (可选) args.best_model_path
# 作用: 指定在训练过程中保存验证损失最佳的模型的路径。虽然您注释掉了这部分，但如果启用，也需要这个参数。
# 代码中的使用: torch.save(..., args.best_model_path)。
# 建议类型: str。
# 示例值: './checkpoints/best_model.pt'。

# args.num_workers

# args.min_atoms

# args.max_atoms


# main.py 中 argparse 的设置示例
import argparse

parser = argparse.ArgumentParser()

# 训练流程
parser.add_argument('--epochs', type=int, default=500, help='Total number of training epochs')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the Adam optimizer(model)')
parser.add_argument('--s_learning_rate', type=float, default=1e-4, help='Learning rate for the Adam optimizer(s_model)')
# 损失权重
parser.add_argument('--w_a', type=float, default=1.0, help='Weight for atom type loss')
parser.add_argument('--w_r', type=float, default=1.0, help='Weight for coordinate loss')
parser.add_argument('--w_b', type=float, default=1.0, help='Weight for bond type loss')
parser.add_argument('--lambda_aux', type=float, default=0.01, help='Weight for auxiliary loss term in D3PM')
# 验证配置
parser.add_argument('--batch_ratio_val', type=int, default=10, help='Ratio to determine number of validation samples (num_val_samples = batch_size * batch_ratio)')

# 保存路径
parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Path to save the model')

# 扩散过程
parser.add_argument('--T_full', type=int, default=1000, 
                    help="'alpha' 调度的完整长度 (The full length of the alpha schedule)")

parser.add_argument('--T1', type=int, default=100, 
                    help="'alpha' 调度实际使用的步数 (The actual steps used in the alpha schedule)")

parser.add_argument('--T2', type=int, default=900, 
                    help="'gamma'/'delta' 调度的步数 (The steps for the gamma/delta schedule)")

parser.add_argument('--s', type=float, default=0.008, 
                    help="Cosine schedule 的偏移量 (The offset for the Cosine schedule)")

# 生成参数
parser.add_argument(
    '--max_atoms', 
    type=int, 
    default=108, 
    help="生成分子的最大原子数。这是主生成循环的上限。"
)

parser.add_argument(
    '--min_atoms', 
    type=int, 
    default=8, 
    help="在因新原子未连接而停止生成之前，所要求的最小原子数。"
)


args = parser.parse_args()