import os
import argparse
import torch
import numpy as np
import random
from pathlib import Path
from datetime import datetime
from torch.utils.data import Dataset, DistributedSampler
from torch_geometric.data import Data as PyGData # 使用别名以避免混淆
from contextlib import suppress
import torch.cuda.amp
import torch_geometric.utils
from torch_geometric.loader import DataLoader
from src.utils.logger import FileLogger # Equiformer提供的通用模块
from timm.utils import NativeScaler # timm是一个强大的深度学习第三方库，需要安装
# distributed training
from src.utils import utils # highly general, boilerplate code for setting up distributed training in PyTorch and are not specifically tied to the Equiformer model or its task.
from src.models.EDiT_network.e_dit_network import E_DiT_Network
from src.training.train_B import train
from src.training.scheduler import HierarchicalDiffusionScheduler
from dataset_for_json import JsonFragmentDataset
from torch.utils.data import RandomSampler
from src.utils.custom_sampler import DistributedFixedLengthRandomSampler

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def compute_dataset_stats(data_loader, logger, print_freq=100):
    """
    计算并记录数据集中图的平均节点数、边数和度数。
    此函数已修改为直接使用 PyG Data 对象中的属性，而不是重新计算边。
    """
    logger.info('\n--- Calculating Dataset Statistics ---')
        
    avg_node = AverageMeter()
    avg_edge = AverageMeter()
    avg_degree = AverageMeter()
    
    for step, data in enumerate(data_loader):###
        # 注意：DataLoader 会自动将单个样本打包成 Batch 对象
        # 我们需要按图来分解 Batch
        from torch_geometric.data import Batch
        if isinstance(data, Batch):
            graph_list = data.to_data_list()
        else: # 如果 batch_size=1, 可能不是 Batch 对象
            graph_list = [data]

        for graph in graph_list:
            num_nodes = graph.num_nodes
            num_edges = graph.num_edges
            
            # 计算度数
            if num_edges > 0:
                degree = torch_geometric.utils.degree(graph.edge_index[0], num_nodes=num_nodes).mean()
            else:
                degree = torch.tensor(0.0)

            avg_node.update(num_nodes)
            avg_edge.update(num_edges)
            avg_degree.update(degree.item())
            
    log_str = (f"Average #Nodes: {avg_node.avg:.2f}, "
               f"Average #Edges: {avg_edge.avg:.2f}, "
               f"Average Degree: {avg_degree.avg:.2f}")
    logger.info(log_str)
    logger.info('--- Statistics Calculation Complete ---\n')

    return avg_node.avg, avg_edge.avg, avg_degree.avg

## 模块一：参数与配置
def get_args_parser():
    """
    定义并解析所有可配置的命令行参数。
    """
    parser = argparse.ArgumentParser('Diffusion-Autoregressive Molecule Generation', add_help=False)

    # --- 通用设置 ---
    parser.add_argument('--output_dir', type=str, default='./output', help='Path to save logs, checkpoints, and generated molecules.')
    parser.add_argument('--args_save_dir', type=str, default='./saved_args', help='Directory to save the configuration object of each run.') 
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use for training.')
    parser.add_argument('--val_log_freq', type=int, default=5)
    parser.add_argument('--val_thre', type=int, default=10)
    parser.add_argument('--resume_ckpt', type=str, default=None, 
                        help="断点加载参数路径 (Path to checkpoint to resume training from)")

    # 训练流程
    parser.add_argument('--epochs', type=int, default=500, help='Total number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the Adam optimizer(model)')
    parser.add_argument('--lr_min_factor', type=float, default=0.001, 
                    help="学习率下限因子，最终学习率 = lr_min_factor × 初始学习率")
    parser.add_argument('--pos_noise_std', type=float, default=0.05,
                    help="坐标噪声增强的标准差，用于在训练时对节点位置添加高斯噪声作为数据增强")
    parser.add_argument('--warmup_epochs_for_EDiT', type=int, default=5, 
                         help='Number of epochs for linear warmup.')
    parser.add_argument('--warmup_factor', type=float, default=1e-2, 
                         help='The initial learning rate will be this factor times the base learning rate.')
    # 损失权重
    parser.add_argument('--w_a', type=float, default=1.0, help='Weight for atom type loss')
    parser.add_argument('--w_r', type=float, default=1.0, help='Weight for coordinate loss')
    parser.add_argument('--w_b', type=float, default=1.0, help='Weight for bond type loss')
    parser.add_argument('--lambda_aux', type=float, default=0.01, help='Weight for auxiliary loss term in D3PM')
    # 保存路径
    # parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Path to save the model')
    # 扩散过程
    parser.add_argument('--T_full', type=int, default=1000, 
                        help="'alpha' 调度的完整长度 (The full length of the alpha schedule)")
    parser.add_argument('--T1', type=int, default=100, 
                        help="'alpha' 调度实际使用的步数 (The actual steps used in the alpha schedule)")
    parser.add_argument('--T2', type=int, default=1000, 
                        help="'gamma'/'delta' 调度的步数 (The steps for the gamma/delta schedule)")
    parser.add_argument('--s', type=float, default=0.008, 
                        help="Cosine schedule 的偏移量 (The offset for the Cosine schedule)")
    # 生成参数
    parser.add_argument('--max_atoms', type=int, default=50, help="生成分子的最大原子数。这是主生成循环的上限。")
    parser.add_argument('--min_atoms', type=int, default=3, help="在因新原子未连接而停止生成之前，所要求的最小原子数。")
    parser.add_argument('--num_generate', type=int, default=100, help='要生成的分子数量')
    parser.add_argument('--model_ckpt', type=str, default='./output/2025-08-16_12-43-43/checkpoints/checkpoint_epoch_10.pth', help='生成模型路径') # 修改
    
    # --- 数据集参数 ---
    parser.add_argument('--data_split_path', type=str, default='./data_splits',
                        help='Directory to save/load data split indices.')
    parser.add_argument('--fragment_data_dir', type=str, default='./prepared_data/32for_test',
                        help='Directory containing some JSON fragment files.')  ###
    # parser.add_argument('--fragment_data_dir', type=str, default='./prepared_data/gdb9_bfs_fragments_json', help='Directory containing the JSON fragment files.')
    parser.add_argument('--val_split_percentage', type=float, default=0.1, help='Percentage of data to use for validation.')
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--val_batch_size', type=int, default=2)
    parser.add_argument('--batch_ratio_val', type=int, default=10, help='Ratio to determine number of validation samples (num_val_samples = batch_size * batch_ratio)')
    parser.add_argument('--batch_ratio_train', type=int, default=10, help='Ratio to determine number of training samples per epoch (num_train_samples = batch_size * batch_ratio)')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--accumulation_steps', type=int, default=1)

    # E-DiT参数
    # --- 模型架构参数 (Model Architecture) ---
    g_arch = parser.add_argument_group('Architecture')
    g_arch.add_argument('--num_blocks', type=int, default=6, help='Number of E-DiT blocks.')
    g_arch.add_argument('--num_heads', type=int, default=6, help='Number of attention heads.')
    g_arch.add_argument('--norm_layer', type=str, default='layer', help='Type of normalization layer (e.g., "layer").')
    g_arch.add_argument('--time_embed_dim', type=int, default=96, help='Dimension of timestep embedding.')

    # --- Irreps 参数 (Irreps Definitions) ---
    g_irreps = parser.add_argument_group('Irreps')
    g_irreps.add_argument('--irreps_node_hidden', type=str, default='96x0e+48x1o+24x2e',
                          help='Hidden node feature irreps.')
    g_irreps.add_argument('--irreps_edge', type=str, default='96x0e+48x1o+24x2e', help='Hidden edge feature irreps.')
    g_irreps.add_argument('--irreps_node_attr', type=str, default='6x0e', help='Node attribute (atom type) irreps.')
    g_irreps.add_argument('--irreps_edge_attr_type', type=str, default='5x0e',
                          help='Edge attribute (bond type) irreps.')
    g_irreps.add_argument('--irreps_sh', type=str, default='1x0e+1x1e+1x2e', help='Spherical harmonics irreps.')
    g_irreps.add_argument('--irreps_head', type=str, default='64x0e+32x1o+16x2e', help='Single attention head irreps.')
    g_irreps.add_argument('--irreps_mlp_mid', type=str, default='192x0e+96x1o+48x2e',
                          help='Irreps for the middle layer of FFN.')
    g_irreps.add_argument('--irreps_pre_attn', type=str, default=None,
                          help='Optional irreps for pre-attention linear layer.')

    # --- 嵌入层参数 (Embedding Layers) ---
    g_embed = parser.add_argument_group('Embeddings')
    g_embed.add_argument('--num_atom_types', type=int, default=6, help='Number of atom types for embedding.')
    g_embed.add_argument('--num_bond_types', type=int, default=5, help='Number of bond types for embedding.')
    g_embed.add_argument('--node_embedding_hidden_dim', type=int, default=96,
                         help='Hidden dimension in node embedding MLP.')
    g_embed.add_argument('--bond_embedding_dim', type=int, default=96, help='Hidden dimension in edge embedding MLP.')
    g_embed.add_argument('--edge_update_hidden_dim', type=int, default=96,
                         help='Hidden dimension in EdgeUpdateNetwork MLP.')
    g_embed.add_argument('--num_rbf', type=int, default=96, help='Number of radial basis functions.')
    g_embed.add_argument('--rbf_cutoff', type=float, default=5.0, help='Cutoff radius for RBF.')
    g_embed.add_argument('--fc_neurons', type=int, nargs='+', default=[64, 64],
                         help='List of hidden layer sizes for FC network in attention.')
    g_embed.add_argument('--avg_degree', type=float, default=9.21, help='Average degree of nodes in the dataset.')

    # --- 注意力机制参数 (Attention Mechanism) ---
    g_attn = parser.add_argument_group('Attention')
    g_attn.add_argument('--rescale_degree', action='store_true', default=False,
                        help='If set, rescale features by node degree in attention.')
    g_attn.add_argument('--nonlinear_message', action='store_true', default=False,
                        help='If set, use non-linearity in message calculation.')

    # --- 输出头参数 (Output Head) ---
    g_output = parser.add_argument_group('OutputHead')
    g_output.add_argument('--hidden_dim', type=int, default=128,
                          help='Hidden dimension for the final MLP in output heads.')

    # --- 训练和正则化 (Training & Regularization) ---
    g_train = parser.add_argument_group('Training')
    g_train.add_argument('--alpha_drop', type=float, default=0.2, help='Dropout rate for attention scores.')
    g_train.add_argument('--proj_drop', type=float, default=0.0,
                         help='Dropout rate for the final projection layer in attention/FFN.')
    g_train.add_argument('--out_drop', type=float, default=0.0, help='Dropout rate for the output heads.')
    g_train.add_argument('--drop_path_rate', type=float, default=0.0, help='Stochastic depth drop rate.')
    
    # 环生成指导网络
    parser.add_argument('--ring_guide_ckpt', type=str, default='./src/models/ring_network/ring_predictor_epoch_50.pt', help='Path to pre-trained ring guidance network checkpoint.')

    parser.add_argument('--optimizer', type=str, default='sam', help='Optimizer (e.g., adamw, sgd).')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay for optimizer.')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Number of warmup epochs for LR scheduler.')
    
    # AMP
    parser.add_argument('--amp', action='store_true', help='Enable automatic mixed precision training.')
    parser.add_argument('--no-amp', action='store_false', dest='amp')
    parser.set_defaults(amp=False)

    # 数据统计计算
    parser.add_argument('--compute_stats', action='store_true',
                        help='If specified, compute dataset stats and exit.')

    # 分布式训练参数
    parser.add_argument('--world_size', default=1, type=int, help='Number of distributed processes.')
    parser.add_argument('--dist_url', default='env://', help='URL used to set up distributed training.')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--distributed', action='store_true')

    return parser

def main(args):
    """
    项目主执行函数
    """
    ## 模块2: 环境初始化

    # 配置多GPU训练环境
    utils.init_distributed_mode(args) # automatically detect if the script is being run in a multi-GPU environment:check related variables and initialize the distributed communication backend.
    is_main_process = (args.rank == 0)

    # 检查并创建输出文件夹
    # if is_main_process: # 防止多个进程竞争IO操作
    #     # 获取当前运行的时间戳
    #     timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #     # 将run_dir作为一个新的属性，run_dir = output_dir + timestamp
    #     args.run_dir = os.path.join(args.output_dir, timestamp) 
    #     print(f"All outputs for this run will be saved to: {args.run_dir}")

    #     # 定义所有需要创建的的子文件夹名称
    #     sub_dirs = {
    #         "checkpoints": os.path.join(args.run_dir, "checkpoints"), # 训练权重文件
    #         "generated_pyg": os.path.join(args.run_dir, "generated_pyg"), # 新生成的PyG数据
    #         "generated_images": os.path.join(args.run_dir, "generated_images"), # PyG数据画图得到的分子图
    #         "tb_logs": os.path.join(args.run_dir, "tb_logs"), # TensorBoard 日志
    #         "results": os.path.join(args.run_dir, "results") # 评估结果
    #     }
        
    #     # 循环创建所有文件夹
    #     for path in sub_dirs.values():
    #         Path(path).mkdir(parents=True, exist_ok=True)
        
    #     # 保存配置(args)对象
    #     # 从 run_dir中提取时间戳，确保文件名与本次运行完全对应
    #     timestamp_from_path = os.path.basename(args.run_dir)
        
    #     # 构建保存文件的完整路径
    #     args_save_filename = f"args_{timestamp_from_path}.pt"
    #     args_save_path = os.path.join(args.args_save_dir, args_save_filename)
        
    #     # 保存args对象
    #     torch.save(args, args_save_path)
    #     print(f"Configuration for this run has been saved to: {args_save_path}")
        
    #     # 准备要广播的对象列表:如果不做处理，只有主进程的args对象增加了新创建的路径，其他进程的args对象则没有，后续调用时就会出错
    #     # 包含动态生成的run_dir所有子目录路径字典
    #     objects_to_broadcast = [args.run_dir, sub_dirs]
    
    # else:
    #     # 其他进程准备好相同结构的空列表接收数据
    #     objects_to_broadcast = [None, None]
    
    # # 执行广播操作(只有在分布式模式下需要广播)
    # if args.distributed:
    #     # broadcast_object_list会将src=0(主进程)的列表内容发送给所有其他进程
    #     torch.distributed.broadcast_object_list(objects_to_broadcast, src=0)
    
    # # 所有进程用广播来的数据更新自己的配置
    # # 主进程已经有这些数据了，但为了代码统一，可以都执行
    # run_dir_synced, sub_dirs_synced = objects_to_broadcast
    
    # args.run_dir = run_dir_synced
    # for key, path in sub_dirs_synced.items():
    #     setattr(args, f"{key}_dir", path)

    # # 设置屏障:确保所有进程都执行完前面的所有步骤后(参数更新)一起进入下一个阶段，防止时序问题
    # if args.distributed:
    #     torch.distributed.barrier()

    if is_main_process:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.run_dir = os.path.join(args.output_dir, timestamp) 
        
        sub_dirs = {
            "checkpoints": os.path.join(args.run_dir, "checkpoints"),
            "generated_pyg": os.path.join(args.run_dir, "generated_pyg"),
            "generated_images": os.path.join(args.run_dir, "generated_images"),
            "tb_logs": os.path.join(args.run_dir, "tb_logs"),
            "results": os.path.join(args.run_dir, "results")
        }
        for path in sub_dirs.values():
            Path(path).mkdir(parents=True, exist_ok=True)
            
        timestamp_from_path = os.path.basename(args.run_dir)
        args_save_filename = f"args_{timestamp_from_path}.pt"
        args_save_path = os.path.join(args.args_save_dir, args_save_filename)

        # --- 关键修改：在这里增加两行代码 ---
        # 在保存 args 之前，先把所有子目录路径添加到 args 对象中
        for key, path in sub_dirs.items():
            setattr(args, f"{key}_dir", path)
        # 同样地，把 args_save_path 也添加进去
        args.args_save_path = args_save_path
        # ------------------------------------
        
        # 现在再保存，args 对象已经是完整的了
        torch.save(args, args_save_path)
        
        print(f"All outputs for this run will be saved to: {args.run_dir}")
        print(f"Configuration for this run has been saved to: {args_save_path}")
        
        objects_to_broadcast = [args.run_dir, sub_dirs, args_save_path]
    else:
        objects_to_broadcast = [None, None, None]

    # 步骤 2: 广播路径信息给所有进程

    if args.distributed:
        torch.distributed.broadcast_object_list(objects_to_broadcast, src=0)

    # 步骤 3: 所有进程用接收到的信息更新自己的配置
    run_dir_synced, sub_dirs_synced, args_save_path_synced = objects_to_broadcast
    args.run_dir = run_dir_synced
    args.args_save_path = args_save_path_synced
    for key, path in sub_dirs_synced.items():
        setattr(args, f"{key}_dir", path)

    # 步骤 4: 在所有进程都获得了正确的 run_dir 后，才初始化 Logger
    logger = FileLogger(
        output_dir=args.run_dir, 
        is_master=is_main_process,
        is_rank0=is_main_process
    )

    # 步骤 5: 设置屏障，确保所有进程都完成了初始化
    if args.distributed:
        torch.distributed.barrier()

    # 步骤 6: 现在可以安全地进行日志记录了
    logger.info("Logger initialized. Distributed config synchronized.")
    if is_main_process:
        logger.info(f"Configuration save path: {args.args_save_path}")
    logger.info(f"Master process (rank 0) configuration:\n{args}")

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    if args.device == 'cuda' and torch.cuda.is_available():
        # This log is helpful for confirming which GPU each process is using
        logger.info(f"Process {args.rank} assigned to CUDA device: {torch.cuda.get_device_name(args.local_rank)}")

    ## 模块3: 数据准备(Dataset & DataLoader)
    logger.info("--- Starting Data Preparation (JSON Fragment Version) ---")
    
    # 实例化自定义数据集：The dataset class will discover all .json files in the directory.
    full_dataset = JsonFragmentDataset(root_dir=args.fragment_data_dir)
    num_fragments = len(full_dataset)

    # 加载或创建数据划分索引
    split_file = os.path.join(args.data_split_path, 'bfs_json_split_indices.pt') 

    if os.path.exists(split_file):
        logger.info(f"Loading existing data split from: {split_file}")
        split_data = torch.load(split_file)
        train_indices, val_indices = split_data['train'], split_data['val']
    else:
        logger.info("No existing data split found. Creating a new one...")
        if is_main_process:
            logger.info(f"Total number of fragments found: {num_fragments}")
            indices = list(range(num_fragments))
            random.shuffle(indices)
            split_point = int(np.floor(args.val_split_percentage * num_fragments))
            val_indices, train_indices = indices[:split_point], indices[split_point:]
            
            logger.info(f"Splitting data: {len(train_indices)} training, {len(val_indices)} validation.")
            Path(args.data_split_path).mkdir(parents=True, exist_ok=True)
            torch.save({'train': train_indices, 'val': val_indices}, split_file)
            logger.info(f"New data split indices saved to: {split_file}")
        
        # Ensure all processes wait for the main process to save the file
        if args.distributed:
            torch.distributed.barrier()
        
        # All processes load the saved split file
        split_data = torch.load(split_file, map_location='cpu')
        train_indices, val_indices = split_data['train'], split_data['val']

    # 使用PyTorch的Subset来根据索引创建训练集和验证集
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

    logger.info(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")

    # 计算每个 epoch 需要抽取的样本数
    num_train_samples_per_epoch = args.train_batch_size * args.batch_ratio_train
    num_val_samples_per_epoch = args.val_batch_size * args.batch_ratio_val

    logger.info(f"定义一个 'epoch' 包含 {num_train_samples_per_epoch} 个训练样本和 {num_val_samples_per_epoch} 个验证样本。")

    # 创建DataLoader
    # if args.distributed:
    #     train_sampler = DistributedSampler(train_dataset, shuffle=True)
    #     val_sampler = DistributedSampler(val_dataset, shuffle=False)
    # else:
    #     train_sampler = None
    #     val_sampler = None
    # 创建 DataLoader 的 Sampler
    if args.distributed:
        # 使用自定义分布式采样器
        train_sampler = DistributedFixedLengthRandomSampler(
            train_dataset, 
            num_samples_per_epoch=num_train_samples_per_epoch,
            seed=args.seed
        )
        val_sampler = DistributedFixedLengthRandomSampler(
            val_dataset,
            num_samples_per_epoch=num_val_samples_per_epoch,
            seed=args.seed
        )
    else:
        # 在单 GPU 情况下，使用标准的 RandomSampler
        # 它内置了对有放回采样和指定样本数的支持
        train_sampler = RandomSampler(
            train_dataset, 
            replacement=True, 
            num_samples=num_train_samples_per_epoch
        )
        val_sampler = RandomSampler(
            val_dataset, 
            replacement=True, 
            num_samples=num_val_samples_per_epoch
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        sampler=train_sampler,
        shuffle=False, # 当使用自定义 sampler 时，shuffle 必须为 False
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    logger.info("DataLoaders created.")
    logger.info("--- Data Preparation Complete ---")

    # 模块6：计算数据集统计量
    if args.compute_stats:
        # 使用训练集加载器来计算统计数据
        avg_nodes, avg_edges, avg_degree = compute_dataset_stats(train_loader, logger=logger)
        print(f"avg_nodes = {avg_nodes}, avg_edges = {avg_edges}, avg_degree = {avg_degree}")
        return # 计算完毕后直接退出程序
    
    ## 模块4: 模型实例化与配置 

    # 再次设置随机种子，确保模型初始化的随机性绝对可控，不受数据加载过程的任何影响
    torch.manual_seed(args.seed + args.rank) 
    np.random.seed(args.seed + args.rank)
    random.seed(args.seed + args.rank)

    logger.info("--- Initializing Models ---")
    
    # 实例化各个模块
    
    # b. 生成网络
    generator_network = E_DiT_Network(args)
    
    # 将模型移动到指定设备
    generator_network.to(device)
    logger.info("Models moved to device.")

    # 模型分布式训练封装
    if args.distributed:
        generator_network = torch.nn.parallel.DistributedDataParallel(
            generator_network, 
            device_ids=[args.local_rank],
            find_unused_parameters=True 
        )
        logger.info("Models wrapped with DistributedDataParallel (find_unused_parameters=True).")
    # 记录模型和参数统计
    if is_main_process:
        logger.info("--- Model Architectures ---")
        logger.info(generator_network)
        
        total_params_generator = sum(p.numel() for p in generator_network.parameters() if p.requires_grad)
        logger.info("--- Parameter Statistics ---")
        logger.info(f"Generator Network Trainable Params: {total_params_generator / 1e6:.2f} M")
        logger.info("-----------------------------")

        logger.info("--- Model Initialization Complete ---")
    
    ## 模块5: AMP(自动混合精度)配置
    amp_autocast = suppress
    loss_scaler = None
    if args.amp:
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        logger.info("Automatic Mixed Precision (AMP) enabled.")

    # 模型训练
    scheduler = HierarchicalDiffusionScheduler(
        num_atom_types=args.num_atom_types,
        num_bond_types=args.num_bond_types,
        T_full=args.T_full,
        T1=args.T1,
        T2=args.T2,
        s=args.s,
        device=args.device
    )
    train(args, logger, train_loader, val_loader, generator_network, scheduler, amp_autocast, loss_scaler, train_sampler)

    # 分子生成接口

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Molecule Generation Main BFS', parents=[get_args_parser()])
    args = parser.parse_args()  
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # 确保用于保存args的目录存在
    if args.args_save_dir:
        Path(args.args_save_dir).mkdir(parents=True, exist_ok=True)
    main(args)