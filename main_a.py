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
from src.utils import logger
from src.utils import utils
from src.utils.logger import FileLogger # Equiformer提供的通用模块
from timm.utils import NativeScaler # timm是一个强大的深度学习第三方库，需要安装
# distributed training
from src.models.sorting_network.sorting_network import SortingNetwork
from src.models.EDiT_network.e_dit_network import E_DiT_Network
from src.data.preprocess_bfs import generate_single_centered_subgraph
from src.training.train_A import train
from src.training.scheduler import HierarchicalDiffusionScheduler

class PairedMoleculeDataset(Dataset):
    """
    一个优化的、处理成对分子图数据的Dataset。
    它在初始化时接收文件路径，并自行管理数据的加载，实现“延迟加载”结构。
    """
    def __init__(self, data_path_1: str, data_path_2: str, indices: list, logger=None):
        """
        Args:
            data_path_1 (str): 数据集1的.pt文件路径。
            data_path_2 (str): 数据集2的.pt文件路径。
            indices (list or np.ndarray): 指定使用数据集中哪些样本的索引。
            logger: 日志记录器实例，用于打印加载信息。
        """
        super().__init__()
        self.data_path_1 = data_path_1
        self.data_path_2 = data_path_2
        self.indices = indices
        self.logger = logger

        # 将数据初始化为None，实现延迟加载
        self._dataset1 = None
        self._dataset2 = None

    @property
    def dataset1(self):
        # 当第一次访问 self.dataset1 时，才真正从硬盘加载数据
        if self._dataset1 is None:
            if self.logger:
                self.logger.info(f"First access: Lazily loading dataset 1 from: {self.data_path_1}")
            self._dataset1 = torch.load(self.data_path_1, weights_only=False)
        return self._dataset1

    @property
    def dataset2(self):
        # 当第一次访问 self.dataset2 时，才真正从硬盘加载数据
        if self._dataset2 is None:
            if self.logger:
                self.logger.info(f"First access: Lazily loading dataset 2 from: {self.data_path_2}")
            self._dataset2 = torch.load(self.data_path_2, weights_only=False)
        return self._dataset2

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # 检查数据集长度是否一致
        assert len(self.dataset1) == len(self.dataset2), \
            "Paired datasets must have the same length."
            
        real_idx = self.indices[idx]
        data1 = self.dataset1[real_idx]
        data2 = self.dataset2[real_idx]
        return data1, data2  # 逗号分隔，返回元组类型

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

'''
def compute_dataset_stats(data_loader, logger, print_freq=100):
    """
    计算并记录数据集中图的平均节点数、边数和度数。
    此函数已修改为直接使用 PyG Data 对象中的属性，而不是重新计算边。
    """
    logger.info('\n--- Calculating Dataset Statistics ---')
        
    avg_node = AverageMeter()
    avg_edge = AverageMeter()
    avg_degree = AverageMeter()
    
    # data_loader 产出的是 (data1, data2) 对，我们只在 data1 上计算统计量
    for step, (data1, _) in enumerate(data_loader):
        # 注意：DataLoader 会自动将单个样本打包成 Batch 对象
        # 我们需要按图来分解 Batch
        from torch_geometric.data import Batch
        if isinstance(data1, Batch):
            graph_list = data1.to_data_list()
        else: # 如果 batch_size=1, 可能不是 Batch 对象
            graph_list = [data1]

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
'''
@torch.no_grad() # 确保在统计过程中不计算梯度
def compute_dataset_stats(
    data_loader, 
    sorting_network, 
    subgraph_generator_fn,
    device, 
    logger,
    num_samples_per_graph=1 # 为减少计算量，每个图只采样一个片段进行统计
):
    """
    计算并记录数据集中 *分子片段* 的平均节点数、边数和度数。
    此函数完整模拟了数据预处理流程，以确保统计量与模型输入匹配。
    """
    logger.info('\n--- Calculating Dataset Fragment Statistics ---')
    
    sorting_network.eval() # 确保模型在评估模式
    
    avg_node = AverageMeter()
    avg_edge = AverageMeter()
    avg_degree = AverageMeter()
    
    # data_loader 产出的是 (data1, data2) 对
    for step, (data1, data2) in enumerate(data_loader):
        data1 = data1.to(device)
        data2 = data2.to(device)

        # 1. 使用排序网络生成原子顺序
        # sorting_network的forward方法返回局部索引的顺序列表和对数概率
        local_orders, _ = sorting_network(data1)

        # 将批处理的 data2 解包成单个图的列表
        graph_list_2 = data2.to_data_list()
        
        # 2. 对批次中的每个图进行切分和统计
        for i in range(data1.num_graphs):
            graph2 = graph_list_2[i]
            order = local_orders[i]
            
            # 3. 使用切割模块生成一个分子片段
            # 注意：这里的order是局部索引，可以直接用于切割对应的 graph2
            fragment = subgraph_generator_fn(graph2, order)
            
            # 4. 在生成的片段上计算统计量
            num_nodes = fragment.num_nodes
            num_edges = fragment.num_edges
            
            if num_edges > 0:
                degree = torch_geometric.utils.degree(fragment.edge_index[0], num_nodes=num_nodes).mean()
            else:
                degree = torch.tensor(0.0)

            avg_node.update(num_nodes)
            avg_edge.update(num_edges)
            avg_degree.update(degree.item())
            
    log_str = (f"Average Fragment #Nodes: {avg_node.avg:.2f}, "
               f"Average Fragment #Edges: {avg_edge.avg:.2f}, "
               f"Average Fragment Degree: {avg_degree.avg:.2f}")
    logger.info(log_str)
    logger.info('--- Statistics Calculation Complete ---\n')
    
    # 返回计算出的平均度数，以便在模型初始化时使用
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

    # 训练流程
    parser.add_argument('--epochs', type=int, default=500, help='Total number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the Adam optimizer(model)')
    parser.add_argument('--s_learning_rate', type=float, default=1e-4, help='Learning rate for the Adam optimizer(s_model)')
    parser.add_argument('--lr_min_factor', type=float, default=0.01, 
                    help="学习率下限因子，最终学习率 = lr_min_factor × 初始学习率")
    # 损失权重
    parser.add_argument('--w_a', type=float, default=1.0, help='Weight for atom type loss')
    parser.add_argument('--w_r', type=float, default=1.0, help='Weight for coordinate loss')
    parser.add_argument('--w_b', type=float, default=1.0, help='Weight for bond type loss')
    parser.add_argument('--lambda_aux', type=float, default=0.01, help='Weight for auxiliary loss term in D3PM')
    # 验证配置(暂未使用)
    parser.add_argument('--batch_ratio_val', type=int, default=10, help='Ratio to determine number of validation samples (num_val_samples = batch_size * batch_ratio)')
    # 保存路径
    # parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Path to save the model')
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
    parser.add_argument('--max_atoms', type=int, default=50, help="生成分子的最大原子数。这是主生成循环的上限。")
    parser.add_argument('--min_atoms', type=int, default=3, help="在因新原子未连接而停止生成之前，所要求的最小原子数。")
    parser.add_argument('--num_generate', type=int, default=100, help='要生成的分子数量')
    parser.add_argument('--model_ckpt', type=str, default='./output/model.pt', help='生成模型路径') # 修改
    
    # --- 数据集参数 ---
    parser.add_argument('--data_path_1', type=str, default='./prepared_data/small.pt', help='Path to dataset 1 (for SortingNetwork).')
    parser.add_argument('--data_path_2', type=str, default='./prepared_data/small_fully_connected.pt', help='Path to dataset 2 (for E-DiT).')
    parser.add_argument('--data_split_path', type=str, default='./data_splits', help='Directory to save/load data split indices.')
    parser.add_argument('--val_split_percentage', type=float, default=0.05, help='Percentage of data to use for validation.')
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--accumulation_steps', type=int, default=8)

    # --- 模型参数: 排序网络 (SortingNetwork) ---
    parser.add_argument('--sorting_num_atom_features', type=int, default=6,
                    help='Input dimension of atom features for the Sorting Network.')
    parser.add_argument('--sorting_num_bond_features', type=int, default=5,
                    help='Input dimension of bond features for the Sorting Network.')
    parser.add_argument('--sorting_hidden_dim', type=int, default=128,
                    help='Hidden dimension within the Sorting Network.')
    parser.add_argument('--sorting_gnn_layers', type=int, default=4,
                    help='Number of EGNN layers in the Sorting Network core.')
    parser.add_argument('--max_nodes', type=int, default=500,
                    help='Maximum number of nodes for Positional Encoding, shared across models.')

    # E-DiT参数
    # --- 模型架构参数 (Model Architecture) ---
    g_arch = parser.add_argument_group('Architecture')
    g_arch.add_argument('--num_blocks', type=int, default=4, help='Number of E-DiT blocks.')
    g_arch.add_argument('--num_heads', type=int, default=4, help='Number of attention heads.')
    g_arch.add_argument('--norm_layer', type=str, default='layer', help='Type of normalization layer (e.g., "layer").')
    g_arch.add_argument('--time_embed_dim', type=int, default=128, help='Dimension of timestep embedding.')

    # --- Irreps 参数 (Irreps Definitions) ---
    g_irreps = parser.add_argument_group('Irreps')
    g_irreps.add_argument('--irreps_node_hidden', type=str, default='64x0e+32x1o+16x2e',
                          help='Hidden node feature irreps.')
    g_irreps.add_argument('--irreps_edge', type=str, default='64x0e+32x1o+16x2e', help='Hidden edge feature irreps.')
    g_irreps.add_argument('--irreps_node_attr', type=str, default='6x0e', help='Node attribute (atom type) irreps.')
    g_irreps.add_argument('--irreps_edge_attr_type', type=str, default='5x0e',
                          help='Edge attribute (bond type) irreps.')
    g_irreps.add_argument('--irreps_sh', type=str, default='1x0e+1x1e+1x2e', help='Spherical harmonics irreps.')
    g_irreps.add_argument('--irreps_head', type=str, default='32x0e+16x1o+8x2e', help='Single attention head irreps.')
    g_irreps.add_argument('--irreps_mlp_mid', type=str, default='128x0e+64x1o+32x2e',
                          help='Irreps for the middle layer of FFN.')
    g_irreps.add_argument('--irreps_pre_attn', type=str, default=None,
                          help='Optional irreps for pre-attention linear layer.')

    # --- 嵌入层参数 (Embedding Layers) ---
    g_embed = parser.add_argument_group('Embeddings')
    g_embed.add_argument('--num_atom_types', type=int, default=6, help='Number of atom types for embedding.')
    g_embed.add_argument('--num_bond_types', type=int, default=5, help='Number of bond types for embedding.')
    g_embed.add_argument('--node_embedding_hidden_dim', type=int, default=64,
                         help='Hidden dimension in node embedding MLP.')
    g_embed.add_argument('--bond_embedding_dim', type=int, default=64, help='Hidden dimension in edge embedding MLP.')
    g_embed.add_argument('--edge_update_hidden_dim', type=int, default=64,
                         help='Hidden dimension in EdgeUpdateNetwork MLP.')
    g_embed.add_argument('--num_rbf', type=int, default=64, help='Number of radial basis functions.')
    g_embed.add_argument('--rbf_cutoff', type=float, default=5.0, help='Cutoff radius for RBF.')
    g_embed.add_argument('--fc_neurons', type=int, nargs='+', default=[64, 64],
                         help='List of hidden layer sizes for FC network in attention.')
    g_embed.add_argument('--avg_degree', type=float, default=10.0, help='Average degree of nodes in the dataset.')

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
    parser.add_argument('--ring_guide_ckpt', type=str, default='./ring_network/ring_predictor_epoch_40.pt', help='Path to pre-trained ring guidance network checkpoint.')

    parser.add_argument('--optimizer', type=str, default='adamw', help='Optimizer (e.g., adamw, sgd).')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay for optimizer.')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Number of warmup epochs for LR scheduler.')
    
    # AMP
    parser.add_argument('--amp', action='store_false', help='Enable automatic mixed precision training.')
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
    if is_main_process: # 防止多个进程竞争IO操作
        # 获取当前运行的时间戳
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # 将run_dir作为一个新的属性，run_dir = output_dir + timestamp
        args.run_dir = os.path.join(args.output_dir, timestamp) 
        print(f"All outputs for this run will be saved to: {args.run_dir}")

        # 定义所有需要创建的的子文件夹名称
        sub_dirs = {
            "checkpoints": os.path.join(args.run_dir, "checkpoints"), # 训练权重文件
            "generated_pyg": os.path.join(args.run_dir, "generated_pyg"), # 新生成的PyG数据
            "generated_images": os.path.join(args.run_dir, "generated_images"), # PyG数据画图得到的分子图
            "tb_logs": os.path.join(args.run_dir, "tb_logs"), # TensorBoard 日志
            "results": os.path.join(args.run_dir, "results") # 评估结果
        }
        
        # 循环创建所有文件夹
        for path in sub_dirs.values():
            Path(path).mkdir(parents=True, exist_ok=True)

        # 保存配置(args)对象
        # 从 run_dir中提取时间戳，确保文件名与本次运行完全对应
        timestamp_from_path = os.path.basename(args.run_dir)
        
        # 构建保存文件的完整路径
        args_save_filename = f"args_{timestamp_from_path}.pt"
        args_save_path = os.path.join(args.args_save_dir, args_save_filename)
        
        # 保存args对象
        torch.save(args, args_save_path)
        print(f"Configuration for this run has been saved to: {args_save_path}")
        
        # 准备要广播的对象列表:如果不做处理，只有主进程的args对象增加了新创建的路径，其他进程的args对象则没有，后续调用时就会出错
        # 包含动态生成的run_dir所有子目录路径字典
        objects_to_broadcast = [args.run_dir, sub_dirs]
    
    else:
        # 其他进程准备好相同结构的空列表接收数据
        objects_to_broadcast = [None, None]
    
    # 执行广播操作(只有在分布式模式下需要广播)
    if args.distributed:
        # broadcast_object_list会将src=0(主进程)的列表内容发送给所有其他进程
        torch.distributed.broadcast_object_list(objects_to_broadcast, src=0)
    
    # 所有进程用广播来的数据更新自己的配置
    # 主进程已经有这些数据了，但为了代码统一，可以都执行
    run_dir_synced, sub_dirs_synced = objects_to_broadcast
    
    args.run_dir = run_dir_synced
    for key, path in sub_dirs_synced.items():
        setattr(args, f"{key}_dir", path)

    # 设置屏障:确保所有进程都执行完前面的所有步骤后(参数更新)一起进入下一个阶段，防止时序问题
    if args.distributed:
        torch.distributed.barrier()

    # 初始化Logger
    # 日志只会由主进程(rank 0)创建和写入
    logger = FileLogger(
    output_dir=args.run_dir, 
    is_master=is_main_process, # is_master 控制是否写入文件
    is_rank0=is_main_process   # is_rank0 控制是否创建真实logger
    )

    # 使用logger记录信息
    logger.info("Logger initialized. Distributed config synchronized.")
    logger.info(f"All outputs for this run will be saved to: {args.run_dir}")
    logger.info(f"Master process (rank 0) configuration:\n{args}") # 只记录主进程的配置作为代表
    logger.info(f"args_save_path:{args_save_path}")
    # 如果想确认每个进程都已启动，可以像下面这样写，但通常没必要
    # logger.info(f"Process {args.rank} has started.") # 这条日志只会在 rank 0 的控制台和文件中出现
    
    # 设置随机种子和模型使用的设备,为不同进程设置不同种子
    torch.manual_seed(args.seed + args.rank) 
    np.random.seed(args.seed + args.rank)
    random.seed(args.seed + args.rank)
    
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    if args.device == 'cuda' and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True # 一个性能优化选项
        logger.info(f"CUDA device name: {torch.cuda.get_device_name(args.local_rank)}")

    logger.info("Environment initialization complete.")

    ## 模块3: 数据准备(Dataset & DataLoader)
    logger.info("--- Starting Data Preparation ---")

    # 加载或创建数据划分
    split_file = os.path.join(args.data_split_path, 'split_indices.pt') # args.data_split_path/args.output_dir都用相对路径即可
    
    if os.path.exists(split_file):
        logger.info(f"Loading existing data split from: {split_file}")
        split_data = torch.load(split_file)
        train_indices, val_indices = split_data['train'], split_data['val']
    else:
        logger.info("No existing data split found. Creating a new one...")
        if is_main_process: # 仅主进程执行数据划分和保存
            # 注意：为了获取数据集总长度，仍然需要加载一次数据。
            # 这是当前“先划分再加载”策略下不可避免的开销。
            # 如果数据集过大无法一次性载入内存，就需要改变数据存储方式
            # (例如，每个样本一个文件，然后统计文件数量)。
            num_molecules = len(torch.load(args.data_path_1, weights_only=False))
            logger.info(f"Total number of molecules: {num_molecules}")

            indices = list(range(num_molecules))
            random.shuffle(indices)

            split_point = int(np.floor(args.val_split_percentage * num_molecules))
            val_indices, train_indices = indices[:split_point], indices[split_point:]
            
            logger.info(f"Splitting data: {len(train_indices)} training samples, {len(val_indices)} validation samples.")
            
            # 创建保存目录并保存
            Path(args.data_split_path).mkdir(parents=True, exist_ok=True)
            torch.save({'train': train_indices, 'val': val_indices}, split_file)
            logger.info(f"New data split saved to: {split_file}")
        
        # 在分布式环境中，确保所有进程都等待主进程完成文件写入
        if args.distributed:
            torch.distributed.barrier()
        
        # 所有进程都从保存的文件中加载索引
        split_data = torch.load(split_file, map_location='cpu')
        train_indices, val_indices = split_data['train'], split_data['val']

    # 实例化Dataset，传入文件路径而不是数据本身
    train_dataset = PairedMoleculeDataset(
        data_path_1=args.data_path_1, 
        data_path_2=args.data_path_2, 
        indices=train_indices, 
        logger=logger
    )
    val_dataset = PairedMoleculeDataset(
        data_path_1=args.data_path_1,
        data_path_2=args.data_path_2,
        indices=val_indices,
        logger=logger
    )
    logger.info(f"Train dataset instance created with {len(train_dataset)} samples.")
    logger.info(f"Validation dataset instance created with {len(val_dataset)} samples.")

    # 创建DataLoader
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None), # 只有在非分布式时才启用shuffle
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


    ## 模块4: 模型实例化与配置 

    # 再次设置随机种子，确保模型初始化的随机性绝对可控，不受数据加载过程的任何影响
    torch.manual_seed(args.seed + args.rank) 
    np.random.seed(args.seed + args.rank)
    random.seed(args.seed + args.rank)

    logger.info("--- Initializing Models ---")
    
    # 实例化各个模块
    # a. 排序网络
    sorting_network = SortingNetwork(
        num_atom_features=args.sorting_num_atom_features,
        num_bond_features=args.sorting_num_bond_features,
        hidden_dim=args.sorting_hidden_dim,
        gnn_layers=args.sorting_gnn_layers,
        max_nodes=args.max_nodes
    )
    
    # b. 生成网络
    generator_network = E_DiT_Network(args)
    
    # 将模型移动到指定设备
    sorting_network.to(device)
    generator_network.to(device)
    logger.info("Models moved to device.")

    # 模型分布式训练封装
    if args.distributed:
        sorting_network = torch.nn.parallel.DistributedDataParallel(sorting_network, device_ids=[args.local_rank])
        generator_network = torch.nn.parallel.DistributedDataParallel(generator_network, device_ids=[args.local_rank])
        logger.info("Models wrapped with DistributedDataParallel.")

    # 记录模型和参数统计
    if is_main_process:
        logger.info("--- Model Architectures ---")
        logger.info(sorting_network)
        logger.info(generator_network)
        
        total_params_sorting = sum(p.numel() for p in sorting_network.parameters() if p.requires_grad)
        total_params_generator = sum(p.numel() for p in generator_network.parameters() if p.requires_grad)
        logger.info("--- Parameter Statistics ---")
        logger.info(f"Sorting Network Trainable Params: {total_params_sorting / 1e6:.2f} M")
        logger.info(f"Generator Network Trainable Params: {total_params_generator / 1e6:.2f} M")
        logger.info("-----------------------------")

        logger.info("--- Model Initialization Complete ---")

    # 统计阶段排序网络处于未训练状态，但是这种方法是合理的，可以打破依赖关系；
    # 后续可以通过打印训练好的排序网络得到的统计数来进行验证
    if args.compute_stats:
        # 使用训练集加载器、实例化的模型和工具来计算统计数据
        avg_nodes, avg_edges, avg_degree = compute_dataset_stats(
            train_loader,
            sorting_network=sorting_network,
            subgraph_generator_fn=generate_single_centered_subgraph,
            device=device,
            logger=logger
        )
        print(f"avg_nodes = {avg_nodes}, avg_edges = {avg_edges}, avg_degree = {avg_degree}")
        # 打印完统计信息后，可以直接退出
        # 需要将计算出的avg_degree值，手动填入E-DiT网络的配置变量或模型定义中的avg_aggregate_num参数里。
        return 
    
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
    train(args, logger, train_loader, val_loader, sorting_network, generator_network, scheduler, subgraph_generator_fn=generate_single_centered_subgraph, amp_autocast=amp_autocast,
        loss_scaler=loss_scaler)

    # 分子生成接口

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Molecule Generation Main', parents=[get_args_parser()])
    args = parser.parse_args()  
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # 确保用于保存args的目录存在
    if args.args_save_dir:
        Path(args.args_save_dir).mkdir(parents=True, exist_ok=True)
    main(args)