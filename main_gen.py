# 文件: main_generate_ddp.py

import os
import math
import torch
import random
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

from src.utils.logger import FileLogger
from generate_qm9 import generate_molecule
from src.models.EDiT_network.e_dit_network import E_DiT_Network
from src.models.ring_network.train_loop_network import RingPredictor
from src.training.scheduler import HierarchicalDiffusionScheduler

def load_args(path):
    args = torch.load(path)  # 使用 PyTorch 的加载方式
    print(f"[✓] 参数加载成功: {path}")
    return args


def setup_environment(args):
    # 分布式初始化
    if args.distributed:
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()
        args.device = f"cuda:{args.local_rank}"
    else:
        args.rank = 0
        args.world_size = 1
        args.device = torch.device(args.device)

    # 是否是主进程
    is_main_process = (not args.distributed) or (args.rank == 0)

    # if is_main_process:
    #     timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    #     args.run_dir = os.path.join(args.output_dir, timestamp)

    #     sub_dirs = {
    #         "generated_pyg": os.path.join(args.run_dir, "generated_pyg"),
    #         "generated_images": os.path.join(args.run_dir, "generated_images"),
    #         "results": os.path.join(args.run_dir, "results")
    #     }

    #     for path in sub_dirs.values():
    #         Path(path).mkdir(parents=True, exist_ok=True)

    #     broadcast_objects = [args.run_dir, sub_dirs]
    # else:
    #     broadcast_objects = [None, None]

    # # 广播路径
    # if args.distributed:
    #     torch.distributed.broadcast_object_list(broadcast_objects, src=0)

    # args.run_dir, sub_dirs_synced = broadcast_objects
    # for key, val in sub_dirs_synced.items():
    #     setattr(args, f"{key}_dir", val)

    if args.distributed:
        torch.distributed.barrier()

    # 日志
    logger = FileLogger(
        output_dir=args.generated_pyg_dir,
        is_master=is_main_process,
        is_rank0=is_main_process
    )

    logger.info("Logger initialized. Distributed config synchronized.")
    logger.info(f"All outputs for this run will be saved to: {args.run_dir}")
    logger.info(f"Master process (rank 0) configuration:\n{args}")

    # 设置随机种子
    torch.manual_seed(args.seed + args.rank)
    np.random.seed(args.seed + args.rank)
    random.seed(args.seed + args.rank)

    if torch.cuda.is_available() and "cuda" in str(args.device):
        torch.backends.cudnn.benchmark = True
        logger.info(f"CUDA device name: {torch.cuda.get_device_name(args.local_rank)}")

    logger.info("Environment initialization complete.")
    return logger


def main(args):
    logger = setup_environment(args)

    # 再次设定随机种子
    torch.manual_seed(args.seed + args.rank)
    np.random.seed(args.seed + args.rank)
    random.seed(args.seed + args.rank)

    # 模型实例化与加载
    model = E_DiT_Network(args).to(args.device)
    # 1. 先加载整个 checkpoint 文件
    checkpoint = torch.load(args.model_ckpt, map_location=args.device)
    
    # 2. 从 checkpoint 中提取出模型的状态字典
    model_state_dict = checkpoint['model_state_dict']
    
    # 3. 加载模型的状态字典
    model.load_state_dict(model_state_dict)
    
    model.eval()

    p_model = RingPredictor(
        node_feature_dim=args.num_atom_types,
        edge_feature_dim=args.num_bond_types,
        num_ring_classes=1
    ).to(args.device)
    p_model.load_state_dict(torch.load(args.ring_guide_ckpt, map_location=args.device))
    p_model.eval()

    scheduler = HierarchicalDiffusionScheduler(
        num_atom_types=args.num_atom_types,
        num_bond_types=args.num_bond_types,
        T_full=args.T_full,
        T1=args.T1,
        T2=args.T2,
        s=args.s,
        device=args.device
    )

    # 每个进程独立生成 num_generate // world_size 个分子
    num_per_rank = math.ceil(args.num_generate / args.world_size)
    logger.info(f"[Rank {args.rank}] Generating {num_per_rank} molecules...")

    molecules = []
    for _ in tqdm(range(num_per_rank), desc=f"[Rank {args.rank}] Generating"):
        mol = generate_molecule(model, p_model, scheduler, args)
        molecules.append(mol)

    # 1. 从模型路径中提取文件名 (例如: 'checkpoint_epoch_10.pth')
    model_filename = os.path.basename(args.model_ckpt)
    
    # 2. 去掉文件扩展名 .pth，得到更有用的部分 (例如: 'checkpoint_epoch_10')
    model_name_stem = os.path.splitext(model_filename)[0]
    
    # 3. 构建新的、包含模型信息的 pkl 文件名
    #    格式：generated_molecules_from_checkpoint_epoch_10.pkl
    output_pkl_filename = f"generated_molecules_from_{model_name_stem}.pkl"

    # 分布式收集所有分子，仅主进程保存
    if args.distributed:
        gathered = [None for _ in range(args.world_size)]
        torch.distributed.all_gather_object(gathered, molecules)

        if args.rank == 0:
            all_molecules = []
            for mol_list in gathered:
                all_molecules.extend(mol_list)

            # ✅ 截断
            all_molecules = all_molecules[:args.num_generate]
            
            save_path = os.path.join(args.generated_pyg_dir, output_pkl_filename)
            with open(save_path, 'wb') as f:
                pickle.dump(all_molecules, f)
            logger.info(f"✅ 所有进程生成完成，共 {len(all_molecules)} 个分子，保存至: {save_path}")
    else:
        save_path = os.path.join(args.generated_pyg_dir, output_pkl_filename)
        with open(save_path, 'wb') as f:
            pickle.dump(molecules, f)
        logger.info(f"✅ 非分布式，共生成 {len(molecules)} 个分子，保存至: {save_path}")


# if __name__ == '__main__':
#     args = load_args('./saved_args/args_2025-08-16_12-43-43.pt')  # 修改

#     # if args.output_dir:
#     #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)

#     main(args)

if __name__ == '__main__':
    # 1. 创建一个命令行解析器
    parser = argparse.ArgumentParser()
    
    # 2. 添加你想从命令行控制的参数
    #    - 首先，让参数文件的路径本身变成一个参数，这样更灵活！
    parser.add_argument('--args_path', type=str, 
                        default='./saved_args/args_2025-08-16_12-43-43.pt', 
                        help='Path to the saved arguments .pt file')
    
    #    - 其次，添加你想覆盖的参数。设置 default=None 是关键。
    parser.add_argument('--model_ckpt', type=str, default=None, 
                        help='Override the model checkpoint path from the args file.')
    
    #    - 你可以按需添加更多想覆盖的参数
    #    parser.add_argument('--num_generate', type=int, default=None, help='Override number of molecules to generate.')

    # 3. 解析来自命令行的参数
    cli_args = parser.parse_args()

    # 4. 先从文件加载基础参数
    args = load_args(cli_args.args_path)
    print(f"原始模型路径 (from file): {args.model_ckpt}")

    # 5. 用命令行参数覆盖文件中的参数（如果提供了的话）
    if cli_args.model_ckpt is not None:
        args.model_ckpt = cli_args.model_ckpt
        print(f"✅ 模型路径已被命令行覆盖为: {args.model_ckpt}")
    
    # if cli_args.num_generate is not None:
    #     args.num_generate = cli_args.num_generate
    #     print(f"✅ 生成数量已被命令行覆盖为: {args.num_generate}")

    # 6. 使用最终确定的参数运行主函数
    main(args)

# torchrun --nproc_per_node=4 main_generate.py --distributed