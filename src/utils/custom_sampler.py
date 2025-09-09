import torch
import torch.distributed as dist
from torch.utils.data import Sampler
import math

class DistributedFixedLengthRandomSampler(Sampler):
    """
    一个分布式采样器，它从数据集中有放回地随机采样固定数量的元素。

    当一个 "epoch" 被定义为固定的步数/样本数而不是完整遍历一次数据集时，
    这个采样器非常有用。

    参数:
        dataset (Dataset): 需要采样的数据集。
        num_samples_per_epoch (int): 在所有副本（GPU）上，一个 epoch 总共需要抽取的样本数量。
        num_replicas (int, optional): 参与分布式训练的进程数。
            默认情况下，从当前的分布式组中获取 world_size。
        rank (int, optional): 当前进程在 num_replicas 中的排名。
            默认情况下，从当前的分布式组中获取 rank。
        seed (int, optional): 用于打乱采样器的随机种子。默认为 0。
    """
    def __init__(self, dataset, num_samples_per_epoch, num_replicas=None, rank=None, seed=0):
        if num_replicas is None:
            if not dist.is_available() or not dist.is_initialized():
                num_replicas = 1
            else:
                num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available() or not dist.is_initialized():
                rank = 0
            else:
                rank = dist.get_rank()
            
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed
        
        # 一个 epoch 内，在所有 GPU 上生成的样本总数
        self.total_num_samples = num_samples_per_epoch
        
        # 每个副本（GPU）的样本数
        # 我们使用向上取整，以确保生成的样本总数至少达到 total_num_samples
        self.num_samples_per_replica = int(math.ceil(self.total_num_samples / self.num_replicas))
        
        # 为了能被整除，总大小可能比请求的略大
        self.total_size = self.num_samples_per_replica * self.num_replicas

    def __iter__(self):
        # 1. 为所有 GPU 生成完整的随机索引列表
        #    关键在于，每个进程必须生成完全相同的列表。
        #    使用 epoch 作为种子可以确保这一点。
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        # 有放回地采样
        global_indices = torch.randint(high=len(self.dataset), 
                                       size=(self.total_size,), 
                                       dtype=torch.int64, 
                                       generator=g)

        # 2. 每个进程获取自己对应的切片
        #    这是分布式采样的核心逻辑。
        indices_for_this_rank = global_indices[self.rank : self.total_size : self.num_replicas]
        
        # 由于向上取整，我们可能为每个副本生成了稍多的样本，
        # 所以这里将其裁剪回所需的确切数量。
        return iter(indices_for_this_rank.tolist()[:self.num_samples_per_replica])

    def __len__(self):
        return self.num_samples_per_replica

    def set_epoch(self, epoch):
        r"""
        为这个采样器设置 epoch。这可以确保每个 epoch 都有不同的随机排序。
        否则，采样器在每个 epoch 都会产生相同的序列。
        """
        self.epoch = epoch