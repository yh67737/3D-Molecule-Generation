import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from e3nn.o3 import Irreps

# 复用DiT时间步嵌入模块
class TimestepEmbedder(nn.Module):
    """
    将标量时间步嵌入为向量表示.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """创建正弦时间步嵌入."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


# 等变层归一化模块
class AdaEquiLayerNorm(nn.Module):
    """
    使用时间步嵌入来动态生成归一化的scale和shift参数，
    融合了Equiformer和DiT的思想.
    """
    def __init__(self, irreps, time_embedding_dim, eps=1e-5, normalization='component'):
        """
        Args:
            irreps (str or o3.Irreps): 输入特征的Irreps定义.
            time_embedding_dim (int): 时间步嵌入向量维度.
            eps (float): 防止除以零的小常数.
            normalization (str): 'norm' 或 'component' 
        """
        super().__init__()
        self.irreps = Irreps(irreps)
        self.eps = eps
        self.normalization = normalization
        
        # 计算动态生成的参数数量  
        # scale: 每个irrep特征类型一个
        # shift: 每个标量通道一个/等于标量的维度
        self.num_features = self.irreps.num_irreps
        # self.irreps是一个e3nn.o3.Irreps对象，可以将其视作一个列表，列表中每一项是一个元组 (mul, ir)
        # mul (multiplicity): 代表这个类型的特征有多少个通道，例如在'128x0e'中，mul=128
        # ir (irrep): 不可约表示的类型本身，有.l（阶数）和.p（宇称）两个属性
        self.num_scalar_channels = sum(mul for mul, ir in self.irreps if ir.l == 0 and ir.p == 1)
        
        # 时间步嵌入和调制网络
        self.t_embedder = TimestepEmbedder(time_embedding_dim)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embedding_dim, self.num_features + self.num_scalar_channels, bias=True)
        )
        
        # Zero-out初始化策略
        nn.init.constant_(self.modulation[-1].weight, 0)
        nn.init.constant_(self.modulation[-1].bias, 0)

    def __repr__(self): 
        return f"{self.__class__.__name__}({self.irreps}, eps={self.eps}, time_dim={self.t_embedder.mlp[0].in_features})"

    def forward(self, node_input, t, batch, **kwargs): 
        """
        Args:
            考虑批处理
            node_input (Tensor): 输入的Irreps特征张量,shape -> [num_nodes, feature_dims];当前批次(batch)中所有图(分子)的节点(原子)总数
            t (Tensor): 扩散模型的时间步.
            batch (Tensor): 每个节点所属的图的索引.
        """
        # 处理一个批次的N个图，每个图有一个扩散时间点 
        time_emb = self.t_embedder(t) # (N,) -> (N, D)
        
        # batch确定了每个节点所属图的索引
        # 通过高级所索引给每个节点分配对应的时间
        c = time_emb[batch] # (N, D) -> (num_nodes, D)
        
        # 生成自适应层归一化参数
        # (num_nodes, D) -> (num_nodes, num_features + num_scalar_channels)
        mod_params = self.modulation(c)
        dynamic_scale, dynamic_shift = mod_params.split([self.num_features, self.num_scalar_channels], dim=-1)

        # 执行归一化和自适应调制
        dim = node_input.shape[-1]
        fields = []
        ix = 0 # for slicing node_input
        iw = 0 # for slicing dynamic_scale
        ib = 0 # for slicing dynamic_shift

        for mul, ir in self.irreps:
            d = ir.dim
            field = node_input.narrow(1, ix, mul * d)
            ix += mul * d
            
            # 标量特征(L=0, '0e')
            if ir.l == 0 and ir.p == 1:
                # 手动实现F.layer_norm，以便使用动态的shift和scale
                # 归一化(无仿射变换)
                mean = torch.mean(field, dim=1, keepdim=True)
                var = torch.var(field, dim=1, keepdim=True, unbiased=False)
                normalized_field = (field - mean) * torch.rsqrt(var + self.eps)
                
                # 应用动态调制
                # scale: (num_nodes, 1) -> (num_nodes, mul)
                # shift: (num_nodes, mul)
                # 残差式缩放：网络直接学习和输出的dynamic_scale，并不是最终的缩放参数γ本身，而是最终缩放参数γ相对于1的残差或偏移量
                current_scale = 1 + dynamic_scale[:, iw].unsqueeze(-1)
                current_shift = dynamic_shift[:, ib:(ib + mul)]
                modulated_field = normalized_field * current_scale + current_shift
                
                fields.append(modulated_field)
                iw += 1 # scale 是 per-feature
                ib += mul # shift 是 per-channel
                continue

            # 非标量特征 (L>0)
            field = field.reshape(-1, mul, d)
            
            # 计算RMS进行归一化
            if self.normalization == 'norm':
                field_norm = field.pow(2).sum(-1)
            elif self.normalization == 'component':
                field_norm = field.pow(2).mean(-1)  
            else:
                raise ValueError("Invalid normalization option {}".format(self.normalization))
            field_norm = torch.mean(field_norm, dim=1, keepdim=True)
            field_norm = torch.rsqrt(field_norm + self.eps)
            normalized_field = field * field_norm.unsqueeze(-1)
            
            # 应用动态调制 (只有scale, 没有shift)
            # scale: (num_nodes, 1) -> (num_nodes, 1, 1)
            current_scale = (1 + dynamic_scale[:, iw]).unsqueeze(-1).unsqueeze(-1)
            modulated_field = normalized_field * current_scale
            
            fields.append(modulated_field.reshape(-1, mul * d))
            iw += 1 # scale 是 per-feature

        assert ix == dim
        output = torch.cat(fields, dim=-1)
        return output # shape:[num_nodes, feature_dims],与node_input一致