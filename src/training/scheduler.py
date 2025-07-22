import torch
import math
import args # 修改

class HierarchicalDiffusionScheduler:
    """
    一个分层扩散调度器，用于处理两种耦合的噪声过程。
    现在它也负责计算和存储离散特征的累积转移矩阵。
    """
    def __init__(
        self,
        num_atom_types: int,
        num_bond_types: int,
        T_full: int = args.T_full,
        T1: int = args.T1,
        T2: int = args.T2,
        s: float = args.s,
        device: str = args.device
    ):
        """
        Args:
            num_atom_types (int): 原子类型的类别总数 (例如 6)。
            num_bond_types (int): 边类型的类别总数 (例如 5)。
            T_full (int): 'alpha' 调度的完整长度。
            T1 (int):     'alpha' 调度实际使用的步数。
            T2 (int):     'gamma'/'delta' 调度的步数。
            s (float):    Cosine schedule 的偏移量。
            device (str): 计算所在的设备。
        """
        self.num_atom_types = num_atom_types
        self.num_bond_types = num_bond_types
        self.T_full = T_full
        self.T1 = T1
        self.T2 = T2
        self.s = s
        self.device = torch.device(device)

        # --- 1. 计算坐标加噪的 alpha_bar 和 delta_bar ---
        t_alpha = torch.linspace(0, T_full, T_full + 1, device=self.device)
        self.alpha_bars_full = self._cosine_schedule(t_alpha, T_full, s)
        self.alpha_bars = self.alpha_bars_full[:T1 + 1]

        t_gamma = torch.linspace(0, T2, T2 + 1, device=self.device)
        self.gamma_bars = self._cosine_schedule(t_gamma, T2, s)

        alpha_bar_at_T1 = self.alpha_bars_full[T1]
        self.delta_bars = alpha_bar_at_T1 * self.gamma_bars

        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)
        self.sqrt_delta_bars = torch.sqrt(self.delta_bars)
        self.sqrt_one_minus_delta_bars = torch.sqrt(1.0 - self.delta_bars)

        # 完整调度
        self.alphas_full = self.alpha_bars_full / torch.cat([torch.tensor([1.0], device=self.device), self.alpha_bars_full[:-1]])
        self.betas_full = 1.0 - self.alphas_full
        
        # T1 调度
        self.alphas = self.alpha_bars / torch.cat([torch.tensor([1.0], device=self.device), self.alpha_bars[:-1]])
        self.betas = 1.0 - self.alphas
        
        # T2/gamma 调度
        self.gammas = self.gamma_bars / torch.cat([torch.tensor([1.0], device=self.device), self.gamma_bars[:-1]])
        self.gamma_betas = 1.0 - self.gammas # 这是公式中的 (1-gamma_t)

        # 预计算 sigma_t^2 的两个部分
        # 用于 alpha 调度
        self.posterior_variance_alpha = self.betas 
        # 用于 delta 调度 
        self.posterior_variance_delta = self.gamma_betas
        
        # --- 2. 计算离散特征加噪的累积转移矩阵 Q_bar ---
        self.Q_bar_alpha_a = self._calculate_absorbing_q_bar(self.alpha_bars, self.num_atom_types)
        self.Q_bar_gamma_a = self._calculate_absorbing_q_bar(self.gamma_bars, self.num_atom_types)
        
        self.Q_bar_alpha_b = self._calculate_absorbing_q_bar(self.alpha_bars, self.num_bond_types)
        self.Q_bar_gamma_b = self._calculate_absorbing_q_bar(self.gamma_bars, self.num_bond_types)


    def _cosine_schedule(self, t, T, s):
        steps = t / T
        f_t = torch.cos(((steps + s) / (1 + s)) * (math.pi / 2)) ** 2
        f_0 = torch.cos(torch.tensor(s / (1 + s) * (math.pi / 2), device=self.device)) ** 2
        return f_t / f_0

    def _calculate_absorbing_q_bar(self, bar_schedule, num_classes):
        """
        根据给定的 bar_schedule (alpha_bar或gamma_bar) 计算吸收态的累积转移矩阵 Q_bar。
        """
        T = len(bar_schedule) - 1
        bar_t = bar_schedule.view(T + 1, 1, 1) # Reshape for broadcasting [T+1, 1, 1]

        # 创建一个单位矩阵作为基础
        identity = torch.eye(num_classes, device=self.device).view(1, num_classes, num_classes)
        
        # Q_bar_t = alpha_bar_t * I + (1 - alpha_bar_t) * P_m
        # 其中 P_m 是一个投影到吸收态的矩阵
        # 这里我们直接构造
        Q_bar = torch.zeros(T + 1, num_classes, num_classes, device=self.device)
        
        # 设置对角线: Q_bar_t[i,i] = bar_t for non-absorbing
        diag_indices = torch.arange(num_classes - 1, device=self.device)
        Q_bar[:, diag_indices, diag_indices] = bar_t.squeeze(-1)

        # 设置到吸收态的转移: Q_bar_t[i, m] = 1 - bar_t for non-absorbing
        absorbing_idx = num_classes - 1
        Q_bar[:, diag_indices, absorbing_idx] = 1.0 - bar_t.squeeze(-1)
        
        # 设置吸收态行: Q_bar_t[m, m] = 1
        Q_bar[:, absorbing_idx, absorbing_idx] = 1.0

        return Q_bar

    def _get_vals_at_t(self, values, t):
        batch_size = t.shape[0]
        out = values[t.long()]
        return out.reshape(batch_size, 1)

    # q_sample 和 get_predicted_noise_from_r0 方法
    def q_sample(self, r_0, t, noise, schedule_type='alpha'):
        if schedule_type == 'alpha':
            sqrt_bar = self._get_vals_at_t(self.sqrt_alpha_bars, t)
            sqrt_one_minus_bar = self._get_vals_at_t(self.sqrt_one_minus_alpha_bars, t)
        elif schedule_type == 'delta':
            sqrt_bar = self._get_vals_at_t(self.sqrt_delta_bars, t)
            sqrt_one_minus_bar = self._get_vals_at_t(self.sqrt_one_minus_delta_bars, t)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        r_t = sqrt_bar * r_0 + sqrt_one_minus_bar * noise
        return r_t

    def get_predicted_noise_from_r0(self, r_t, t, predicted_r0, schedule_type='alpha'):
        if schedule_type == 'alpha':
            sqrt_bar_t = self._get_vals_at_t(self.sqrt_alpha_bars, t)
            sqrt_one_minus_bar_t = self._get_vals_at_t(self.sqrt_one_minus_alpha_bars, t)
        elif schedule_type == 'delta':
            sqrt_bar_t = self._get_vals_at_t(self.sqrt_delta_bars, t)
            sqrt_one_minus_bar_t = self._get_vals_at_t(self.sqrt_one_minus_delta_bars, t)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        predicted_noise = (r_t - sqrt_bar_t * predicted_r0) / torch.clamp(sqrt_one_minus_bar_t, min=1e-8)
        return predicted_noise