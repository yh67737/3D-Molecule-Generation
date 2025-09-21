import torch
from torch_scatter import scatter
import torch_geometric

from e3nn import o3
from e3nn.util.jit import compile_mode

from .layer_norm import AdaEquiLayerNorm
from .fast_activation import Activation, Gate
from .drop import EquivariantDropout
from .tensor_product_rescale import FullyConnectedTensorProductRescaleSwishGate, LinearRS, FullyConnectedTensorProductRescale, TensorProductRescale, irreps2gate, sort_irreps_even_first
from .radial_func import RadialProfile
_RESCALE = True


def get_norm_layer(norm_type):
    """
    根据输入的字符串选择返回对应的等变归一化层类.
    """
    if norm_type == 'graph':
        # TODO: 后续需要实现 EquivariantGraphNorm 类
        raise NotImplementedError("EquivariantGraphNorm is not implemented yet. Please add the class definition to use this option.")
    elif norm_type == 'instance':
        # TODO: 后续需要实现 EquivariantInstanceNorm 类
        raise NotImplementedError("EquivariantInstanceNorm is not implemented yet. Please add the class definition to use this option.")
    elif norm_type == 'layer':
        return AdaEquiLayerNorm
    elif norm_type is None:
        return None
    else:
        raise ValueError(f'Norm type {norm_type} not supported.')


class SmoothLeakyReLU(torch.nn.Module):
    """
    一个平滑、可微的LeakyReLU激活函数的近似实现(非线性).
    
    与标准的LeakyReLU在x=0处有一个尖锐的“拐点”不同，
    这个函数使用sigmoid函数来构造一个在零点附近平滑过渡的曲线.
    """
    def __init__(self, negative_slope=0.2):
        """
        初始化函数.
        
        Args:
            negative_slope (float): 当输入x为负数时激活函数的斜率.
        """
        super().__init__()
        self.alpha = negative_slope
        
    
    def forward(self, x):
        """
        执行前向传播.
        
        Args:
            x (torch.Tensor): 输入的张量.

        Returns:
            torch.Tensor: 经过平滑LeakyReLU激活后的张量，形状与输入x相同.
        """
        # 激活函数的线性分量
        x1 = ((1 + self.alpha) / 2) * x
        # 激活函数的非线性分量
        x2 = ((1 - self.alpha) / 2) * x * (2 * torch.sigmoid(x) - 1)

        # 将线性和非线性部分相加，得到最终平滑的激活曲线
        return x1 + x2
    
    
    def extra_repr(self):
        """
        辅助函数，用于在打印模块信息时显示negative_slope的值.
        """
        return 'negative_slope={}'.format(self.alpha)


# 辅助函数(utility function)
def get_mul_0(irreps):
    """
    计算并返回一个Irreps对象中所有Type-0偶性标量('0e')特征的通道总数.

    这个函数是一个辅助工具，用于快速查询一个复杂的Irreps特征中包含了多少个标量通道.
    
    Args:
        irreps (e3nn.o3.Irreps): 需要被检查的Irreps对象.
                                 例如: o3.Irreps('128x0e + 64x1e + 32x0o')

    Returns:
        int: '0e'类型的标量特征的通道数总和.
             对于上面的例子会返回 128.
    """
    # 初始化计数器，用于累加标量通道的数量
    mul_0 = 0
    for mul, ir in irreps:
        if ir.l == 0 and ir.p == 1:
            mul_0 += mul
    return mul_0

def DepthwiseTensorProduct(irreps_node_input, irreps_edge_attr, irreps_node_output, 
    internal_weights=False, bias=True):
    '''
        The irreps of output is pre-determined. 
        `irreps_node_output` is used to get certain types of vectors.
    '''
    irreps_output = []
    instructions = []
    
    for i, (mul, ir_in) in enumerate(irreps_node_input):
        for j, (_, ir_edge) in enumerate(irreps_edge_attr):
            for ir_out in ir_in * ir_edge:
                if ir_out in irreps_node_output or ir_out == o3.Irrep(0, 1):
                    k = len(irreps_output)
                    irreps_output.append((mul, ir_out))
                    instructions.append((i, j, k, 'uvu', True))
        
    irreps_output = o3.Irreps(irreps_output)
    irreps_output, p, _ = sort_irreps_even_first(irreps_output) #irreps_output.sort()
    instructions = [(i_1, i_2, p[i_out], mode, train)
        for i_1, i_2, i_out, mode, train in instructions]
    tp = TensorProductRescale(irreps_node_input, irreps_edge_attr,
            irreps_output, instructions,
            internal_weights=internal_weights,
            shared_weights=internal_weights,
            bias=bias, rescale=_RESCALE)
    return tp  

class SeparableFCTP(torch.nn.Module):
    '''
        Use separable FCTP for spatial convolution.

        实现一个“可分离”的等变卷积操作，用于空间消息传递(具体执行DTP计算，然后输出结果)
        融合节点和边的信息
        
        这个模块的设计思想借鉴了深度可分离卷积（depthwise separable convolution）。
        它将一个复杂的操作分解为两个更简单、更高效的步骤：
        1. 深度张量积 (Depthwise Part): 进行空间/几何维度的混合。
        2. 逐点线性层 (Pointwise Part): 进行通道维度的混合。
    '''
    def __init__(self, irreps_node_input, irreps_edge_attr, irreps_node_output, 
        fc_neurons, use_activation=False, norm_layer='layer', 
        internal_weights=False):
        """
        初始化函数。
        
        Args:
            irreps_node_input: 节点输入特征的Irreps。
            irreps_edge_attr: 边几何属性的Irreps (球谐函数)。
            irreps_node_output: 期望的最终输出特征的Irreps。
            fc_neurons (list): 用于生成张量积权重的径向函数(MLP)的隐藏层维度。
            use_activation (bool): 是否在最后应用门控激活函数。
            norm_layer (str): 使用的归一化层类型。
            internal_weights (bool): 张量积是否使用内部权重。
        """
        
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        norm = get_norm_layer(norm_layer)
        
        self.dtp = DepthwiseTensorProduct(self.irreps_node_input, self.irreps_edge_attr, 
            self.irreps_node_output, bias=False, internal_weights=internal_weights)
        
        self.dtp_rad = None
        if fc_neurons is not None:
            self.dtp_rad = RadialProfile(fc_neurons + [self.dtp.tp.weight_numel])
            for (slice, slice_sqrt_k) in self.dtp.slices_sqrt_k.values():
                self.dtp_rad.net[-1].weight.data[slice, :] *= slice_sqrt_k
                self.dtp_rad.offset.data[slice] *= slice_sqrt_k
                
        irreps_lin_output = self.irreps_node_output
        irreps_scalars, irreps_gates, irreps_gated = irreps2gate(self.irreps_node_output)
        if use_activation:
            irreps_lin_output = irreps_scalars + irreps_gates + irreps_gated
            irreps_lin_output = irreps_lin_output.simplify()
        self.lin = LinearRS(self.dtp.irreps_out.simplify(), irreps_lin_output)
        
        self.norm = None
        if norm_layer is not None:
            self.norm = norm(self.lin.irreps_out)
        
        self.gate = None
        if use_activation:
            if irreps_gated.num_irreps == 0:
                gate = Activation(self.irreps_node_output, acts=[torch.nn.SiLU()])
            else:
                gate = Gate(
                    irreps_scalars, [torch.nn.SiLU() for _, ir in irreps_scalars],  # scalar
                    irreps_gates, [torch.sigmoid for _, ir in irreps_gates],  # gates (scalars)
                    irreps_gated  # gated tensors
                )
            self.gate = gate
    
    
    def forward(self, node_input, edge_attr, edge_scalars, batch=None, **kwargs):
        '''
            Depthwise TP: `node_input` TP `edge_attr`, with TP parametrized by 
            self.dtp_rad(`edge_scalars`).
            执行前向传播。
            
            Args:
                node_input: 节点特征。
                edge_attr: 边的几何方向特征 (球谐函数)。
                edge_scalars: 边的标量距离特征 (RBF输出)。
                batch: 批处理索引。

            Returns:
                torch.Tensor: 经过等变卷积后的输出节点特征。
        """
        '''
        weight = None
        if self.dtp_rad is not None and edge_scalars is not None:    
            weight = self.dtp_rad(edge_scalars)
        out = self.dtp(node_input, edge_attr, weight)
        out = self.lin(out)
        if self.norm is not None:
            out = self.norm(out, batch=batch)
        if self.gate is not None:
            out = self.gate(out)
        return out
        

@compile_mode('script')
class Vec2AttnHeads(torch.nn.Module):
    '''
        Reshape vectors of shape [N, irreps_mid] to vectors of shape
        [N, num_heads, irreps_head].

        一个专门用于塑形(Reshape)的模块。
        
        功能: 将一个扁平的Irreps特征张量，沿着其通道(multiplicity)维度，
            拆分为多个并行的注意力头(attention heads)。
            
        输入形状: [N, D_mid]
        输出形状: [N, num_heads, D_head]
        
        其中 D_mid 是 num_heads * D_head
    '''
    def __init__(self, irreps_head, num_heads):
        """
        初始化函数。

        Args:
            irreps_head (o3.Irreps): 单个注意力头的Irreps定义。
            num_heads (int): 注意力头的数量。
        """
        super().__init__()
        self.num_heads = num_heads
        self.irreps_head = irreps_head

        # --- 准备阶段：计算期望的输入Irreps和切片索引 ---
        
        # 1. 根据单头的Irreps和头数，反向计算出期望的输入Irreps (`irreps_mid_in`)
        #    方法是将单头Irreps中每个分量的通道数(mul)都乘以头数(num_heads)。
        #    例如: 单头是'32x0e', 头数是4, 则期望输入就是'128x0e'。
        self.irreps_mid_in = []
        for mul, ir in irreps_head:
            self.irreps_mid_in.append((mul * num_heads, ir))
        self.irreps_mid_in = o3.Irreps(self.irreps_mid_in)
        
        # 2. 为了提高forward的效率，提前计算好在扁平输入张量中，
        #    每个Irreps分量对应的切片起止位置。
        self.mid_in_indices = []
        start_idx = 0
        for mul, ir in self.irreps_mid_in:
            self.mid_in_indices.append((start_idx, start_idx + mul * ir.dim))
            start_idx = start_idx + mul * ir.dim
    
    
    def forward(self, x):
        """
        执行前向塑形操作。
        
        Args:
            x (torch.Tensor): 待塑形的扁平Irreps特征张量。
                              形状为 [N, irreps_mid_in.dim]。

        Returns:
            torch.Tensor: 塑形后的多头Irreps特征张量。
                          形状为 [N, num_heads, irreps_head.dim]。
        """
        N, _ = x.shape
        out = []
        for ir_idx, (start_idx, end_idx) in enumerate(self.mid_in_indices):
            temp = x.narrow(1, start_idx, end_idx - start_idx)
            temp = temp.reshape(N, self.num_heads, -1)
            out.append(temp)
        out = torch.cat(out, dim=2)
        return out
    
    
    def __repr__(self):
        return '{}(irreps_head={}, num_heads={})'.format(
            self.__class__.__name__, self.irreps_head, self.num_heads)
    
    
@compile_mode('script')
class AttnHeads2Vec(torch.nn.Module):
    '''
        Convert vectors of shape [N, num_heads, irreps_head] into
        vectors of shape [N, irreps_head * num_heads].
        一个专门用于塑形(Reshape)的模块，其功能与Vec2AttnHeads完全相反。
        
        功能: 将一个并行的多头注意力特征张量，合并（拼接）回一个单一的、扁平的特征张量。
            
        输入形状: [N, num_heads, D_head]
        输出形状: [N, D_mid]
        
        其中 D_mid 是 num_heads * D_head
    '''
    def __init__(self, irreps_head):
        """
        初始化函数。

        Args:
            irreps_head (o3.Irreps): 单个注意力头的Irreps定义。
                                     这决定了如何切分输入的特征维度。
        """
        super().__init__()
        self.irreps_head = irreps_head
        self.head_indices = []
        start_idx = 0
        for mul, ir in self.irreps_head:
            self.head_indices.append((start_idx, start_idx + mul * ir.dim))
            start_idx = start_idx + mul * ir.dim
    
    
    def forward(self, x):
        """
        执行前向塑形操作。
        
        Args:
            x (torch.Tensor): 待塑形的多头Irreps特征张量。
                              形状为 [N, num_heads, D_head]。

        Returns:
            torch.Tensor: 塑形后的扁平Irreps特征张量。
                          形状为 [N, num_heads * D_head]。
        """
        N, _, _ = x.shape
        out = []
        for ir_idx, (start_idx, end_idx) in enumerate(self.head_indices):
            temp = x.narrow(2, start_idx, end_idx - start_idx)
            temp = temp.reshape(N, -1)
            out.append(temp)
        out = torch.cat(out, dim=1)
        return out
    
    
    def __repr__(self):
        return '{}(irreps_head={})'.format(self.__class__.__name__, self.irreps_head)


@compile_mode('script')
class GraphAttention(torch.nn.Module):
    '''
        Equiformer模型中等变图注意力(Equivariant Graph Attention)机制的完整实现。
        它负责计算节点之间基于几何和内容信息的相互作用，并据此更新节点特征

        1. 为每条边(i, j)构建融合了节点i、j和边(i,j)信息的消息(message)。
        2. 从消息中分离出两部分：用于计算注意力权重的标量部分(alpha)，和用于传递信息的值部分(value)。
        3. 计算注意力权重 a_ij = softmax(MLP(alpha_ij))。
        4. 将消息聚合到目标节点：node_output_i = sum_j(a_ij * value_ij)。
    '''
    def __init__(self,
        irreps_node_input, irreps_node_attr,
        irreps_edge_attr, irreps_node_output,
        fc_neurons,
        irreps_head, num_heads, irreps_pre_attn=None, 
        rescale_degree=False, nonlinear_message=False,
        alpha_drop=0.1, proj_drop=0.1):
        """
        初始化函数。

        Args:
            irreps_node_input (o3.Irreps): 输入节点特征的Irreps。
            irreps_node_attr (o3.Irreps): 节点属性的Irreps (通常为'1x0e')。###
            irreps_edge_attr (o3.Irreps): 边几何属性的Irreps (球谐函数)。
            irreps_node_output (o3.Irreps): 最终输出节点特征的Irreps。
            fc_neurons (list): 用于径向函数(MLP)的隐藏层维度。
            irreps_head (o3.Irreps): 单个注意力头的Irreps定义。###
            num_heads (int): 注意力头的数量。
            irreps_pre_attn (o3.Irreps, optional): 预注意力消息的Irreps。默认为输入Irreps。
            rescale_degree (bool, optional): 是否用节点度数对聚合结果进行缩放。
            nonlinear_message (bool, optional): 是否使用非线性消息传递路径。
            alpha_drop (float, optional): 注意力权重的Dropout率。
            proj_drop (float, optional): 最终投影层的Dropout率。
        """
        
        super().__init__()
        # 保存并初始化基本参数和Irreps定义
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.irreps_pre_attn = self.irreps_node_input if irreps_pre_attn is None \
            else o3.Irreps(irreps_pre_attn)
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message
        
        # Merge src and dst
        # 通过两个独立的线性层Linear_src和Linear_dst分别处理源节点j和目标节点i的特征
        self.merge_src = LinearRS(self.irreps_node_input, self.irreps_pre_attn, bias=True)
        self.merge_dst = LinearRS(self.irreps_node_input, self.irreps_pre_attn, bias=False)
        
        # 计算多头注意力所需的Irreps
        irreps_attn_heads = irreps_head * num_heads
        irreps_attn_heads, _, _ = sort_irreps_even_first(irreps_attn_heads) #irreps_attn_heads.sort()
        irreps_attn_heads = irreps_attn_heads.simplify() 
        # 计算用于注意力的标量部分(alpha)的总通道数和单头通道数
        mul_alpha = get_mul_0(irreps_attn_heads)
        mul_alpha_head = mul_alpha // num_heads
        # 定义alpha部分的Irreps
        irreps_alpha = o3.Irreps('{}x0e'.format(mul_alpha)) # for attention score
        # 定义线性消息路径下，单个模块需要输出的总Irreps (包含alpha和value)
        irreps_attn_all = (irreps_alpha + irreps_attn_heads).simplify()
        
        # 定义两种不同的消息计算路径
        self.sep_act = None
        if self.nonlinear_message:
            # 非线性消息路径 (更复杂，表达能力更强)
            # Use an extra separable FCTP and Swish Gate for value

            # 第一次DTP，用于生成中间消息，并带有门控激活
            self.sep_act = SeparableFCTP(self.irreps_pre_attn, 
                self.irreps_edge_attr, self.irreps_pre_attn, fc_neurons, 
                use_activation=True, norm_layer=None, internal_weights=False)
            # 线性层，从中间消息中提取用于计算权重的alpha标量部分
            self.sep_alpha = LinearRS(self.sep_act.dtp.irreps_out, irreps_alpha)
            # 第二次DTP，将门控激活后的特征再次与边特征交互，生成最终的value
            self.sep_value = SeparableFCTP(self.irreps_pre_attn, 
                self.irreps_edge_attr, irreps_attn_heads, fc_neurons=None, 
                use_activation=False, norm_layer=None, internal_weights=True)
            # 用于alpha和value的塑形层
            self.vec2heads_alpha = Vec2AttnHeads(o3.Irreps('{}x0e'.format(mul_alpha_head)), 
                num_heads)
            self.vec2heads_value = Vec2AttnHeads(self.irreps_head, num_heads)
        else:
            # 线性消息路径 (更简单)

            # 对应论文中生成 f_ij 的整个过程，被封装在 SeparableFCTP 模块中
            self.sep = SeparableFCTP(self.irreps_pre_attn, 
                self.irreps_edge_attr, irreps_attn_all, fc_neurons, 
                use_activation=False, norm_layer=None)
            # 用于将f_ij拆分为多头的alpha和value
            self.vec2heads = Vec2AttnHeads(
                (o3.Irreps('{}x0e'.format(mul_alpha_head)) + irreps_head).simplify(), 
                num_heads)
        
        # 定义注意力权重计算和最终投影相关的模块
        # 对alpha标量部分应用的激活函数
        self.alpha_act = Activation(o3.Irreps('{}x0e'.format(mul_alpha_head)), 
            [SmoothLeakyReLU(0.2)])
        # 将多头结果合并回一个扁平向量的塑形层
        self.heads2vec = AttnHeads2Vec(irreps_head)
        
        self.mul_alpha_head = mul_alpha_head
        # 对应论文中可学习的向量a，用于和alpha计算内积
        self.alpha_dot = torch.nn.Parameter(torch.randn(1, num_heads, mul_alpha_head))
        torch_geometric.nn.inits.glorot(self.alpha_dot) # # 采用GATv2的初始化方法
        
        # 注意力权重的Dropout
        self.alpha_dropout = None
        if alpha_drop != 0.0:
            self.alpha_dropout = torch.nn.Dropout(alpha_drop)
        
        # 最终的线性投影层，将聚合后的特征投影到期望的输出维度
        self.proj = LinearRS(irreps_attn_heads, self.irreps_node_output)
        self.proj_drop = None
        # 最终的Dropout
        if proj_drop != 0.0:
            self.proj_drop = EquivariantDropout(self.irreps_node_input, 
                drop_prob=proj_drop)

        
        
    def forward(self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalars, 
        batch, **kwargs):
        """
        执行等变图注意力的前向传播。

        Args:
            node_input (torch.Tensor): 节点特征, 形状 [num_nodes, irreps_node_input.dim]
            node_attr (torch.Tensor): 节点属性, 形状 [num_nodes, irreps_node_attr.dim]
            edge_src (torch.Tensor): 边的源节点索引, 形状 [num_edges]
            edge_dst (torch.Tensor): 边的目标节点索引, 形状 [num_edges]
            edge_attr (torch.Tensor): 边的几何特征 (球谐函数), 形状 [num_edges, irreps_edge_attr.dim]
            edge_scalars (torch.Tensor): 边的标量特征 (RBF), 形状 [num_edges, RBF_dim]
            batch (torch.Tensor): 节点到图的分配索引, 形状 [num_nodes]

        Returns:
            torch.Tensor: 更新后的节点特征, 形状 [num_nodes, irreps_node_output.dim]
        """
        
        ## 融合节点内容与几何信息
        # 相加融合源节点和目标节点的特征-x_ij = Linear_dst(x_i) + Linear_src(x_j)
        message_src = self.merge_src(node_input)
        message_dst = self.merge_dst(node_input)
        message = message_src[edge_src] + message_dst[edge_dst]
        
        ## 计算Alpha和Value (根据是否使用非线性消息，路径不同)
        # value计算：值v_ij可以是线性的(直接使用f_{ij}的非标量部分），也可以是非线性的(对f_{ij}$进行Gate激活和第二次DTP）
        # 非线性消息的计算过程
        if self.nonlinear_message:          
            weight = self.sep_act.dtp_rad(edge_scalars)
            message = self.sep_act.dtp(message, edge_attr, weight)
            alpha = self.sep_alpha(message)
            alpha = self.vec2heads_alpha(alpha)
            value = self.sep_act.lin(message)
            # Gate激活
            value = self.sep_act.gate(value)
            # 第二次DPT计算
            value = self.sep_value(value, edge_attr=edge_attr, edge_scalars=edge_scalars)
            value = self.vec2heads_value(value)
        else:
            # 线性消息计算过程
            # message 对应 x_ij，edge_attr 对应 SH(r_ij)，edge_scalars 用于调节权重
            # self.sep(...) 的输出对应论文的f_ij
            message = self.sep(message, edge_attr=edge_attr, edge_scalars=edge_scalars)
            # message 被塑形为多头形式
            message = self.vec2heads(message)
            head_dim_size = message.shape[-1]
            # alpha 是 f_ij^(0) 的多头版本
            alpha = message.narrow(2, 0, self.mul_alpha_head)
            # value 是 f_ij 中除去 alpha 的其余部分
            value = message.narrow(2, self.mul_alpha_head, (head_dim_size - self.mul_alpha_head))
        
        ## 计算注意力权重
        # inner product
        # 对应 LeakyReLU(f_ij^(0))
        alpha = self.alpha_act(alpha)
        # 对应与可学习向量 a 的内积，self.alpha_dot 就是向量 a
        alpha = torch.einsum('bik, aik -> bi', alpha, self.alpha_dot)
        # 对应 softmax_j(z_ij)
        alpha = torch_geometric.utils.softmax(alpha, edge_dst)
        alpha = alpha.unsqueeze(-1)
        if self.alpha_dropout is not None:
            alpha = self.alpha_dropout(alpha)
        
        ## 消息传递与聚合
        # 对应 m_ij = a_ij * v_ij
        attn = value * alpha
        # 对应聚合操作 sum over j
        attn = scatter(attn, index=edge_dst, dim=0, dim_size=node_input.shape[0])
        # 将所有头计算完成并聚合后的结果重新拼接成一个扁平的向量
        attn = self.heads2vec(attn)
        
        ## 可选的度缩放和最终投影
        if self.rescale_degree:
            degree = torch_geometric.utils.degree(edge_dst, 
                num_nodes=node_input.shape[0], dtype=node_input.dtype)
            degree = degree.view(-1, 1)
            attn = attn * degree
            
        # 对拼接后的特征进行最终的线性变换，得到该注意力块的最终输出
        node_output = self.proj(attn)
        
        if self.proj_drop is not None:
            node_output = self.proj_drop(node_output)
        
        return node_output
    
    
    def extra_repr(self):
        output_str = super(GraphAttention, self).extra_repr()
        output_str = output_str + 'rescale_degree={}, '.format(self.rescale_degree)
        return output_str
                    

@compile_mode('script')
class FeedForwardNetwork(torch.nn.Module):
    """
    一个等变的前馈网络 (Feed-Forward Network, FFN)
    紧跟在GraphAttention模块之后，其结构对标标准Transformer中的前馈网络（FFN）层
    
    功能: 它的作用是在注意力层之后对每个节点的特征进行进一步的非线性变换，
          增加模型的容量和表达能力。
          
    结构: 它由两个等变线性层组成，中间夹着一个门控激活函数 (Gated Activation)，
          这在Equiformer论文的图1(d)和4.3节中有描述。
         
    """
    def __init__(self,
        irreps_node_input, irreps_node_attr,
        irreps_node_output, irreps_mlp_mid=None,
        proj_drop=0.1):
        """
        初始化函数。

        Args:
            irreps_node_input (o3.Irreps): 输入节点特征的Irreps。
            irreps_node_attr (o3.Irreps): 节点属性的Irreps (通常为'1x0e')。###
            irreps_node_output (o3.Irreps): 最终输出节点特征的Irreps。
            irreps_mlp_mid (o3.Irreps, optional): FFN中间隐藏层的Irreps。
                                                  如果为None，则默认为输入Irreps。
                                                  通常设置为比输入更高的维度。
            proj_drop (float, optional): 输出层的Dropout率。
        """
        
        super().__init__()
        # 保存和初始化基本参数和Irreps定义
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_mlp_mid = o3.Irreps(irreps_mlp_mid) if irreps_mlp_mid is not None \
            else self.irreps_node_input
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        
        # 定义FFN的两个核心层
        # 第一层：一个带门控SiLU激活的等变全连接层。它将输入特征从 irreps_node_input 映射到中间维度 irreps_mlp_mid
        self.fctp_1 = FullyConnectedTensorProductRescaleSwishGate(
            self.irreps_node_input, self.irreps_node_attr, self.irreps_mlp_mid, 
            bias=True, rescale=_RESCALE)
        # 第二层：一个纯粹的等变全连接线性层（无激活）。它将特征从中间维度 irreps_mlp_mid 映射到最终的输出维度 irreps_node_output。
        self.fctp_2 = FullyConnectedTensorProductRescale(
            self.irreps_mlp_mid, self.irreps_node_attr, self.irreps_node_output, 
            bias=True, rescale=_RESCALE)
        
        # 定义可选的Dropout层
        self.proj_drop = None
        if proj_drop != 0.0:
            self.proj_drop = EquivariantDropout(self.irreps_node_output, 
                drop_prob=proj_drop)
            
        
    def forward(self, node_input, node_attr, **kwargs):
        """
        执行FFN的前向传播。

        Args:
            node_input (torch.Tensor): 输入的节点特征张量。
                                       形状: [num_nodes, irreps_node_input.dim]
            node_attr (torch.Tensor): 节点的属性特征张量。
                                      形状: [num_nodes, irreps_node_attr.dim]

        Returns:
            torch.Tensor: 经过FFN变换后的节点特征张量。
                          形状: [num_nodes, irreps_node_output.dim]
        """
         # 通过第一层 (线性变换 + 门控激活)
        node_output = self.fctp_1(node_input, node_attr)
        # 通过第二层 (线性变换)
        node_output = self.fctp_2(node_output, node_attr)
        # 可选Dropout
        if self.proj_drop is not None:
            node_output = self.proj_drop(node_output)
        return node_output