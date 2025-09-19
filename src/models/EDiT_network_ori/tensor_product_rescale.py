'''
    Rescale output and weights of tensor product.
'''

import torch
import e3nn
from e3nn import o3

from e3nn.nn.models.v2106.gate_points_message_passing import tp_path_exists

import collections
from e3nn.math import perm

_RESCALE = True
def sort_irreps_even_first(irreps):
    """Sorts Irreps with even parity first, then odd, then by L."""
    irreps = o3.Irreps(irreps)

    def sort_key(mul_ir):
        mul, ir = mul_ir
        return (ir.p, ir.l, mul)  # Sort by parity, then l, then multiplicity

    sorted_irreps = sorted(irreps, key=sort_key)

    p = []
    i = 0
    for mul, ir in irreps:
        i_new = sorted_irreps.index((mul, ir))
        p.append(i_new)
        i += 1

    return o3.Irreps(sorted_irreps), p, None  # Returning None for backward compatibility


def irreps2gate(irreps):
    """Decomposes Irreps for a gated activation."""
    irreps = o3.Irreps(irreps)
    scalars = o3.Irreps([(mul, ir) for mul, ir in irreps if ir.l == 0 and ir.p == 1])
    gates = o3.Irreps([(mul, "0e") for mul, ir in irreps if ir.l > 0])
    gated = o3.Irreps([(mul, ir) for mul, ir in irreps if ir.l > 0])
    return scalars, gates, gated

class TensorProductRescale(torch.nn.Module):
    """
    A wrapper around e3nn.o3.TensorProduct that includes rescaling and biases.
    This is a foundational, robust implementation.
    """

    def __init__(
            self,
            irreps_in1,
            irreps_in2,
            irreps_out,
            instructions,
            bias=True,
            rescale=True,
            internal_weights=False,
            shared_weights=False,
            normalization=None
    ):
        super().__init__()
        self.irreps_in1 = o3.Irreps(irreps_in1)
        self.irreps_in2 = o3.Irreps(irreps_in2)
        self.irreps_out = o3.Irreps(irreps_out)
        self.rescale = rescale

        self.tp = o3.TensorProduct(
            self.irreps_in1,
            self.irreps_in2,
            self.irreps_out,
            instructions,
            internal_weights=internal_weights,
            shared_weights=shared_weights,
            normalization=normalization,
        )

        self.use_bias = bias
        if self.use_bias:
            irreps_bias = o3.Irreps([(mul, ir) for mul, ir in self.irreps_out if ir.l == 0])
            self.bias = torch.nn.ParameterList([
                torch.nn.Parameter(torch.zeros(mul)) for mul, ir in irreps_bias
            ])

    def forward_tp_rescale_bias(self, x, y, weight=None):
        # Calculate fan_in for rescaling
        fan_in = self.irreps_in1.dim * self.irreps_in2.dim

        # Perform the tensor product
        out = self.tp(x, y, weight)

        # Rescale the output
        if self.rescale:
            out = out / (fan_in ** 0.5)

        # Add bias to the scalar part
        if self.use_bias and len(self.bias) > 0:
            out_bias = torch.cat([b for b in self.bias], dim=0)
            out[:, :out_bias.shape[0]] += out_bias

        return out

    def forward(self, x, y, weight=None):
        return self.forward_tp_rescale_bias(x, y, weight)


class FullyConnectedTensorProductRescale(TensorProductRescale):
    def __init__(self, irreps_in1, irreps_in2, irreps_out,
                 bias=True, rescale=True,
                 internal_weights=None, shared_weights=None,
                 normalization=None):

        irreps_in1 = o3.Irreps(irreps_in1)
        irreps_in2 = o3.Irreps(irreps_in2)
        irreps_out = o3.Irreps(irreps_out)

        instructions = []
        for i_out, (mul_out, ir_out) in enumerate(irreps_out):
            for i_in1, (mul_in1, ir_in1) in enumerate(irreps_in1):
                for i_in2, (mul_in2, ir_in2) in enumerate(irreps_in2):
                    if ir_out in ir_in1 * ir_in2:
                        instructions.append((i_in1, i_in2, i_out, "uvw", True, mul_out * mul_in1 * mul_in2))

        super().__init__(irreps_in1, irreps_in2, irreps_out, instructions,
                         bias=bias, rescale=rescale,
                         internal_weights=True,
                         shared_weights=True,
                         normalization=normalization)


def DepthwiseTensorProduct(irreps_node_input, irreps_edge_attr, irreps_node_output,
                           internal_weights=False, bias=True):
    '''
        The irreps of output is pre-determined.
        `irreps_node_output` is used to get certain types of vectors.

        一个工厂函数，用于构建并返回一个“深度（depth-wise）”张量积模块
        函数本身不进行计算，而是返回一个模块(功能是执行节点特征和边特征的“深度张量积”运算)
        实现DTP的计算路径构建，区别于普通的FCT

        与全连接张量积（每个输出通道都依赖于所有输入通道的组合）不同，
        深度张量积更加高效，它的一个输出通道只依赖于一个输入通道（通常是节点特征的通道），

        Args:
            irreps_node_input (o3.Irreps): 节点输入特征的Irreps。
            irreps_edge_attr (o3.Irreps): 边属性特征的Irreps (通常是球谐函数)。
            irreps_node_output (o3.Irreps): 期望的输出Irreps的“目标类型”。
            internal_weights (bool): 张量积是否包含内部可学习权重。
            bias (bool): 是否使用偏置。

        Returns:
            TensorProductRescale: 一个配置好的、可执行深度张量积的模块实例。
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
    irreps_output, p, _ = sort_irreps_even_first(irreps_output)  # irreps_output.sort()
    instructions = [(i_1, i_2, p[i_out], mode, train)
                    for i_1, i_2, i_out, mode, train in instructions]
    tp = TensorProductRescale(irreps_node_input, irreps_edge_attr,
                              irreps_output, instructions,
                              internal_weights=internal_weights,
                              shared_weights=internal_weights,
                              bias=bias, rescale=_RESCALE)
    return tp

class LinearRS(FullyConnectedTensorProductRescale):
    """
    A robust E(3) equivariant Linear layer.
    It takes one Irreps tensor as input and applies a linear transformation.
    """

    def __init__(self, irreps_in, irreps_out, bias=True, rescale=True):
        # The __init__ method correctly sets up the tensor product
        # with '1x0e' as the second input's Irreps. This part is correct.
        super().__init__(
            irreps_in,
            o3.Irreps('1x0e'),
            irreps_out,
            bias=bias,
            rescale=rescale,
            # This combination satisfies the e3nn assertion from the previous error
            internal_weights=True,
            shared_weights=True
        )

    def forward(self, x, **kwargs):
        """
        The forward method should only accept one input `x`.
        """
        # Create the trivial second input tensor `y` internally.
        # It should have the same batch dimension as `x` and a feature dimension of 1.
        y = torch.ones_like(x.narrow(1, 0, 1))

        # Now, call the parent's forward method with both `x` and the created `y`.
        out = super().forward(x, y)
        return out
    

def irreps2gate(irreps):
    irreps_scalars = []
    irreps_gated = []
    for mul, ir in irreps:
        if ir.l == 0 and ir.p == 1:
            irreps_scalars.append((mul, ir))
        else:
            irreps_gated.append((mul, ir))
    irreps_scalars = o3.Irreps(irreps_scalars).simplify()
    irreps_gated = o3.Irreps(irreps_gated).simplify()
    if irreps_gated.dim > 0:
        ir = '0e'
    else:
        ir = None
    irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated]).simplify()
    return irreps_scalars, irreps_gates, irreps_gated


class FullyConnectedTensorProductRescaleSwishGate(FullyConnectedTensorProductRescale):
    def __init__(self,
        irreps_in1, irreps_in2, irreps_out,
        bias=True, rescale=True,
        internal_weights=None, shared_weights=None,
        normalization=None):
        
        irreps_scalars, irreps_gates, irreps_gated = irreps2gate(irreps_out)
        if irreps_gated.num_irreps == 0:
            gate = e3nn.nn.Activation(irreps_out, acts=[torch.nn.functional.silu])
        else:
            gate = e3nn.nn.Gate(
                irreps_scalars, [torch.nn.functional.silu for _, ir in irreps_scalars],  # scalar
                irreps_gates, [torch.sigmoid for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated  # gated tensors
            )
        super().__init__(irreps_in1, irreps_in2, gate.irreps_in,
            bias=bias, rescale=rescale,
            internal_weights=internal_weights, shared_weights=shared_weights,
            normalization=normalization)
        self.gate = gate
        
    
    def forward(self, x, y, weight=None):
        out = self.forward_tp_rescale_bias(x, y, weight)
        out = self.gate(out)
        return out
    
    
def sort_irreps_even_first(irreps):
    Ret = collections.namedtuple("sort", ["irreps", "p", "inv"])
    out = [(ir.l, -ir.p, i, mul) for i, (mul, ir) in enumerate(irreps)]
    out = sorted(out)
    inv = tuple(i for _, _, i, _ in out)
    p = perm.inverse(inv)
    irreps = o3.Irreps([(mul, (l, -p)) for l, p, _, mul in out])
    return Ret(irreps, p, inv)
        

if __name__ == '__main__':
    

    irreps_1 = o3.Irreps('32x0e+16x1o+8x2e')
    irreps_2 = o3.Irreps('4x0e+4x1o+4x2e')
    irreps_out = o3.Irreps('16x0e+8x1o+4x2e')
    
    irreps_mid = []
    instructions = []
    for i, (mul, ir_in) in enumerate(irreps_1):
        for j, (_, ir_edge) in enumerate(irreps_2):
            for ir_out in ir_in * ir_edge:
                if ir_out in irreps_out or ir_out == o3.Irrep(0, 1):
                    k = len(irreps_mid)
                    irreps_mid.append((mul, ir_out))
                    instructions.append((i, j, k, 'uvu', True))
    irreps_mid = o3.Irreps(irreps_mid)
    irreps_mid, p, _ = irreps_mid.sort()

    instructions = [
        (i_1, i_2, p[i_out], mode, train)
        for i_1, i_2, i_out, mode, train in instructions
    ]
    
    torch.manual_seed(0)
    tp = o3.TensorProduct(irreps_1, irreps_2, irreps_mid, instructions)
    
    torch.manual_seed(0)
    tp_rs = TensorProductRescale(irreps_1, irreps_2, irreps_mid, instructions, 
        bias=False, rescale=False)
    
    inputs_1 = irreps_1.randn(10, -1)
    inputs_2 = irreps_2.randn(10, -1)
    
    out_tp = tp.forward(inputs_1, inputs_2)
    out_tp_rs = tp_rs.forward(inputs_1, inputs_2)
    print('[TP] before rescaling difference: {}'.format(torch.max(torch.abs(out_tp - out_tp_rs))))
    
    tp_rs.rescale = True
    tp_rs.init_rescale_bias()
    out_tp_rs = tp_rs.forward(inputs_1, inputs_2)
    print('[TP] after rescaling difference: {}'.format(torch.max(torch.abs(out_tp - out_tp_rs))))
    
    # FullyConnectedTensorProduct
    torch.manual_seed(0)
    fctp = o3.FullyConnectedTensorProduct(irreps_1, irreps_2, irreps_out)
    torch.manual_seed(0)
    fctp_rs = FullyConnectedTensorProductRescale(irreps_1, irreps_2, irreps_out, 
        bias=False, rescale=False)
    
    out_fctp = fctp.forward(inputs_1, inputs_2)
    out_fctp_rs = fctp_rs.forward(inputs_1, inputs_2)
    print('[FCTP] before rescaling difference: {}'.format(torch.max(torch.abs(out_fctp - out_fctp_rs))))
    
    fctp_rs.rescale = True
    fctp_rs.init_rescale_bias()
    out_fctp_rs = fctp_rs.forward(inputs_1, inputs_2)
    print('[FCTP] after rescaling difference: {}'.format(torch.max(torch.abs(out_fctp - out_fctp_rs))))
    