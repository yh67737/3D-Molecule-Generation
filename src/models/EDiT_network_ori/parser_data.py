# parser.py
import argparse


def create_parser():
    parser = argparse.ArgumentParser(description='E-DiT Model Training Arguments')

    # --- 模型架构参数 (Model Architecture) ---
    g_arch = parser.add_argument_group('Architecture')
    g_arch.add_argument('--num_blocks', type=int, default=6, help='Number of E-DiT blocks.')
    g_arch.add_argument('--num_heads', type=int, default=4, help='Number of attention heads.')
    g_arch.add_argument('--norm_layer', type=str, default='layer', help='Type of normalization layer (e.g., "layer").')
    g_arch.add_argument('--time_embed_dim', type=int, default=128, help='Dimension of timestep embedding.')

    # --- Irreps 参数 (Irreps Definitions) ---
    g_irreps = parser.add_argument_group('Irreps')
    g_irreps.add_argument('--irreps_node_hidden', type=str, default='128x0e+64x1o+32x2e',
                          help='Hidden node feature irreps.')
    g_irreps.add_argument('--irreps_edge', type=str, default='128x0e+64x1o+32x2e', help='Hidden edge feature irreps.')
    g_irreps.add_argument('--irreps_node_attr', type=str, default='6x0e', help='Node attribute (atom type) irreps.')
    g_irreps.add_argument('--irreps_edge_attr_type', type=str, default='5x0e',
                          help='Edge attribute (bond type) irreps.')
    g_irreps.add_argument('--irreps_sh', type=str, default='1x0e+1x1e+1x2e', help='Spherical harmonics irreps.')
    g_irreps.add_argument('--irreps_head', type=str, default='32x0e+16x1o+8x2e', help='Single attention head irreps.')
    g_irreps.add_argument('--irreps_mlp_mid', type=str, default='384x0e+192x1o+96x2e',
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
    g_embed.add_argument('--num_rbf', type=int, default=128, help='Number of radial basis functions.')
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

    return parser