import math

import torch
import torch.nn as nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn import functional as F
from torch import Tensor

from src import utils
from src.diffusion import diffusion_utils
from src.models.layers import Xtoy, Etoy, masked_softmax


class TransformerLayer(nn.Module):
    """Transformer layer updating node, edge, and global features."""

    def __init__(self, dx, de, n_head,
                 dim_ffX=256, dim_ffE=128,
                 dropout=0.1, layer_norm_eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.self_attn = NodeEdgeBlock(dx, de, n_head, device=device, dtype=dtype)
        self._init_layers(dx, de, dim_ffX, dim_ffE, dropout, layer_norm_eps, device, dtype)

    def _init_layers(self, dx, de, dim_ffX, dim_ffE,
                     dropout, layer_norm_eps, device, dtype):
        # Layers for node features X
        kw = {'device': device, 'dtype': dtype}

        self.linX1 = Linear(dx+de, dim_ffX, **kw)
        self.linX2 = Linear(dim_ffX, dx+de, **kw)
        self.normX1, self.normX2 = (LayerNorm(dx+de, eps=layer_norm_eps, **kw) for _ in range(2))
        self.dropoutX1, self.dropoutX2, self.dropoutX3 = (Dropout(dropout) for _ in range(3))
        self.activation = F.relu

    def forward(self, X, E, node_mask):
        newX = self.self_attn(X, E, node_mask)
        X = self.normX1(X + self.dropoutX1(newX))
        X = self.normX2(X + self.dropoutX3(self.linX2(self.dropoutX2(self.activation(self.linX1(X))))))
        return X


class NodeEdgeBlock(nn.Module):
    """Self-attention layer updating node, edge, and global features."""

    def __init__(self, dx, de, n_head, **kwargs):
        super().__init__()
        assert (dx+de) % n_head == 0, f"dx+de: {dx+de} -- nhead: {n_head}"
        assert dx % n_head == 0, f"dx: {dx} -- nhead: {n_head}"
        self.dx, self.de, self.df, self.n_head = dx, de, (dx+de) // n_head, n_head
        self.q, self.k, self.v = (Linear(dx+de, dx+de, **kwargs) for _ in range(3))
        self.e_add, self.e_mul = Linear(de, dx+de, **kwargs), Linear(de, dx+de, **kwargs)
        self.x_out= Linear(dx+de, dx+de, **kwargs)

    def forward(self, X, E, node_mask):
        # inout: X bs*n*(dx+de), E: |\gE|*de
        bs, n, _ = X.shape
        x_mask = node_mask.unsqueeze(-1)
        e_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(1)
        Q, K, V = self.q(X) * x_mask, self.k(X) * x_mask, self.v(X) * x_mask

        diffusion_utils.assert_correctly_masked(Q, x_mask)
        Q = Q.reshape(bs, n, self.n_head, self.df).unsqueeze(2)
        K = K.reshape(bs, n, self.n_head, self.df).unsqueeze(1)
        V = V.reshape(bs, n, self.n_head, self.df).unsqueeze(1)
        Y = Q * K / math.sqrt(self.df)
        diffusion_utils.assert_correctly_masked(Y, e_mask.unsqueeze(-1))
        E1 = (self.e_mul(E) * e_mask).reshape((bs, n, n, self.n_head, self.df))
        E2 = (self.e_add(E) * e_mask).reshape((bs, n, n, self.n_head, self.df))
        Y = Y * (E1 + 1) + E2
        attn = masked_softmax(Y, x_mask.unsqueeze(1).expand(-1, n, -1, self.n_head), dim=2)
        weighted_V = (attn * V).sum(dim=2).flatten(start_dim=2)
        newX = self.x_out(weighted_V) * x_mask
        return newX


class GraphTransformerLowrank(nn.Module):
    """Graph Transformer with multiple layers."""

    def __init__(self, n_layers, input_dims, hidden_mlp_dims, hidden_dims, output_dims,
                 act_fn_in=nn.ReLU(), act_fn_out=nn.ReLU()):
        super().__init__()
        self.n_layers = n_layers
        self.out_dim_X, self.out_dim_E = output_dims['X'], output_dims['E']
        self.dx = hidden_dims['dx']
        self.de = hidden_dims['de']

        self.mlp_in_X = nn.Sequential(nn.Linear(input_dims['X'], hidden_mlp_dims['X']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['X'], self.dx+self.de), act_fn_in)

        # Create a tensor for the embedding matrix
        embed_tensor = torch.randn(input_dims['E'], hidden_dims['de'])
        # Initialize the tensor using a specific method, e.g., uniform distribution
        torch.nn.init.xavier_normal_(embed_tensor)
        # Convert the tensor to a parameter
        self.embed_param = nn.Parameter(embed_tensor)

        self.tf_layers = nn.ModuleList([TransformerLayer(dx=hidden_dims['dx'],
                                                            de=hidden_dims['de'],
                                                            n_head=hidden_dims['n_head'],
                                                            dim_ffX=hidden_dims['dim_ffX'],
                                                            dim_ffE=hidden_dims['dim_ffE'])
                                        for _ in range(n_layers)])

        self.mlp_out_X = nn.Sequential(nn.Linear(self.dx, hidden_mlp_dims['X']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['X'], output_dims['X']))


    def forward(self, X, E, node_mask):
        # E: all zero indicates no edge. [1, 0] indicates the edge type
        bs, n = X.shape[:2]
        diag_mask = ~torch.eye(n, dtype=bool).to(E.device).unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)
        E = E @ self.embed_param
        X, E = self.mlp_in_X(X), (E + E.transpose(1, 2)) / 2
        for layer in self.tf_layers:
            X = layer(X, E, node_mask)
        X_out = self.mlp_out_X(X[..., :self.dx])
        ne, de = self.embed_param.shape
        E = X[..., self.dx:].reshape((bs, 1, n, 1, -1)) \
            * X[..., self.dx:].reshape((bs, n, 1, 1, -1)) \
            * self.embed_param.reshape(1, 1, 1, ne, de)
        E = torch.sum(E, dim=-1) * diag_mask
        return utils.PlaceHolder(X=X_out, E=(E + E.transpose(1, 2)) / 2, y=None).mask(node_mask)
