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


class XEyTransformerLayer(nn.Module):
    """Transformer layer updating node, edge, and global features."""

    def __init__(self, dx, de, n_head,
                 dim_ffX=2048, dim_ffE=128,
                 dropout=0.1, layer_norm_eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.self_attn = NodeEdgeBlock(dx, de, n_head, device=device, dtype=dtype)
        self._init_layers(dx, de, dim_ffX, dim_ffE, dropout, layer_norm_eps, device, dtype)

    def _init_layers(self, dx, de, dim_ffX, dim_ffE,
                     dropout, layer_norm_eps, device, dtype):
        # Layers for node features X
        kw = {'device': device, 'dtype': dtype}

        self.linX1 = Linear(dx, dim_ffX, **kw)
        self.linX2 = Linear(dim_ffX, dx, **kw)
        self.normX1, self.normX2 = (LayerNorm(dx, eps=layer_norm_eps, **kw) for _ in range(2))
        self.dropoutX1, self.dropoutX2, self.dropoutX3 = (Dropout(dropout) for _ in range(3))

        # Layers for edge features X
        self.linE1 = Linear(de, dim_ffE, **kw)
        self.linE2 = Linear(dim_ffE, de, **kw)
        self.normE1, self.normE2 = (LayerNorm(de, eps=layer_norm_eps, **kw) for _ in range(2))
        self.dropoutE1, self.dropoutE2, self.dropoutE3 = (Dropout(dropout) for _ in range(3))

        self.activation = F.relu

    def forward(self, X, E, node_mask):
        newX, newE, = self.self_attn(X, E, node_mask)

        X = self.normX1(X + self.dropoutX1(newX))
        E = self.normE1(E + self.dropoutE1(newE))

        X = self.normX2(X + self.dropoutX3(self.linX2(self.dropoutX2(self.activation(self.linX1(X))))))
        E = self.normE2(E + self.dropoutE3(self.linE2(self.dropoutE2(self.activation(self.linE1(E))))))

        return X, E


class NodeEdgeBlock(nn.Module):
    """Self-attention layer updating node, edge, and global features."""

    def __init__(self, dx, de, n_head, **kwargs):
        super().__init__()
        assert dx % n_head == 0, f"dx: {dx} -- nhead: {n_head}"
        self.dx, self.de, self.df, self.n_head = dx, de, dx // n_head, n_head
        self._init_attention_layers(dx, de, kwargs)

    def _init_attention_layers(self, dx, de, kwargs):
        self.q, self.k, self.v = Linear(dx, dx, **kwargs), Linear(dx, dx, **kwargs), Linear(dx, dx, **kwargs)
        self.e_add, self.e_mul = Linear(de, dx, **kwargs), Linear(de, dx, **kwargs)
        self.x_out, self.e_out = Linear(dx, dx, **kwargs), Linear(dx, de, **kwargs)

    def forward(self, X, E, node_mask):
        bs, n, _ = X.shape
        x_mask = node_mask.unsqueeze(-1)
        e_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(1)
        Q, K, V = self.q(X) * x_mask, self.k(X) * x_mask, self.v(X) * x_mask

        diffusion_utils.assert_correctly_masked(Q, x_mask)

        Q = Q.reshape(bs, n, self.n_head, self.df).unsqueeze(2)
        K = K.reshape(bs, n, self.n_head, self.df).unsqueeze(1)
        V = V.reshape(bs, n, self.n_head, self.df).unsqueeze(1)
        Y = (Q * K / math.sqrt(self.df))
        diffusion_utils.assert_correctly_masked(Y, (e_mask).unsqueeze(-1))

        E1 = (self.e_mul(E) * e_mask).reshape((bs, n, n, self.n_head, self.df))
        E2 = (self.e_add(E) * e_mask).reshape((bs, n, n, self.n_head, self.df))
        Y = (Y * (E1 + 1) + E2)

        newE = self.e_out(Y.flatten(start_dim=3)) * e_mask
        attn = masked_softmax(Y, x_mask.unsqueeze(1).expand(-1, n, -1, self.n_head), dim=2)
        weighted_V = (attn * V).sum(dim=2).flatten(start_dim=2)
        newX = self.x_out(weighted_V) * x_mask
        return newX, newE


class GraphTransformerUncon(nn.Module):
    """Graph Transformer with multiple layers."""

    def __init__(self, n_layers, input_dims, hidden_mlp_dims, hidden_dims, output_dims,
                 act_fn_in=nn.ReLU(), act_fn_out=nn.ReLU()):
        super().__init__()
        self.n_layers = n_layers
        self.out_dim_X, self.out_dim_E = output_dims['X'], output_dims['E']

        self.mlp_in_X = nn.Sequential(nn.Linear(input_dims['X'], hidden_mlp_dims['X']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['X'], hidden_dims['dx']), act_fn_in)

        self.mlp_in_E = nn.Sequential(nn.Linear(input_dims['E'], hidden_mlp_dims['E']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['E'], hidden_dims['de']), act_fn_in)

        self.tf_layers = nn.ModuleList([XEyTransformerLayer(dx=hidden_dims['dx'],
                                                            de=hidden_dims['de'],
                                                            n_head=hidden_dims['n_head'],
                                                            dim_ffX=hidden_dims['dim_ffX'],
                                                            dim_ffE=hidden_dims['dim_ffE'])
                                        for _ in range(n_layers)])

        self.mlp_out_X = nn.Sequential(nn.Linear(hidden_dims['dx'], hidden_mlp_dims['X']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['X'], output_dims['X']))

        self.mlp_out_E = nn.Sequential(nn.Linear(hidden_dims['de'], hidden_mlp_dims['E']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['E'], output_dims['E']))

    def forward(self, X, E, node_mask):
        # E: all zero indicates no edge. [1, 0] indicates the edge type
        bs, n = X.shape[:2]
        diag_mask = ~torch.eye(n, dtype=bool).to(E.device).unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

        X, E = self.mlp_in_X(X), (self.mlp_in_E(E) + self.mlp_in_E(E).transpose(1, 2)) / 2
        for layer in self.tf_layers:
            X, E = layer(X, E, node_mask)

        X, E = self.mlp_out_X(X), self.mlp_out_E(E) * diag_mask
        return utils.PlaceHolder(X=X, E=(E + E.transpose(1, 2)) / 2, y=None).mask(node_mask)
