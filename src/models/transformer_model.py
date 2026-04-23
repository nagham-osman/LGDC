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
    """ Transformer that updates node, edge and global features
        d_x: node features
        d_e: edge features
        dz : global features
        n_head: the number of heads in the multi_head_attention
        dim_feedforward: the dimension of the feedforward network model after self-attention
        dropout: dropout probablility. 0 to disable
        layer_norm_eps: eps value in layer normalizations.
    """
    def __init__(self, dx: int, de: int, dy: int, n_head: int, dim_ffX: int = 2048,
                 dim_ffE: int = 128, dim_ffy: int = 2048, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5, device=None, dtype=None) -> None:
        kw = {'device': device, 'dtype': dtype}
        super().__init__()

        self.self_attn = NodeEdgeBlock(dx, de, dy, n_head, **kw)

        self.linX1, self.linX2 = Linear(dx, dim_ffX, **kw), Linear(dim_ffX, dx, **kw)
        self.normX1, self.normX2 = (LayerNorm(dx, eps=layer_norm_eps, **kw) for _ in range(2))

        self.dropoutX1, self.dropoutX2, self.dropoutX3 = (Dropout(dropout) for _ in range(3))

        self.linE1, self.linE2 = Linear(de, dim_ffE, **kw), Linear(dim_ffE, de, **kw)
        self.normE1, self.normE2 = (LayerNorm(de, eps=layer_norm_eps, **kw) for _ in range(2))
        self.dropoutE1, self.dropoutE2, self.dropoutE3 = (Dropout(dropout) for _ in range(3))

        self.lin_y1, self.lin_y2 = Linear(dy, dim_ffy, **kw), Linear(dim_ffy, dy, **kw)
        self.norm_y1, self.norm_y2 = LayerNorm(dy, eps=layer_norm_eps, **kw), LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.dropout_y1, self.dropout_y2, self.dropout_y3 = (Dropout(dropout) for _ in range(3))
        self.activation = F.relu

    def forward(self, X: Tensor, E: Tensor, y, node_mask: Tensor):
        """ Pass the input through the encoder layer.
            X: (bs, n, d)
            E: (bs, n, n, d)
            y: (bs, dy)
            node_mask: (bs, n) Mask for the src keys per batch (optional)
            Output: newX, newE, new_y with the same shape.
        """

        newX, newE, new_y = self.self_attn(X, E, y, node_mask=node_mask)

        newX_d = self.dropoutX1(newX)
        X = self.normX1(X + newX_d)

        newE_d = self.dropoutE1(newE)
        E = self.normE1(E + newE_d)

        new_y_d = self.dropout_y1(new_y)
        y = self.norm_y1(y + new_y_d)

        ff_outputX = self.linX2(self.dropoutX2(self.activation(self.linX1(X))))
        ff_outputX = self.dropoutX3(ff_outputX)
        X = self.normX2(X + ff_outputX)

        ff_outputE = self.linE2(self.dropoutE2(self.activation(self.linE1(E))))
        ff_outputE = self.dropoutE3(ff_outputE)
        E = self.normE2(E + ff_outputE)

        ff_output_y = self.lin_y2(self.dropout_y2(self.activation(self.lin_y1(y))))
        ff_output_y = self.dropout_y3(ff_output_y)
        y = self.norm_y2(y + ff_output_y)

        return X, E, y


class NodeEdgeBlock(nn.Module):
    """ Self attention layer that also updates the representations on the edges. """
    def __init__(self, dx, de, dy, n_head, **kwargs):
        super().__init__()
        assert dx % n_head == 0, f"dx: {dx} -- nhead: {n_head}"
        self.dx, self.de, self.dy, self.df = dx, de, dy, int(dx / n_head)
        self.n_head = n_head

        # Attention
        self.q, self.k, self.v = (Linear(dx, dx) for _ in range(3))

        # FiLM E to X
        self.e_add, self.e_mul = (Linear(de, dx) for _ in range(2))

        # FiLM y to E # Warning: here it's dx and not de
        self.y_e_mul, self.y_e_add = Linear(dy, dx),  Linear(dy, dx)

        # FiLM y to X
        self.y_x_mul, self.y_x_add = Linear(dy, dx), Linear(dy, dx)

        # Process y
        self.y_y = Linear(dy, dy)
        self.x_y = Xtoy(dx, dy)
        self.e_y = Etoy(de, dy)

        # Output layers
        self.x_out = Linear(dx, dx)
        self.e_out = Linear(dx, de)
        self.y_out = nn.Sequential(nn.Linear(dy, dy), nn.ReLU(), nn.Linear(dy, dy))

    def forward(self, X, E, y, node_mask):
        """
        :param X: bs, n, d        node features
        :param E: bs, n, n, d     edge features
        :param y: bs, dz           global features
        :param node_mask: bs, n
        :return: newX, newE, new_y with the same shape.
        """
        bs, n, _ = X.shape
        x_mask = node_mask.unsqueeze(-1)        # bs, n, 1
        # e_mask1 with (bs, n, 1, 1) ; e_mask2 with bs, 1, n, 1
        e_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(1)  # bs, n, n, 1

        # 1. Map X to keys and queries
        # Q: (bs, n, dx); K: # (bs, n, dx)
        Q, K = self.q(X) * x_mask, self.k(X) * x_mask
        diffusion_utils.assert_correctly_masked(Q, x_mask)

        # 2. Reshape to (bs, n, n_head, df) with dx = n_head * df,
        # then Q: (bs, 1, n, n_head, df); K: (bs, n, 1, n head, df)
        Q = Q.reshape((bs, n, self.n_head, self.df)).unsqueeze(2)
        K = K.reshape((bs, n, self.n_head, self.df)).unsqueeze(1)

        # Compute unnormalized attentions. Y is (bs, n, n, n_head, df)
        Y = (Q * K) / math.sqrt((Q * K).size(-1))
        diffusion_utils.assert_correctly_masked(Y, (e_mask).unsqueeze(-1))

        E1 = (self.e_mul(E) * e_mask).reshape((bs, n, n, self.n_head, self.df))
        E2 = (self.e_add(E) * e_mask).reshape((bs, n, n, self.n_head, self.df))

        # Incorporate edge features to the self attention scores.
        Y = Y * (E1 + 1) + E2                  # (bs, n, n, n_head, df)

        # Incorporate y to E
        newE = Y.flatten(start_dim=3)                      # bs, n, n, dx
        ye1 = self.y_e_add(y).unsqueeze(1).unsqueeze(1)  # bs, 1, 1, de
        ye2 = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)
        newE = ye1 + (ye2 + 1) * newE

        # Output E
        newE = self.e_out(newE) * e_mask     # bs, n, n, de
        diffusion_utils.assert_correctly_masked(newE, e_mask)

        # Compute attentions. attn is still (bs, n, n, n_head, df)
        softmax_mask = x_mask.unsqueeze(1).expand(-1, n, -1, self.n_head)    # bs, 1, n, 1
        attn = masked_softmax(Y, softmax_mask, dim=2)  # bs, n, n, n_head, replace with -inf

        # bs, n, dx -> # (bs, 1, n, n_head, df)
        V = (self.v(X) * x_mask).reshape((bs, n, self.n_head, self.df)).unsqueeze(1)

        # Compute weighted values
        weighted_V = (attn * V).sum(dim=2)

        # Send output to input dim
        weighted_V = weighted_V.flatten(start_dim=2)            # bs, n, dx

        # Incorporate y to X
        yx1 = self.y_x_add(y).unsqueeze(1)
        yx2 = self.y_x_mul(y).unsqueeze(1)
        newX = yx1 + (yx2 + 1) * weighted_V

        # Output X
        newX = self.x_out(newX) * x_mask
        diffusion_utils.assert_correctly_masked(newX, x_mask)

        # Process y based on X axnd E
        y = self.y_y(y)
        e_y = self.e_y(E)
        x_y = self.x_y(X)
        new_y = y + x_y + e_y
        new_y = self.y_out(new_y)               # bs, dy

        return newX, newE, new_y


class GraphTransformer(nn.Module):
    """
    n_layers : int -- number of layers
    dims : dict -- contains dimensions for each feature type
    """
    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict, hidden_dims: dict,
                 output_dims: dict, act_fn_in: nn.ReLU(), act_fn_out: nn.ReLU()):
        super().__init__()
        self.n_layers = n_layers
        self.out_dim_X, self.out_dim_E, self.out_dim_y = output_dims['X'], output_dims['E'], output_dims['y']
        print(input_dims)
        print('out_dim')
        print(output_dims)

        self.mlp_in_X = nn.Sequential(nn.Linear(input_dims['X'], hidden_mlp_dims['X']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['X'], hidden_dims['dx']), act_fn_in)

        self.mlp_in_E = nn.Sequential(nn.Linear(input_dims['E'], hidden_mlp_dims['E']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['E'], hidden_dims['de']), act_fn_in)

        self.mlp_in_y = nn.Sequential(nn.Linear(input_dims['y'], hidden_mlp_dims['y']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['y'], hidden_dims['dy']), act_fn_in)

        self.tf_layers = nn.ModuleList([XEyTransformerLayer(dx=hidden_dims['dx'],
                                                            de=hidden_dims['de'],
                                                            dy=hidden_dims['dy'],
                                                            n_head=hidden_dims['n_head'],
                                                            dim_ffX=hidden_dims['dim_ffX'],
                                                            dim_ffE=hidden_dims['dim_ffE'])
                                        for i in range(n_layers)])

        self.mlp_out_X = nn.Sequential(nn.Linear(hidden_dims['dx'], hidden_mlp_dims['X']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['X'], output_dims['X']))

        self.mlp_out_E = nn.Sequential(nn.Linear(hidden_dims['de'], hidden_mlp_dims['E']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['E'], output_dims['E']))

        self.mlp_out_y = nn.Sequential(nn.Linear(hidden_dims['dy'], hidden_mlp_dims['y']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['y'], output_dims['y']))

    def forward(self, X, E, y, node_mask):
        bs, n = X.shape[0], X.shape[1]

        diag_mask = torch.eye(n)
        diag_mask = ~diag_mask.type_as(E).bool()
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

        X_to_out, E_to_out, y_to_out = X[..., :self.out_dim_X], E[..., :self.out_dim_E], y[..., :self.out_dim_y]

        new_E = self.mlp_in_E(E)
        new_E = (new_E + new_E.transpose(1, 2)) / 2
        after_in = utils.PlaceHolder(X=self.mlp_in_X(X), E=new_E, y=self.mlp_in_y(y)).mask(node_mask)
        X, E, y = after_in.X, after_in.E, after_in.y

        for layer in self.tf_layers:
            X, E, y = layer(X, E, y, node_mask)

        X = self.mlp_out_X(X)
        E = self.mlp_out_E(E)
        y = self.mlp_out_y(y)

        X = (X + X_to_out)
        E = (E + E_to_out) * diag_mask
        y = y + y_to_out

        E = 1/2 * (E + torch.transpose(E, 1, 2))

        return utils.PlaceHolder(X=X, E=E, y=y).mask(node_mask)
