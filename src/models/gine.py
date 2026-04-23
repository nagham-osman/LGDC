import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Linear, Sequential
from torch.nn.modules.normalization import LayerNorm

from src import utils
from src.diffusion import diffusion_utils

class DenseGINELayer(nn.Module):
    """GIN layer updating node and edge features."""

    def __init__(self, node_mlp, edge_mlp, hidden_dims):
        super().__init__()
        self.node_mlp = node_mlp
        self.edge_mlp = edge_mlp
        self.normX = LayerNorm(hidden_dims['dx'], eps=1e-5)
        self.normE = LayerNorm(hidden_dims['de'], eps=1e-5)

    def forward(self, emb_node, emb_edge, adj_mtx, node_mask, eps=0):
        # Get batch size and number of nodes
        B, N, _, _ = emb_edge.shape

        # Create masks for nodes and edges
        x_mask = node_mask.unsqueeze(-1)
        e_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(1)

        # Apply masks to node and edge embeddings
        emb_node, emb_edge = emb_node * x_mask, emb_edge * e_mask

        # Update node features using the adjacency matrix and edge features
        emb_node_new = (1 + eps) * emb_node + (adj_mtx * (emb_node.unsqueeze(2) + emb_edge)).sum(axis=1)
        emb_node_new = self.node_mlp(emb_node_new)

        # Combine node and edge attributes for edge update
        combined_attr = torch.cat([
            emb_node_new.unsqueeze(2).repeat(1, 1, N, 1),
            emb_node_new.unsqueeze(1).repeat(1, N, 1, 1),
            emb_edge
        ], dim=-1)

        # Update edge features
        new_emb_edge = self.edge_mlp(combined_attr)
        emb_node = self.normX(emb_node + emb_node_new)
        emb_edge = self.normE(emb_edge + new_emb_edge)

        return emb_node, emb_edge

class DenseGINNet(nn.Module):
    """Dense GIN Network with edge feature processing"""

    def __init__(self, n_layers, input_dims, hidden_mlp_dims, hidden_dims, output_dims,
                 act_fn_in=nn.ReLU(), act_fn_out=nn.ReLU()):
        super(DenseGINNet, self).__init__()

        # Ensure hidden dimensions for nodes and edges are equal
        assert hidden_dims['de'] == hidden_dims['dx'], f'for GINE the hidden dimension size need match'
        self.n_layers = n_layers
        self.out_dim_X, self.out_dim_E = output_dims['X'], output_dims['E']

        # Define input MLP for node features
        self.mlp_in_X = Sequential(
            Linear(input_dims['X'], hidden_mlp_dims['X']),
            act_fn_in,
            Linear(hidden_mlp_dims['X'], hidden_dims['dx']),
            act_fn_in
        )

        # Define input MLP for edge features
        self.mlp_in_E = Sequential(
            Linear(input_dims['E'], hidden_mlp_dims['E']),
            act_fn_in,
            Linear(hidden_mlp_dims['E'], hidden_dims['de']),
            act_fn_in
        )

        self.gine_convs = nn.ModuleList()

        # Define GIN layers
        for _ in range(n_layers):
            self.gine_convs.append(DenseGINELayer(
                node_mlp=self.build_mlp(hidden_dims['dx'], hidden_dims['dx'], hidden_dims['dx']),
                edge_mlp=self.build_mlp(hidden_dims['de'] + 2 * hidden_dims['dx'], hidden_dims['dx'], hidden_dims['de'])
            ))

        # Define output MLP for node features
        self.mlp_out_X = nn.Sequential(
            nn.Linear(hidden_dims['dx'], hidden_mlp_dims['X']),
            act_fn_out,
            nn.Linear(hidden_mlp_dims['X'], output_dims['X'])
        )

        # Define output MLP for edge features
        self.mlp_out_E = nn.Sequential(
            nn.Linear(hidden_dims['de'], hidden_mlp_dims['E']),
            act_fn_out,
            nn.Linear(hidden_mlp_dims['E'], output_dims['E'])
        )

    def build_mlp(self, input_dim, hidden_mlp, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_mlp),
            nn.ReLU(),
            nn.Linear(hidden_mlp, output_dim),
            nn.ReLU()
        )

    def forward(self, X, E, node_mask):
        bs, n = X.shape[:2]

        # Create diagonal mask to avoid self-loops
        diag_mask = ~torch.eye(n, dtype=bool).to(E.device).unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

        # Create adjacency matrix
        adj_mtx = (torch.sum(E, axis=-1) != 0).unsqueeze(-1)

        # Apply input MLPs to node and edge features
        X = self.mlp_in_X(X)
        E = (self.mlp_in_E(E) + self.mlp_in_E(E).transpose(1, 2)) / 2

        # Apply GIN layers
        for conv in self.gine_convs:
            X, E = conv(emb_node=X, emb_edge=E, adj_mtx=adj_mtx, node_mask=node_mask, eps=0.1)
            X = F.relu(X)

        # Apply output MLPs to node and edge features
        X = self.mlp_out_X(X)
        E = self.mlp_out_E(E) * diag_mask

        return utils.PlaceHolder(X=X, E=(E + E.transpose(1, 2)) / 2, y=None).mask(node_mask)
