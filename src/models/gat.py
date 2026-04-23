import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Linear, Sequential
from torch.nn.modules.normalization import LayerNorm

from src import utils
from src.diffusion import diffusion_utils


class DenseGATLayer(nn.Module):
    """GAT layer updating node and edge features."""

    def __init__(self, node_mlp, edge_mlp, in_features, out_features, num_heads, alpha=0.2):
        super().__init__()
        self.node_mlp = node_mlp
        self.edge_mlp = edge_mlp
        self.num_heads = num_heads
        self.alpha = alpha

        # Define the linear transformations for the attention mechanism
        self.Ws = nn.Linear(in_features, out_features * num_heads, bias=False)
        self.Wt = nn.Linear(in_features, out_features * num_heads, bias=False)
        self.We = nn.Linear(in_features, out_features * num_heads, bias=False)
        self.attn = nn.Parameter(torch.zeros(size=(1, num_heads, out_features)))
        nn.init.xavier_uniform_(self.attn.data, gain=1.414)

        self.normX = LayerNorm(in_features, eps=1e-5)
        self.normE = LayerNorm(in_features, eps=1e-5)

    def forward(self, emb_node, emb_edge, adj_mtx, node_mask, eps=0):
        # Get batch size and number of nodes
        B, N, _ = emb_node.shape
        _, _, _, E = emb_edge.shape

        # Create masks for nodes and edges
        x_mask = node_mask.unsqueeze(-1)
        e_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(1)

        # Apply masks to node and edge embeddings
        emb_node, emb_edge = emb_node * x_mask, emb_edge * e_mask

        # Linear transformations for attention mechanism
        h_s = self.Ws(emb_node).view(B, N, self.num_heads, -1)
        h_t = self.Wt(emb_node).view(B, N, self.num_heads, -1)
        h_e = self.We(emb_edge).view(B, N, N, self.num_heads, -1)

        # Compute attention scores
        attn_input = F.leaky_relu(h_s.unsqueeze(2) + h_t.unsqueeze(1) + h_e, negative_slope=self.alpha)
        attn_scores = torch.sum(self.attn * attn_input, dim=-1)
        attn_scores = torch.exp(attn_scores)
        attn_scores = attn_scores * adj_mtx * e_mask  # Incorporate adjacency matrix
        attn_scores = attn_scores / (attn_scores.sum(dim=2, keepdim=True) + 1e-6)

        # Update node features using the adjacency matrix and edge features
        # Node update with attention scores
        h_t_prime = h_t.view(B, N, self.num_heads, -1)
        new_emb_node = torch.einsum('bijh,bjhd->bid', attn_scores, h_t_prime)

        new_emb_node = new_emb_node.view(B, N, -1)
        new_emb_node = self.node_mlp(new_emb_node)

        # Combine node and edge attributes for edge update
        combined_attr = torch.cat([
            new_emb_node.unsqueeze(2).repeat(1, 1, N, 1),
            new_emb_node.unsqueeze(1).repeat(1, N, 1, 1),
            emb_edge
        ], dim=-1)

        # Update edge features
        new_emb_edge = self.edge_mlp(combined_attr)
        emb_node = self.normX(emb_node + new_emb_node)
        emb_edge = self.normE(emb_edge + new_emb_edge)

        return emb_node, emb_edge

class DenseGATNet(nn.Module):
    """Dense GAT Network with edge feature processing"""

    def __init__(self, n_layers, input_dims, hidden_mlp_dims, hidden_dims, output_dims,
                 act_fn_in=nn.ReLU(), act_fn_out=nn.ReLU(), heads=3, dropout=0.0):
        super(DenseGATNet, self).__init__()

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

        self.gat_layers = nn.ModuleList()

        # Define GAT layers
        for _ in range(n_layers):
            node_mlp = nn.Sequential(
                nn.Linear(hidden_dims['dx'], hidden_dims['dx']),
                act_fn_in
            )
            edge_mlp = nn.Sequential(
                nn.Linear(2 * hidden_dims['dx'] + hidden_dims['de'], hidden_dims['de']),
                act_fn_in
            )
            self.gat_layers.append(DenseGATLayer(
                node_mlp=node_mlp,
                edge_mlp=edge_mlp,
                in_features=hidden_dims['dx'],
                out_features=hidden_dims['dx'],
                num_heads=heads,
                alpha=0.2
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

    def forward(self, X, E, node_mask):
        bs, n = X.shape[:2]

        # Create adjacency matrix
        adj_mtx = (torch.sum(E, axis=-1) != 0).unsqueeze(-1)

        # Apply input MLPs to node and edge features
        X = self.mlp_in_X(X)
        E = (self.mlp_in_E(E) + self.mlp_in_E(E).transpose(1, 2)) / 2

        # Apply GAT layers
        for layer in self.gat_layers:
            X, E = layer(emb_node=X, emb_edge=E, adj_mtx=adj_mtx, node_mask=node_mask)
            X = F.relu(X)

        # Apply output MLPs to node and edge features
        X = self.mlp_out_X(X)
        E = self.mlp_out_E(E)
        return utils.PlaceHolder(X=X, E=(E + E.transpose(1, 2)) / 2, y=None).mask(node_mask)
