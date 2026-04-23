import os
import torch_geometric.utils
from omegaconf import OmegaConf, open_dict
from torch_geometric.utils import to_dense_adj, to_dense_batch, dense_to_sparse
import torch.nn.functional as F
import torch
import omegaconf
import wandb
import numpy as np
import scipy.sparse as sp
from torch_sparse import SparseTensor, cat as sparse_cat
from torch_geometric.data import Data


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def create_folders(args):
    try:
        # os.makedirs('checkpoints')
        os.makedirs('graphs')
        os.makedirs('chains')
    except OSError:
        pass

    try:
        # os.makedirs('checkpoints/' + args.general.name)
        os.makedirs('graphs/' + args.general.name)
        os.makedirs('chains/' + args.general.name)
    except OSError:
        pass


def normalize(X, E, y, norm_values, norm_biases, node_mask):
    X = (X - norm_biases[0]) / norm_values[0]
    E = (E - norm_biases[1]) / norm_values[1]
    y = (y - norm_biases[2]) / norm_values[2] if y is not None else None

    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask)


def unnormalize(X, E, y, norm_values, norm_biases, node_mask, collapse=False):
    """
    X : node features
    E : edge features
    y : global features`
    norm_values : [norm value X, norm value E, norm value y]
    norm_biases : same order
    node_mask
    """
    X = (X * norm_values[0] + norm_biases[0])
    E = (E * norm_values[1] + norm_biases[1])
    y = y * norm_values[2] + norm_biases[2] if y is not None else None

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask, collapse)


def to_dense(x, edge_index, edge_attr, batch):
    X, node_mask = to_dense_batch(x=x.float(), batch=batch)
    edge_index, edge_attr = torch_geometric.utils.remove_self_loops(edge_index, edge_attr.float())
    max_num_nodes = X.size(1)
    E = to_dense_adj(edge_index=edge_index, batch=batch, edge_attr=edge_attr, max_num_nodes=max_num_nodes)
    E = encode_no_edge(E)
    return PlaceHolder(X=X, E=E, y=None), node_mask

def to_dense_coarsen(x, adj, batch):
    """
    adj: block-diagonal adjancancy matirx (which is a SparseTensor) of size (N_1+N_2+N_3 +...+ N_B)* (N_1+N_2+N_3 +...+ N_B)
    after batching (with B the batch_size),
    each N_i indicate the simple graph size.
    batch=[N_1+N_2+N_3 +...+ N_B] vector, indicating the batch of each nodes.
     -> construct a dense edge tensor of size B * max_num_nodes * max_num_nodes * 2 tensor,
    """
    device = batch.get_device() if batch.get_device() != -1 else 'cpu'


    X, node_mask = to_dense_batch(x=x.float(), batch=batch)
    # node_mask = node_mask.float()
    max_num_nodes = X.size(1)
    B = batch.max().item() + 1
    # Initialize the dense tensor
    E = torch.zeros((B, max_num_nodes, max_num_nodes, 2), dtype=torch.float32, device=device)
    row, col, value = adj.coo()

    # Get batch assignments for row and col indices
    row_batch = batch[row]
    col_batch = batch[col]

    # Ensure row and col indices belong to the same batch
    valid_indices = row_batch == col_batch
    row, col = row[valid_indices], col[valid_indices]
    row_batch = row_batch[valid_indices]

    # Offset row and col indices within each batch
    node_offsets = torch.cumsum(torch.cat([torch.tensor([0], device=device), batch.bincount()[:-1]]), 0)
    row_offset = row - node_offsets[row_batch]
    col_offset = col - node_offsets[col_batch]

    # Mask valid row/col indices within max_num_nodes
    valid_nodes = (row_offset < max_num_nodes) & (col_offset < max_num_nodes)
    row_offset = row_offset[valid_nodes]
    col_offset = col_offset[valid_nodes]
    row_batch = row_batch[valid_nodes]

    # Fill the edge tensor
    E[row_batch, row_offset, col_offset, 1] = 1  # Set edge existence to 1
    E[..., 0] = 1 - E[..., 1]  # Complementary value (non-edge)

    E = encode_no_edge(E)
    """
    for s in range(B):
        e = s+1
        diff = adj.to_dense()[s*max_num_nodes:e*max_num_nodes, s*max_num_nodes:e*max_num_nodes] - E[s:e, :, :, 1]
        assert torch.nonzero(diff).sum() ==0
    """
    return PlaceHolder(X=X, E=E, y=None), node_mask


def batch_sparse_tensors(sparse_tensors, node_feature_dim=1, edge_feature_dim=2):
    """
    Given a list of SparseTensor objects, create a batched version with:
      - x: node features,
      - edge_index: concatenated edge indices with proper node offset,
      - edge_attr: concatenated edge attributes,
      - batch: a tensor that assigns each node to its original graph.

    Returns:
        Data: A torch_geometric.data.Data object representing the batched graph.
    """
    device = sparse_tensors[0].coo()[0].device
    num_nodes_list = [sparse.size(0).item() for sparse in sparse_tensors]

    offsets = torch.cumsum(torch.tensor([0] + num_nodes_list, device=device), dim=0)

    edge_indices = []
    edge_attrs = []
    for i, sparse in enumerate(sparse_tensors):
        row, col, values = sparse.coo()
        row = row + offsets[i]
        col = col + offsets[i]
        edge_indices.append(torch.stack([row, col], dim=0))

        if values is None:
            default_value = torch.tensor([0, 1], device=device, dtype=torch.float)
            values = default_value.repeat(row.size(0), 1)
        else:
            if values.dim() == 1:
                values = values.unsqueeze(1)
            if values.size(1) != edge_feature_dim:
                values = values.repeat(1, edge_feature_dim)
        edge_attrs.append(values)

    edge_index = torch.cat(edge_indices, dim=1)
    edge_attr = torch.cat(edge_attrs, dim=0)

    batch = torch.repeat_interleave(
        torch.arange(len(sparse_tensors), device=device),
        torch.tensor(num_nodes_list, device=device)
    )

    x = torch.ones(sum(num_nodes_list), node_feature_dim, device=device)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
    return data


def to_sparse(X_dense, E_dense, node_mask):
    """
    Convert dense node and edge features to PyTorch Geometric's sparse format.

    Parameters:
        X_dense (torch.Tensor): Node features of shape [batch_size, max_num_nodes, F_node].
        E_dense (torch.Tensor): Edge features of shape [batch_size, max_num_nodes, max_num_nodes, F_edge].
        node_mask (torch.Tensor): Boolean tensor of shape [batch_size, max_num_nodes] indicating valid nodes.

    Returns:
        node_feature (torch.Tensor): Node features for valid nodes, shape [total_valid_nodes, F_node].
        edge_index (torch.LongTensor): Edge indices, shape [2, total_edges].
        edge_attr (torch.Tensor): Edge attributes, shape [total_edges, F_edge].
        batch (torch.LongTensor): Batch vector indicating graph membership for each node, shape [total_valid_nodes].
    """
    batch_size, max_num_nodes, F_node = X_dense.shape
    _, _, _, F_edge = E_dense.shape

    valid_node_indices = node_mask.nonzero(as_tuple=False)  # Shape: [total_valid_nodes, 2] (batch, node)
    total_valid_nodes = valid_node_indices.size(0)

    if total_valid_nodes == 0:
        raise ValueError("No valid nodes found in the node_mask.")

    node_feature = X_dense[valid_node_indices[:, 0], valid_node_indices[:, 1]]  # Shape: [total_valid_nodes, F_node]

    edge_nonzero = E_dense.nonzero(as_tuple=False)  # Shape: [total_edges, 4] (batch, src, dst, edge_class)

    if edge_nonzero.size(0) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=X_dense.device)
        edge_attr = torch.empty((0, F_edge), dtype=E_dense.dtype, device=X_dense.device)
    else:
        edge_batch = edge_nonzero[:, 0]       # Shape: [total_edges]
        src_nodes = edge_nonzero[:, 1]        # Shape: [total_edges]
        dst_nodes = edge_nonzero[:, 2]        # Shape: [total_edges]
        edge_classes = edge_nonzero[:, 3]     # Shape: [total_edges]

        num_valid_nodes_per_graph = node_mask.sum(dim=1)  # Shape: [batch_size]
        cum_num_valid_nodes = torch.cat([torch.tensor([0], device=X_dense.device),
                                        torch.cumsum(num_valid_nodes_per_graph, dim=0)], dim=0)  # Shape: [batch_size + 1]

        global_src = cum_num_valid_nodes[edge_batch] + src_nodes
        global_dst = cum_num_valid_nodes[edge_batch] + dst_nodes

        src_valid = node_mask[edge_batch, src_nodes]
        dst_valid = node_mask[edge_batch, dst_nodes]
        valid_edges_mask = src_valid & dst_valid

        global_src = global_src[valid_edges_mask]
        global_dst = global_dst[valid_edges_mask]
        edge_classes = edge_classes[valid_edges_mask]
        edge_batch = edge_batch[valid_edges_mask]

        edge_index = torch.stack([global_src, global_dst], dim=0)  # Shape: [2, total_valid_edges]

        edge_attr = torch.nn.functional.one_hot(edge_classes, num_classes=F_edge).float()  # Shape: [total_valid_edges, F_edge]

        edge_index, edge_attr = torch_geometric.utils.remove_self_loops(edge_index, edge_attr)

    batch_vector = valid_node_indices[:, 0]

    return node_feature, edge_index, edge_attr, batch_vector

def encode_no_edge(E):
    assert len(E.shape) == 4
    if E.shape[-1] == 0:
        return E
    no_edge = torch.sum(E, dim=3) == 0
    first_elt = E[:, :, :, 0]
    first_elt[no_edge] = 1
    E[:, :, :, 0] = first_elt
    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0
    return E


def update_config_with_new_keys(cfg, saved_cfg):
    saved_general = saved_cfg.general
    saved_train = saved_cfg.train
    saved_model = saved_cfg.model

    for key, val in saved_general.items():
        OmegaConf.set_struct(cfg.general, True)
        with open_dict(cfg.general):
            if key not in cfg.general.keys():
                setattr(cfg.general, key, val)

    OmegaConf.set_struct(cfg.train, True)
    with open_dict(cfg.train):
        for key, val in saved_train.items():
            if key not in cfg.train.keys():
                setattr(cfg.train, key, val)

    OmegaConf.set_struct(cfg.model, True)
    with open_dict(cfg.model):
        for key, val in saved_model.items():
            if key not in cfg.model.keys():
                setattr(cfg.model, key, val)
    return cfg

def mask_dist_edge(true_E, pred_E, node_mask):
    row_E = torch.zeros(true_E.size(-1), dtype=torch.float, device=true_E.device)
    row_E[0] = 1.
    diag_mask = ~torch.eye(node_mask.size(1), device=node_mask.device, dtype=torch.bool).unsqueeze(0)
    # true_E[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2) * diag_mask), :] = row_E
    pred_E[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2) * diag_mask), :] = row_E
    # true_E = true_E + 1e-7
    pred_E = pred_E + 1e-7
    # true_E = true_E / torch.sum(true_E, dim=-1, keepdim=True)
    pred_E = pred_E / torch.sum(pred_E, dim=-1, keepdim=True)

    # return true_E, pred_E
    return pred_E


# def batched_dense_to_sparse(batched_dense, batched_node_mask):
#     """
#     Converts a batched dense tensor into a block-diagonal SparseTensor.
#     Args:
#         batched_dense (torch.Tensor): Tensor of shape [B, N, N, edge_f].
#         batched_node_mask (torch.Tensor): Boolean tensor of shape [B, N] (True for valid nodes).
#     Returns:
#         SparseTensor: A block-diagonal SparseTensor constructed over valid nodes.
#     """
#     B, N, _, edge_f = batched_dense.shape
#
#     batched_dense = F.softmax(batched_dense, dim=-1)
#     batched_dense = torch.argmax(batched_dense, dim=-1)
#     batched_dense = F.one_hot(batched_dense, num_classes=edge_f).float()
#
#     all_rows = []
#     all_cols = []
#     all_vals = []
#     total_valid = 0
#
#     for i in range(B):
#         dense = batched_dense[i]
#         node_mask = batched_node_mask[i]
#
#         valid_idx = node_mask.nonzero(as_tuple=False).view(-1)
#         num_valid = valid_idx.size(0)
#
#         dense_valid = dense[valid_idx][:, valid_idx]
#
#         edge_mask = (dense_valid[..., 0] == 0) & (dense_valid[..., 1] == 1)
#         row, col = edge_mask.nonzero(as_tuple=True)
#
#         row = row + total_valid
#         col = col + total_valid
#
#         val = torch.ones(row.size(0), dtype=torch.float32, device=batched_dense.device)
#
#         all_rows.append(row)
#         all_cols.append(col)
#         all_vals.append(val)
#
#         total_valid += num_valid
#
#     if all_rows:
#         all_rows = torch.cat(all_rows, dim=0)
#         all_cols = torch.cat(all_cols, dim=0)
#         all_vals = torch.cat(all_vals, dim=0)
#     else:
#         all_rows = torch.tensor([], dtype=torch.long, device=batched_dense.device)
#         all_cols = torch.tensor([], dtype=torch.long, device=batched_dense.device)
#         all_vals = torch.tensor([], dtype=torch.float32, device=batched_dense.device)
#
#     total_nodes = total_valid
#     sparse_adj = SparseTensor(row=all_rows, col=all_cols, value=all_vals, sparse_sizes=(total_nodes, total_nodes))
#
#     return sparse_adj

def batched_dense_to_sparse(logits, node_mask):
    B, N, _, C = logits.shape
    rows, cols, vals = [], [], []
    base = 0
    for i in range(B):
        valid = node_mask[i].nonzero(as_tuple=False).view(-1)
        lab = logits[i].index_select(0, valid).index_select(1, valid).argmax(dim=-1)
        r, c = (lab == 1).nonzero(as_tuple=True)
        if r.numel():
            rows.append(r + base)
            cols.append(c + base)
            vals.append(torch.ones_like(r, dtype=torch.float32, device=logits.device))
        base += valid.numel()
    if rows:
        row = torch.cat(rows)
        col = torch.cat(cols)
        val = torch.cat(vals)
    else:
        row = torch.empty(0, dtype=torch.long, device=logits.device)
        col = torch.empty(0, dtype=torch.long, device=logits.device)
        val = torch.empty(0, dtype=torch.float32, device=logits.device)
    total = int(node_mask.sum().item())
    return SparseTensor(row=row, col=col, value=val, sparse_sizes=(total, total))


class PlaceHolder:
    def __init__(self, X, E, y):
        self.X = X
        self.E = E
        self.y = y

    def type_as(self, x: torch.Tensor):
        """ Changes the device and dtype of X, E, y. """
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x) if self.y is not None else None
        return self

    def mask(self, node_mask, collapse=False):
        x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)             # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)             # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)

            self.X[node_mask == 0] = - 1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        return self


def setup_wandb(cfg):
    config_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    kwargs = {'name': cfg.general.name, 'project': f'graph_ddm_{cfg.dataset.name}', 'config': config_dict,
              'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': cfg.general.wandb}
    wandb.init(**kwargs)
    wandb.save('*.txt')