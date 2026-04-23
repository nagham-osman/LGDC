import torch
from torch_scatter import scatter
from torch_sparse import SparseTensor


class Expansion:
    """Graph generation method generating graphs by local expansion."""

    def __init__(
        self,
        augmented_radius=1,
        augmented_dropout=0.0,
        deterministic_expansion=False,
    ):
        self.augmented_radius = augmented_radius
        self.augmented_dropout = augmented_dropout
        self.deterministic_expansion = deterministic_expansion
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def sample_graphs_with_initial(self, adj, batch, node_expansion, target_size,
                                   model, sign_net, spectrum_extractor, decoarsen_diffusion, emb_features,
                                   iterative=False):
        """Samples a batch of graphs with initial information."""

        if iterative:
            while adj.size(0) < target_size.sum():
                adj, batch, node_expansion = self.expand(
                    adj,
                    batch,
                    node_expansion,
                    target_size,
                    model=model,
                    sign_net=sign_net,
                    spectrum_extractor=spectrum_extractor,
                    decoarsen_diffusion=decoarsen_diffusion,
                    emb_features=emb_features,
                    iterative=iterative,
                )
                if node_expansion.max() <= 1:
                    break
        else:
            adj, batch, node_expansion = self.expand(
                adj,
                batch,
                node_expansion,
                target_size,
                model=model,
                sign_net=sign_net,
                spectrum_extractor=spectrum_extractor,
                decoarsen_diffusion=decoarsen_diffusion,
                emb_features=emb_features,
                iterative=iterative,
            )

        adjs = unbatch_adj(adj, batch)
        return adjs

    @torch.no_grad()
    def expand(
        self,
        adj_reduced,
        batch_reduced,
        node_expansion,
        target_size,
        model,
        sign_net,
        spectrum_extractor,
        decoarsen_diffusion,
        emb_features,
        iterative=False,
    ):
        """Expands a graph by a single level."""
        # the graph size in the batch (a vector of shape batch_size, the value is the number of nodes).
        reduced_size = scatter(torch.ones_like(batch_reduced), batch_reduced)

        # get node embeddings, of shape num_nodes * feature shape
        if spectrum_extractor is not None:
            spectral_features = torch.cat(
                [
                    torch.tensor(
                        spectrum_extractor(adj.to("cpu").to_scipy(layout="coo")),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    for adj in unbatch_adj(adj_reduced, batch_reduced)
                ]
            )
            node_emb_reduced = sign_net(
                spectral_features=spectral_features, edge_index=adj_reduced
            )
        else:
            node_emb_reduced = torch.randn(
                adj_reduced.size(0), emb_features, device=self.device
            )

        # expand
        # node_expansion:
        node_expansion[(reduced_size >= target_size)[batch_reduced]] = 1
        # the expansion map accordingly to node_expansion
        node_map = torch.repeat_interleave(
            torch.arange(0, adj_reduced.size(0), device=self.device), node_expansion
        )
        # node_map: sum of number of node sin the expanded graph
        node_emb = node_emb_reduced[node_map]
        batch = batch_reduced[node_map]
        size = scatter(torch.ones_like(batch), batch)
        expansion_matrix = SparseTensor(
            row=torch.arange(node_map.size(0), device=self.device),
            col=node_map,
            value=torch.ones(node_map.size(0), device=self.device),
        )
        adj_augmented = self.get_augmented_graph(adj_reduced, expansion_matrix)
        augmented_edge_index = (torch.stack(adj_augmented.coo()[:2], dim=0))

        if iterative:
            # compute number of nodes in expanded graph
            random_reduction_fraction = (
                    torch.rand(len(target_size), device=self.device)
                    * (0.3 - 0.1)
                    + 0.1
            )

            # if expanded number of nodes is less than threshold, use max_red_frac
            max_reduction_mask = (
                    torch.ceil(size / (1 - 0.3)) <= 16
            ).float()
            random_reduction_fraction = (
                                                1 - max_reduction_mask
                                        ) * random_reduction_fraction + max_reduction_mask * 0.3

            # expanded number of nodes is ⌈n / (1-r)⌉ and at least n+1 and at most target_size
            expanded_size = torch.minimum(
                torch.maximum(
                    torch.ceil(size / (1 - random_reduction_fraction)).long(),
                    size + 1,
                ),
                target_size,
            )
        else:
            expanded_size = target_size

        # make predictions
        node_pred, augmented_edge_pred = decoarsen_diffusion.sample(
            edge_index=augmented_edge_index,
            batch=batch,
            model=model,
            model_kwargs={
                "node_emb": node_emb,
                "red_frac": 1 - size / expanded_size,
                "target_size": target_size.float(),
            },
        )
        # get node attributes
        if self.deterministic_expansion:
            node_attr = torch.zeros_like(node_pred, dtype=torch.long)
            num_new_nodes = torch.maximum(
                expanded_size - size,
                torch.ones(len(target_size), dtype=torch.long, device=self.device)
            )
            node_range_end = size.cumsum(0)
            node_range_start = node_range_end - size
            # get top-k nodes per graph
            for i in range(len(target_size)):
                new_node_idx = (
                    torch.topk(
                        node_pred[node_range_start[i] : node_range_end[i]],
                        num_new_nodes[i],
                        largest=True,
                    )[1]
                    + node_range_start[i]
                )
                node_attr[new_node_idx] = 1
        else:
            node_attr = (node_pred > 0.5).long()

        # construct new graph
        adj = SparseTensor.from_edge_index(
            augmented_edge_index[:, augmented_edge_pred > 0.5],
            sparse_sizes=adj_augmented.sizes(),
        )

        return adj, batch, node_attr + 1


    def get_augmented_graph(self, adj_reduced, expansion_matrix):
        """Returns the expanded adjacency matrix with additional augmented edges.

        All edge weights are set to 1.
        """
        # construct augmented adjacency matrix
        adj_reduced = adj_reduced.set_diag(1)
        adj_reduced_augmented = adj_reduced.copy()

        for _ in range(1, self.augmented_radius):
            adj_reduced_augmented = adj_reduced_augmented @ adj_reduced

        adj_reduced_augmented = adj_reduced_augmented.set_value(
            torch.ones(adj_reduced_augmented.nnz(), device=self.device), layout="coo"
        )
        adj_augmented = (
            expansion_matrix @ adj_reduced_augmented @ expansion_matrix.t()
        ).remove_diag()
        adj_expanded = (
            expansion_matrix @ adj_reduced @ expansion_matrix.t()
        ).remove_diag()

        # drop out edges
        if self.augmented_dropout > 0.0:
            adj_required = adj_augmented + adj_expanded
            row, col, val = adj_required.coo()
            edge_mask = torch.rand_like(val) >= self.augmented_dropout
            edge_mask = edge_mask | (val > 1)  # keep required edges
            # make undirected
            edge_mask = edge_mask & (row < col)
            edge_index = torch.stack([row[edge_mask], col[edge_mask]], dim=0)
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            adj_augmented = SparseTensor.from_edge_index(
                edge_index,
                edge_attr=torch.ones(edge_index.shape[1], device=self.device),
                sparse_sizes=adj_augmented.sizes(),
            )

        return adj_augmented

def unbatch_adj(adj, batch) -> list:
    size = scatter(torch.ones_like(batch), batch)
    graph_end_idx = size.cumsum(0)
    graph_start_idx = graph_end_idx - size
    return [
        adj[graph_start_idx[i] : graph_end_idx[i], :][
            :, graph_start_idx[i] : graph_end_idx[i]
        ]
        for i in range(len(size))
    ]
