from abc import ABC

import numpy as np
import scipy as sp
import torch as th
from torch.utils.data import IterableDataset
from torch_geometric.data import Data
from torch_geometric.typing import SparseTensor

from .reduction import ReductionFactory


class RandRedDataset(IterableDataset, ABC):
    def __init__(self, adjs, red_factory: ReductionFactory, spectrum_extractor, latent_graph_size):
        super().__init__()

        self.red_factory = red_factory
        self.adjs = adjs
        self.spectrum_extractor = spectrum_extractor
        self.latent_graph_size = latent_graph_size

    def get_random_reduction_sequence(self, graph, rng):
        # Construct a sequence of random graphs
        data = []
        while True:
            reduced_graph = graph.get_reduced_graph(rng)
            data.append(
                ReducedGraphData(
                    target_size=graph.n,
                    reduction_level=graph.level,
                    adj=graph.adj.astype(bool).astype(np.float32),
                    node_expansion=graph.node_expansion,
                    adj_reduced=reduced_graph.adj.astype(bool).astype(np.float32),
                    expansion_matrix=reduced_graph.expansion_matrix,
                    spectral_features_reduced=self.spectrum_extractor(reduced_graph.adj)
                    if self.spectrum_extractor is not None
                    else None,
                )
            )

            if (graph.n <= 1 or reduced_graph.n <= self.red_factory.latent_graph_size):
                break
            graph = reduced_graph
        return data

    def onestep_red_graph(self, reduced_graph_seq):
        """
        Given a coarsening graph sequence, output the corresponding one-step coarsening data.
        Specifically the expansion vector and the expansion matrix.
        Args:
            reduced_graph_seq:

        Returns:
            One step graph reduction.
        """
        # cumulated expansion matrix
        c_expansion_matrix = reduced_graph_seq[0].expansion_matrix
        for reduced_graph in reduced_graph_seq[1:]:
            c_expansion_matrix = c_expansion_matrix @ reduced_graph.expansion_matrix
        c_node_expansion = c_expansion_matrix.sum(dim=0)
        spec_feature_reudced = reduced_graph_seq[-1].spectral_features_reduced.to(th.float32) \
            if self.spectrum_extractor is not None else None

        onestep_graph = ReducedGraphData(
                    target_size=reduced_graph_seq[0].target_size,
                    reduction_level=reduced_graph_seq[0].reduction_level,
                    adj=reduced_graph_seq[0].adj.to(th.float32),
                    node_expansion = reduced_graph_seq[0].node_expansion,
                    reduced_node_expansion=c_node_expansion,
                    adj_reduced=reduced_graph_seq[-1].adj_reduced.to(th.float32),
                    expansion_matrix=c_expansion_matrix.to(th.float32),
                    spectral_features_reduced=spec_feature_reudced,
                )
        return onestep_graph


class FiniteRandRedDataset(RandRedDataset):
    def __init__(
        self, adjs, red_factory: ReductionFactory,
            spectrum_extractor,latent_graph_size, num_red_seqs
    ):
        super().__init__(adjs, red_factory, spectrum_extractor, latent_graph_size)
        self.num_red_seqs = num_red_seqs

        self.rng = np.random.default_rng(seed=0)
        self.graph_reduced_data = {i: [] for i in range(len(adjs))}
        for i, adj in enumerate(adjs):
            graph = red_factory(adj)
            for _ in range(num_red_seqs):
                self.graph_reduced_data[i] += self.get_random_reduction_sequence(
                    graph, self.rng
                )

    def __iter__(self):
        while True:
            i = self.rng.integers(len(self.adjs))
            j = self.rng.integers(len(self.graph_reduced_data[i]))
            yield self.graph_reduced_data[i][j]

    @property
    def max_node_expansion(self):
        return max(
            [
                rgd.node_expansion.max().item()
                for seq in self.graph_reduced_data
                for rgd in seq
            ]
        )


class InfiniteRandRedDataset(RandRedDataset):

    def __init__(self, adjs, red_factory: ReductionFactory,
                 spectrum_extractor, latent_graph_size):
        super().__init__(adjs, red_factory, spectrum_extractor, latent_graph_size)
        graphs = [self.red_factory(adj.copy()) for adj in self.adjs]

        # get process id
        rng = np.random.default_rng(0)
        # initialize graph_reduced_data
        self.graph_reduced_data = self.produce_onestep_data(graphs, rng)

    def __iter__(self):
        for i in range(len(self.graph_reduced_data)):  # Iterate over all indices
            yield self.graph_reduced_data[i]  # Yield one item at a time
    def __len__(self):
        return len(self.graph_reduced_data)  # Return the number of items

    def produce_onestep_data(self, graphs, rng):
        graph_reduced_data = {
            i: self.get_random_reduction_sequence(graph, rng)
            for i, graph in enumerate(graphs)
        }
        if self.latent_graph_size > 0:
            graph_reduced_data = {i: val for i, val in graph_reduced_data.items()
                                  if val[-1].expansion_matrix.size(1) == self.latent_graph_size}

            graph_reduced_data = {i: self.onestep_red_graph(graph)
                                  for i, graph in graph_reduced_data.items()}

        return graph_reduced_data

    @property
    def max_node_expansion(self):
        raise NotImplementedError


class ReducedGraphData(Data):
    def __init__(self, **kwargs):
        if not kwargs:
            super().__init__()
            return
        if isinstance(kwargs["adj"], SparseTensor):
            super().__init__(x=th.zeros(kwargs["adj"].size(dim=0)))
        else:
            super().__init__(x=th.zeros(kwargs["adj"].shape[0]))
        for key, value in kwargs.items():
            if value is None:
                continue
            elif isinstance(value, int):
                value = th.tensor(value).type(th.long)
            elif isinstance(value, np.ndarray):
                value = th.from_numpy(value).type(
                    th.float32 if np.issubdtype(value.dtype, np.floating) else th.long
                )
            elif isinstance(value, sp.sparse.sparray):
                value = SparseTensor.from_scipy(value).type(
                    th.float32 if np.issubdtype(value.dtype, np.floating) else th.long
                )
            elif isinstance(value, SparseTensor):
                value = value.type(
                    th.float32 if value.dtype() in [th.float32, th.float64] else th.long
                )
            elif isinstance(value, th.Tensor):
                value = value.type(
                    th.float32 if th.is_floating_point(value) else th.long
                )
            else:
                raise ValueError(f"Unsupported type {type(value)} for key {key}")

            setattr(self, key, value)

    def compute_batch_reduced(self):
        """
        Computes the batch_reduced tensor for the reduced graph.
        It maps batch indices from the full graph (batch) to the reduced graph.
        """
        if not hasattr(self, "expansion_matrix"):
            raise ValueError("expansion_matrix is required to compute batch_reduced")
        if not hasattr(self, "batch"):
            raise ValueError("batch is required to compute batch_reduced")

        num_nodes_vec = self.expansion_matrix.to_dense().sum(0)
        num_nodes_vec[num_nodes_vec == 0] = 1  # Set to 1 to avoid NaN/inf

        # Use the expansion matrix to aggregate batch indices
        # Sparse matrix multiplication of expansion_matrix^T * batch
        batch_full_expanded = self.expansion_matrix.to_dense().transpose(0, 1) @ self.batch.float()
        # Convert to integer indices
        batch_reduced = (batch_full_expanded/num_nodes_vec).round().long()
        self.batch_reduced = batch_reduced

    def __cat_dim__(self, key, value, *args, **kwargs):
        if isinstance(value, SparseTensor):
            return (0, 1)  # concatenate along diagonal
        return super().__cat_dim__(key, value, *args, **kwargs)
