import os
import pathlib
import random
import networkx as nx
from scipy.sparse import csr_matrix
import numpy as np
import scipy as sp
from collections import defaultdict
from typing import List, Tuple
import copy
from torch_geometric.utils import to_scipy_sparse_matrix

import torch
from torch_geometric.data.separate import separate
import torch_geometric as pyg
from torch_sparse import cat as sparse_cat
from torch_geometric.data import InMemoryDataset, download_url, Data, Batch
from torch_geometric.data.collate import collate
from torch_geometric.transforms import ToSparseTensor
from torch_geometric.transforms import BaseTransform
from torch_geometric.typing import SparseTensor
import torch.nn.functional as F

from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
from src.datasets.spectre_dataset import SpectreGraphDataModule, SpectreGraphDataset
from src.datasets.graph_generator import graph_generator, generate_graphs_for_sizes

from src.graph_coarsen.reduction import ReductionFactory
from src.graph_coarsen.spectral import SpectrumExtractor
from src.datasets.spectre_dataset import SpectreDatasetInfos
import src.utils as utils

class StochasticCoarsen(BaseTransform):
    def __init__(self, red_factory: ReductionFactory, spectrum_extractor: SpectrumExtractor, latent_graph_size: int):
        self.red_factory = red_factory
        self.spectrum_extractor = spectrum_extractor
        self.latent_graph_size = latent_graph_size

    def __call__(self, data):
        to_sparse = ToSparseTensor(remove_edge_index=False)
        data = to_sparse(data)
        graph = self.red_factory(data.adj_t.to_scipy())
        rng = np.random.default_rng()
        reduced_graph_data = self.get_random_reduction_sequence(graph, rng)
        if reduced_graph_data[-1].expansion_matrix.size(1) == self.latent_graph_size:
            reduced_graph_data = self.onestep_red_graph(reduced_graph_data)
            for key in set(data.keys):
                if key not in ['adj_t']:
                    reduced_graph_data[key] = data[key]
            x_reduced = reduced_graph_data['expansion_matrix'].to_dense().transpose(0, 1).to(
                dtype=reduced_graph_data['x'].dtype) @ reduced_graph_data['x']
            reduced_graph_data['x_reduced'] = (x_reduced > 0).to(dtype=reduced_graph_data['x'].dtype)
        else:
            raise NotImplementedError("Dataset size mismatch")
        return reduced_graph_data

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
        c_node_expansion = c_expansion_matrix.sum(0)
        spec_feature_reudced = torch.tensor(reduced_graph_seq[-1].spectral_features_reduced, dtype=torch.float32) \
            if self.spectrum_extractor is not None else None

        onestep_graph = ReducedGraphData(
                    target_size=reduced_graph_seq[0].target_size,
                    reduction_level=reduced_graph_seq[0].reduction_level,
                    adj=reduced_graph_seq[0].adj,
                    node_expansion = reduced_graph_seq[0].node_expansion,
                    reduced_node_expansion=c_node_expansion,
                    adj_reduced=reduced_graph_seq[-1].adj_reduced,
                    expansion_matrix=c_expansion_matrix,
                    spectral_features_reduced=spec_feature_reudced,
                )
        return onestep_graph


class DeterministicCoarsen(StochasticCoarsen):
    def __init__(self, red_factory, spectrum_extractor, latent_graph_size, seed: int):
        super().__init__(red_factory, spectrum_extractor, latent_graph_size)
        self.seed = seed

    def __call__(self, data):
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        to_sparse = ToSparseTensor(remove_edge_index=False)
        data = to_sparse(data)
        graph = self.red_factory(data.adj_t.to_scipy())
        rng = np.random.default_rng(self.seed)
        reduced_graph_data = self.get_random_reduction_sequence(graph, rng)
        if reduced_graph_data[-1].expansion_matrix.size(1) == self.latent_graph_size:
            reduced_graph_data = self.onestep_red_graph(reduced_graph_data)
            for key in set(data.keys):
                if key not in ['adj_t']:
                    reduced_graph_data[key] = data[key]
            x_reduced = reduced_graph_data['expansion_matrix'].to_dense().transpose(0, 1).to(
                dtype=reduced_graph_data['x'].dtype) @ reduced_graph_data['x']
            reduced_graph_data['x_reduced'] = (x_reduced > 0).to(dtype=reduced_graph_data['x'].dtype)
        else:
            raise NotImplementedError("Dataset size mismatch")
        return reduced_graph_data


class ReducedGraphData(Data):
    def __init__(self, **kwargs):
        if not kwargs:
            super().__init__()
            return

        if isinstance(kwargs["adj"], SparseTensor):
            super().__init__(x=torch.zeros(kwargs["adj"].size(dim=0)))
        else:
            super().__init__(x=torch.zeros(kwargs["adj"].shape[0]))

        for key, value in kwargs.items():
            if value is None:
                continue
            elif isinstance(value, int):
                value = torch.tensor(value).type(torch.long)
            elif isinstance(value, np.ndarray):
                value = torch.from_numpy(value).type(
                    torch.float32 if np.issubdtype(value.dtype, np.floating) else torch.long
                )
            elif isinstance(value, sp.sparse.sparray):
                value = SparseTensor.from_scipy(value).type(
                    torch.float32 if np.issubdtype(value.dtype, np.floating) else torch.long
                )
            elif isinstance(value, SparseTensor):
                value = value.type(
                    torch.float32 if value.dtype() in [torch.float32, torch.float64] else torch.long
                )
            elif isinstance(value, torch.Tensor):
                value = value.type(
                    torch.float32 if torch.is_floating_point(value) else torch.long
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
        if key in ['adj','adj_reduced', 'adj_t', 'expansion_matrix']:
            return (0, 1)  # concatenate along diagonal
        return super().__cat_dim__(key, value, *args, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        if key in ['adj', 'adj_t', 'adj_reduced']:
            return self.num_nodes
        else:
            return 0


class CoarsenedSpectreGraphDataset(SpectreGraphDataset):
    def __init__(self, cfg, dataset_name, split, root,
                 transform=None, pre_transform=None, pre_filter=None,
                 force_process=True):
        self.coarsening = cfg.reduction.coarsening
        self.red_factory = ReductionFactory(
            contraction_family=cfg.reduction.contraction_family,
            cost_type=cfg.reduction.cost_type,
            preserved_eig_size=cfg.reduction.preserved_eig_size,
            sqrt_partition_size=cfg.reduction.sqrt_partition_size,
            weighted_reduction=cfg.reduction.weighted_reduction,
            min_red_frac=cfg.reduction.min_red_frac,
            max_red_frac=cfg.reduction.max_red_frac,
            red_threshold=cfg.reduction.red_threshold,
            latent_graph_size=cfg.reduction.latent_graph_size,
            rand_lambda=cfg.reduction.rand_lambda,
        )

        self.spectrum_extractor = (
            SpectrumExtractor(
                num_features=cfg.spectral.num_features,
                normalized=cfg.spectral.normalized_laplacian,
            )
            if cfg.spectral.num_features > 0 else None
        )
        self.latent_graph_size = cfg.reduction.latent_graph_size

        super().__init__(dataset_name, split, root, transform,
                         pre_transform, pre_filter, force_process)

        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        print(self.data)

    def get(self, idx):
        # If there's only one data point, return a copy directly
        if self.len() == 1:
            return copy.copy(self._data)

        # Check if data list is already populated
        if not hasattr(self, '_data_list') or self._data_list is None:
            self._data_list = self.len() * [None]
        elif self._data_list[idx] is not None:
            return copy.copy(self._data_list[idx])

        # Extract and separate the data corresponding to idx
        data = separate(
            cls=self._data.__class__,
            batch=self._data,
            idx=idx,
            slice_dict=self.slices,
            decrement=False,
        )

        # Handle adjacency matrix and related sparse attributes
        if hasattr(data, 'adj'):
            start, end = self.slices['adj'][idx][0], self.slices['adj'][idx+1][0]
            data.adj = self._data.adj[start:end, start:end]  # Adjust as per your dataset's specific slicing logic
        if hasattr(data, 'adj_t'):
            start, end = self.slices['adj_t'][idx][0], self.slices['adj_t'][idx + 1][0]
            data.adj_t = self._data['adj_t'][start:end, start:end]
        if hasattr(data, 'adj_reduced'):
            start, end = self.slices['adj_reduced'][idx][0], self.slices['adj_reduced'][idx + 1][0]
            data.adj_reduced = self._data['adj_reduced'][start:end, start:end]
        if hasattr(data, 'expansion_matrix'):
            start, end = self.slices['expansion_matrix'][idx], self.slices['expansion_matrix'][idx + 1]
            data.expansion_matrix = self._data['expansion_matrix'][start[0]:end[0], start[1]:end[1]]

        # Cache the result in _data_list for faster future access
        self._data_list[idx] = copy.copy(data)
        return data

    def process(self):
        if self.dataset_name in ['comm_pos', 'comm_neg', 'sbm_pos', 'sbm_neg', 'coarsen_comm_pos']:
            num_graphs = {'train': 120, 'val': 40, 'test': 40}
            if self.dataset_name == 'comm_pos' or self.dataset_name == 'coarsen_comm_pos':
                prob_matrix = [[0.3, 0.01], [0.01, 0.3]]
            elif self.dataset_name == 'comm_neg':
                prob_matrix = [[0.01, 0.3], [0.3, 0.01]]
            elif self.dataset_name == 'sbm_pos':
                prob_matrix = [[0.3, 0.01, 0.01, 0.01, 0.01], [0.01, 0.3, 0.01, 0.01, 0.01],
                               [0.01, 0.01, 0.3, 0.01, 0.01], [0.01, 0.01, 0.01, 0.3, 0.01],
                               [0.01, 0.01, 0.01, 0.01, 0.3]]
            elif self.dataset_name == 'sbm_neg':
                prob_matrix = [[0.01, 0.2, 0.2, 0.2, 0.2], [0.2, 0.01, 0.2, 0.2, 0.2],
                               [0.2, 0.2, 0.01, 0.2, 0.2], [0.2, 0.2, 0.2, 0.01, 0.2],
                               [0.2, 0.2, 0.2, 0.2, 0.01]]
            else:
                raise ValueError(f'Unknown dataset {self.dataset_name}')
            data_list = []
            random.seed(self.seed)
            for _ in range(num_graphs[self.split]):
                if self.dataset_name in ['coarsen_comm_pos', 'coarsen_comm_neg']:
                    min_size, max_size = 10, 20
                    size = random.randint(min_size, max_size)
                    sizes = 2 * [size // 2]  # Example: Two communities of equal size
                elif self.dataset_name in ['sbm_pos', 'sbm_neg']:
                    min_size, max_size = 120, 200
                    size = random.randint(min_size, max_size)
                    sizes = 5 * [size // 5]
                else:
                    raise ValueError(f'Unknown dataset {self.dataset_name}')
                data, _ = graph_generator(sizes, prob_matrix, self.seed)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)
            torch.save(self.collate(data_list), self.processed_paths[0])
            return 0

        file_idx = {'train': 0, 'val': 1, 'test': 2}
        raw_dataset = torch.load(self.raw_paths[file_idx[self.split]], weights_only=False)

        data_list = []
        for adj in raw_dataset:
            n = adj.shape[-1]
            X = torch.ones(n, 1, dtype=torch.float)
            y = torch.zeros([1, 0]).float()
            if self.dataset_name == 'coarsen_tree':
                adj_numeric = np.asarray(adj, dtype=np.float32)  # Ensure it's a numeric array.
                adj = torch.from_numpy(adj_numeric)
            edge_index, _ = pyg.utils.dense_to_sparse(adj)
            edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
            edge_attr[:, 1] = 1
            num_nodes = n * torch.ones(1, dtype=torch.long)
            data = pyg.data.Data(x=X, edge_index=edge_index, edge_attr=edge_attr,
                                             y=y, n_nodes=num_nodes)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[0])

    def graph_coarsen(self, data):
        to_sparse_tensor = ToSparseTensor(remove_edge_index=False)

        data = to_sparse_tensor(data)
        graph = self.red_factory(data.adj_t.to_scipy())
        rng = np.random.default_rng(0)
        reduced_graph_data = self.get_random_reduction_sequence(graph, rng)

        if reduced_graph_data[-1].expansion_matrix.size(1) == self.latent_graph_size:
            reduced_graph_data = self.onestep_red_graph(reduced_graph_data)
            for key in set(data.keys):
                if key not in ['adj_t']:
                    reduced_graph_data[key] = data[key]
            x_reduced = reduced_graph_data['expansion_matrix'].to_dense().transpose(0,1).to(dtype=reduced_graph_data['x'].dtype) @ reduced_graph_data['x']
            reduced_graph_data['x_reduced'] = (x_reduced > 0).to(dtype=reduced_graph_data['x'].dtype)
        else:
            raise NotImplementedError("Dataset size mismatch")
        return reduced_graph_data


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
        c_node_expansion = c_expansion_matrix.sum(0)
        spec_feature_reudced = torch.tensor(reduced_graph_seq[-1].spectral_features_reduced, dtype=torch.float32) \
            if self.spectrum_extractor is not None else None

        onestep_graph = ReducedGraphData(
                    target_size=reduced_graph_seq[0].target_size,
                    reduction_level=reduced_graph_seq[0].reduction_level,
                    adj=reduced_graph_seq[0].adj,
                    node_expansion = reduced_graph_seq[0].node_expansion,
                    reduced_node_expansion=c_node_expansion,
                    adj_reduced=reduced_graph_seq[-1].adj_reduced,
                    expansion_matrix=c_expansion_matrix,
                    spectral_features_reduced=spec_feature_reudced,
                )
        return onestep_graph


class CoarsenedSpectreGraphDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)

        red_factory = ReductionFactory(
            contraction_family=cfg.reduction.contraction_family,
            cost_type=cfg.reduction.cost_type,
            preserved_eig_size=cfg.reduction.preserved_eig_size,
            sqrt_partition_size=cfg.reduction.sqrt_partition_size,
            weighted_reduction=cfg.reduction.weighted_reduction,
            min_red_frac=cfg.reduction.min_red_frac,
            max_red_frac=cfg.reduction.max_red_frac,
            red_threshold=cfg.reduction.red_threshold,
            latent_graph_size=cfg.reduction.latent_graph_size,
            rand_lambda=cfg.reduction.rand_lambda,
        )
        self.red_factory = red_factory
        spectrum_extractor = (
            SpectrumExtractor(
                num_features=cfg.spectral.num_features,
                normalized=cfg.spectral.normalized_laplacian,
            )
            if cfg.spectral.num_features > 0 else None
        )
        self.spectrum_extractor = spectrum_extractor
        latent_size = cfg.reduction.latent_graph_size

        stochastic_transform = StochasticCoarsen(red_factory, spectrum_extractor, latent_size)
        deterministic_transform = DeterministicCoarsen(red_factory, spectrum_extractor, latent_size, seed=42)

        datasets = {'train': CoarsenedSpectreGraphDataset(cfg = cfg,
                                                          dataset_name=self.cfg.dataset.name,
                                                          split='train', root=root_path,
                                                          transform=stochastic_transform),
                    'val': CoarsenedSpectreGraphDataset(cfg=cfg,
                                                        dataset_name=self.cfg.dataset.name,
                                                        split='val', root=root_path,
                                                        transform=deterministic_transform),
                    'test': CoarsenedSpectreGraphDataset(cfg=cfg,
                                                         dataset_name=self.cfg.dataset.name,
                                                         split='test', root=root_path,
                                                         transform=deterministic_transform)}

        super().__init__(cfg, datasets, collate_fn=custom_data_collate)

        self.inner = self.train_dataset


def custom_collate(data_list):
    r"""Collates a list of :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData` objects to the internal
    storage format of :class:`~torch_geometric.data.InMemoryDataset`.
    """
    print("calling custom_collate function")
    if len(data_list) == 1:
        data = data_list[0]
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=data.x.device)
        data.compute_batch_reduced()
        return data, None
        # return data_list[0], None

    data, slices, _ = collate(
        data_list[0].__class__,
        data_list=data_list,
        increment=True,
        add_batch=True,
    )
    return data, slices

def custom_data_collate(data_list):
    r"""Collates a list of :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData` objects to the internal
    storage format of :class:`~torch_geometric.data.InMemoryDataset`.
    """
    if len(data_list) == 1:
        data = data_list[0]
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=data.x.device)
        data.compute_batch_reduced()
        return data
        # return data_list[0]

    data, slices, _ = collate(
        data_list[0].__class__,
        data_list=data_list,
        increment=True,
        add_batch=True,
        exclude_keys = ['edge_index']
    )

    # Stupid way... just manually shift the edge index by num_nodes.
    repeats = [
        item.num_nodes
        for item in data_list
    ]
    if isinstance(repeats[0], torch.Tensor):
        repeats = torch.stack(repeats, dim=0)
    else:
        repeats = torch.tensor(repeats)
    incs = torch.cat([torch.tensor([0]),torch.cumsum(repeats[:-1], dim=0)])
    edge_indices = [item.edge_index for item in data_list]
    if incs.dim() > 1 or int(incs[-1]) != 0:
        edge_indices = [
            edge_index + inc.to(edge_index.device)
            for edge_index, inc in zip(edge_indices, incs)
        ]
    data.edge_index = torch.cat(edge_indices, dim=1)
    data.compute_batch_reduced()
    return data


class CoarsenedSpectreDatasetInfos(SpectreDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.name = 'nx_graphs'
        if self.datamodule.cfg.reduction.coarsening:
            self.original_n_nodes, self.n_nodes = self.datamodule.node_counts()
        else:
            self.n_nodes = self.datamodule.node_counts()

        if dataset_config['name'] in ['coarsen_sbm', 'coarsen_comm_pos', 'coarsen_comm20', 'coarsen_planar', 'coarsen_tree']:
            # Get the worst-case maximum expansion size
            N = self.original_n_nodes.size(0) - 1
            L = self.n_nodes.size(0) - 1
            rand_lambda = min(self.datamodule.cfg.reduction.rand_lambda, 0.9999)
            self.max_expansion_size = int((N / L) / (1.0 - rand_lambda))
            # self.max_expansion_size = 32
        else:
            maxes = []
            for batch in datamodule.train_dataloader():
                maxes.append(batch.reduced_node_expansion.max().item())
            self.max_expansion_size = int(max(maxes))

        if dataset_config['name'] in ['comm_pos', 'comm_neg']:
            self.node_types = torch.tensor([0.5, 0.5])
        elif dataset_config['name'] in ['sbm_pos', 'sbm_neg']:
            self.node_types = torch.tensor(5 * [0.2])
        elif dataset_config['name'] in ['coarsen_sbm', 'coarsen_comm_pos', 'coarsen_comm20', 'coarsen_planar', 'coarsen_tree']:
            self.node_types = torch.tensor(self.get_node_types())
        else:
            self.node_types = torch.tensor([1])               # There are no node types
        self.edge_types = self.datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)


    def get_node_types(self):
        """
        max_expansion_size = [node_exp.max()
                              for node_exp in [self.datamodule.train_dataset.reduced_node_expansion,
                                               self.datamodule.val_dataset.reduced_node_expansion,
                                               self.datamodule.test_dataset.reduced_node_expansion]]
        """
        num_classes = self.max_expansion_size

        counts = torch.zeros(num_classes)

        for i, data in enumerate(self.datamodule.train_dataloader()):
            counts += torch.bincount(data.reduced_node_expansion.to(torch.long)-1, minlength=num_classes)

        counts = counts / counts.sum()
        print(f"Counting node types: {counts}")
        return counts

    def compute_input_output_dims(self, datamodule, extra_features, domain_features):
        example_batch = next(iter(datamodule.train_dataloader()))
        node_expansion = F.one_hot(
            example_batch.reduced_node_expansion.to(torch.long) - 1,
            num_classes=self.max_expansion_size
        )

        ex_dense, node_mask = utils.to_dense_coarsen(
            node_expansion,
            example_batch.adj_reduced,
            example_batch.batch_reduced
        )

        ex_dense_og, node_mask_og = utils.to_dense(example_batch.x, example_batch.edge_index,
                                             example_batch.edge_attr, example_batch.batch)

        print(ex_dense.X.shape,ex_dense.E.shape, node_mask.shape)
        if datamodule.cfg.model.uncon_gen is False:
            example_data = {'X_t': ex_dense.X, 'E_t': ex_dense.E, 'y_t': example_batch['y'], 'node_mask': node_mask}

            self.input_dims = {'X': ex_dense.X.size(-1),
                               'E': ex_dense.E.size(-1),
                               'y': example_data['y'].size(1) + 1}  # + 1 due to time conditioning
            ex_extra_feat = extra_features(example_data)
            self.input_dims['X'] += ex_extra_feat.X.size(-1)
            self.input_dims['E'] += ex_extra_feat.E.size(-1)
            self.input_dims['y'] += ex_extra_feat.y.size(-1)

            ex_extra_molecular_feat = domain_features(example_data)
            self.input_dims['X'] += ex_extra_molecular_feat.X.size(-1)
            self.input_dims['E'] += ex_extra_molecular_feat.E.size(-1)
            self.input_dims['y'] += ex_extra_molecular_feat.y.size(-1)

            self.output_dims = {'X': ex_dense.X.size(-1),
                                'E': ex_dense.E.size(-1),
                                'y': 0}

            self.output_dims_decoarse = {'X': ex_dense_og.X.size(-1),
                                'E': ex_dense_og.E.size(-1),
                                'y': 0}

        elif datamodule.cfg.model.uncon_gen is True:
            example_data = {'X_t': ex_dense.X, 'E_t': ex_dense.E, 'y_t': None, 'node_mask': node_mask}

            self.input_dims = {'X': ex_dense.X.size(-1),
                               'E': ex_dense.E.size(-1)}
            ex_extra_feat = extra_features(example_data)
            self.input_dims['X'] += ex_extra_feat.X.size(-1)
            self.input_dims['E'] += ex_extra_feat.E.size(-1)

            ex_extra_molecular_feat = domain_features(example_data)
            self.input_dims['X'] += ex_extra_molecular_feat.X.size(-1)
            self.input_dims['E'] += ex_extra_molecular_feat.E.size(-1)

            self.output_dims = {'X': ex_dense.X.size(-1),
                                'E': ex_dense.E.size(-1)}

            self.output_dims_decoarse = {'X': ex_dense_og.X.size(-1),
                                         'E': ex_dense_og.E.size(-1),
                                         'y': 0}

        else:
            raise NotImplementedError("Unknown model type {}".format(datamodule.cfg.model.uncon_gen))
        print(self.output_dims, self.input_dims)