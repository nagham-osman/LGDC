import os
import pathlib
import random
import networkx as nx
import numpy as np

import torch
from torch.utils.data import random_split
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset, download_url, Data

from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
from src.datasets.graph_generator import graph_generator, generate_graphs_for_sizes



class SpectreGraphDataset(InMemoryDataset):
    def __init__(self, dataset_name, split, root,
                 transform=None, pre_transform=None, pre_filter=None,
                 force_process=True):
        self.sbm_file = 'sbm_200.pt'
        self.planar_file = 'planar_64_200.pt'
        self.comm20_file = 'community_12_21_100.pt'
        self.dataset_name = dataset_name
        self.split = split
        self.num_graphs = 200
        self.seed=0
        super().__init__(root, transform, pre_transform, pre_filter, force_process)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    @property
    def processed_file_names(self):
            return [self.split + '.pt']

    def download(self):
        """
        Download raw qm9 files. Taken from PyG QM9 class
        """
        if self.dataset_name in ['sbm', 'coarsen_sbm']:
            raw_url = 'https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/sbm_200.pt'
        elif self.dataset_name in ['planar', 'coarsen_planar']:
            raw_url = 'https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/planar_64_200.pt'
        elif self.dataset_name in ['comm20', 'coarsen_comm20']:
            raw_url = 'https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/community_12_21_100.pt'
        elif self.dataset_name in ['comm_pos', 'comm_neg', 'sbm_pos', 'sbm_neg', 'coarsen_comm_pos']:
            return 0
        elif self.dataset_name == 'coarsen_tree':
            rng = np.random.default_rng(seed=0)
            graphs = []
            adj_matrices = []

            for _ in range(200):
                # n = rng.integers(5, 30, endpoint=True)
                n = 64
                G = nx.random_tree(n, seed=rng)
                graphs.append(G)
                adj_matrix = nx.to_numpy_array(G)
                adj_matrices.append(adj_matrix)

            save_path = os.path.join(self.raw_dir, 'tree.npz')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            adj_matrices_array = np.array(adj_matrices, dtype=object)
            np.savez(save_path, *adj_matrices_array)
        else:
            raise ValueError(f'Unknown dataset {self.dataset_name}')

        if self.dataset_name == 'coarsen_tree':
            file_path = os.path.join(self.raw_dir, 'tree.npz')
            loaded_data = np.load(file_path, allow_pickle=True)
            adjs = [loaded_data[key] for key in loaded_data.files]
        else:
            file_path = download_url(raw_url, self.raw_dir)
            adjs, eigvals, eigvecs, n_nodes, max_eigval, min_eigval, same_sample, n_max = torch.load(file_path)

        g_cpu = torch.Generator()
        g_cpu.manual_seed(self.seed)

        test_len = int(round(self.num_graphs * 0.2))
        train_len = int(round((self.num_graphs - test_len) * 0.8))
        val_len = self.num_graphs - train_len - test_len
        indices = torch.randperm(self.num_graphs, generator=g_cpu)
        print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')
        train_indices = indices[:train_len]
        val_indices = indices[train_len:train_len + val_len]
        test_indices = indices[train_len + val_len:]

        train_data = []
        val_data = []
        test_data = []

        for i, adj in enumerate(adjs):
            if i in train_indices:
                train_data.append(adj)
            elif i in val_indices:
                val_data.append(adj)
            elif i in test_indices:
                test_data.append(adj)
            else:
                raise ValueError(f'Index {i} not in any split')

        torch.save(train_data, self.raw_paths[0])
        torch.save(val_data, self.raw_paths[1])
        torch.save(test_data, self.raw_paths[2])


    def process(self):
        if self.dataset_name in ['comm_pos', 'comm_neg', 'sbm_pos', 'sbm_neg']:
            num_graphs = {'train': 120, 'val': 40, 'test': 40}
            if self.dataset_name == 'comm_pos':
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
                if self.dataset_name in ['comm_pos', 'comm_neg']:

                    sizes = 2 * [size // 2]  # Example: Two communities of equal size
                elif self.dataset_name in ['sbm_pos', 'sbm_neg']:
                    min_size, max_size = 120, 200
                    size = random.randint(min_size, max_size)
                    sizes = 5 * [size // 5]
                else:
                    raise ValueError(f'Unknown dataset {self.dataset_name}')
                data, _ = graph_generator(sizes, prob_matrix, self.seed)
                data_list.append(data)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
            torch.save(self.collate(data_list), self.processed_paths[0])
            return 0

        file_idx = {'train': 0, 'val': 1, 'test': 2}
        raw_dataset = torch.load(self.raw_paths[file_idx[self.split]])

        data_list = []
        for adj in raw_dataset:
            n = adj.shape[-1]
            X = torch.ones(n, 1, dtype=torch.float)
            y = torch.zeros([1, 0]).float()
            edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
            edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
            edge_attr[:, 1] = 1
            num_nodes = n * torch.ones(1, dtype=torch.long)
            data = torch_geometric.data.Data(x=X, edge_index=edge_index, edge_attr=edge_attr,
                                             y=y, n_nodes=num_nodes)
            data_list.append(data)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[0])



class SpectreGraphDataModule(AbstractDataModule):
    def __init__(self, cfg, n_graphs=200):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)


        datasets = {'train': SpectreGraphDataset(dataset_name=self.cfg.dataset.name,
                                                 split='train', root=root_path),
                    'val': SpectreGraphDataset(dataset_name=self.cfg.dataset.name,
                                        split='val', root=root_path),
                    'test': SpectreGraphDataset(dataset_name=self.cfg.dataset.name,
                                        split='test', root=root_path)}
        # print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')

        super().__init__(cfg, datasets)
        self.inner = self.train_dataset

    def __getitem__(self, item):
        return self.inner[item]



class SpectreDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.name = 'nx_graphs'
        self.n_nodes = self.datamodule.node_counts()
        if dataset_config['name'] in ['comm_pos', 'comm_neg']:
            self.node_types = torch.tensor([0.5, 0.5])
        elif dataset_config['name'] in ['sbm_pos', 'sbm_neg']:
            self.node_types = torch.tensor(5 * [0.2])
        else:
            self.node_types = torch.tensor([1])               # There are no node types
        self.edge_types = self.datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)
