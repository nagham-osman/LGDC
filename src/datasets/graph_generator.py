import random

import networkx as nx
import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data, Batch
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union
from collections import defaultdict


def from_networkx(
    G: Any,
    group_node_attrs: Optional[Union[List[str], Literal['all']]] = None,
    group_edge_attrs: Optional[Union[List[str], Literal['all']]] = None,
) -> 'torch_geometric.data.Data':

    G = G.to_directed() if not nx.is_directed(G) else G

    mapping = dict(zip(G.nodes(), range(G.number_of_nodes())))
    edge_index = torch.empty((2, G.number_of_edges()), dtype=torch.long)
    for i, (src, dst) in enumerate(G.edges()):
        edge_index[0, i] = mapping[src]
        edge_index[1, i] = mapping[dst]

    data_dict: Dict[str, Any] = defaultdict(list)
    data_dict['edge_index'] = edge_index

    node_attrs: List[str] = []
    if G.number_of_nodes() > 0:
        node_attrs = list(next(iter(G.nodes(data=True)))[-1].keys())

    edge_attrs: List[str] = []
    if G.number_of_edges() > 0:
        edge_attrs = list(next(iter(G.edges(data=True)))[-1].keys())

    if group_node_attrs is not None and not isinstance(group_node_attrs, list):
        group_node_attrs = node_attrs

    if group_edge_attrs is not None and not isinstance(group_edge_attrs, list):
        group_edge_attrs = edge_attrs

    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        if set(feat_dict.keys()) != set(node_attrs):
            raise ValueError('Not all nodes contain the same attributes')
        for key, value in feat_dict.items():
            data_dict[str(key)].append(value)

    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        if set(feat_dict.keys()) != set(edge_attrs):
            raise ValueError('Not all edges contain the same attributes')
        for key, value in feat_dict.items():
            key = f'edge_{key}' if key in node_attrs else key
            data_dict[str(key)].append(value)

    for key, value in G.graph.items():
        if key == 'node_default' or key == 'edge_default':
            continue  # Do not load default attributes.
        key = f'graph_{key}' if key in node_attrs else key
        data_dict[str(key)] = value

    for key, value in data_dict.items():
        if isinstance(value, (tuple, list)) and isinstance(value[0], Tensor):
            data_dict[key] = torch.stack(value, dim=0)
        else:
            try:
                data_dict[key] = torch.as_tensor(value)
            except Exception:
                pass

    data = Data.from_dict(data_dict)

    if group_node_attrs is not None:
        xs = []
        for key in group_node_attrs:
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]
        data.x = torch.cat(xs, dim=-1)

    if group_edge_attrs is not None:
        xs = []
        for key in group_edge_attrs:
            key = f'edge_{key}' if key in node_attrs else key
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]
        data.edge_attr = torch.cat(xs, dim=-1)

    if data.x is None and data.pos is None:
        data.num_nodes = G.number_of_nodes()

    return data


def ensure_connected(graph, seed=0):
    """Ensure the graph is connected by adding minimal edges between disconnected components."""
    random.seed(seed)  # For reproducibility
    # While the graph is not connected, connect one pair of nodes from different components
    while not nx.is_connected(graph):
        components = list(nx.connected_components(graph))
        if len(components) <= 1:
            break
        # Choose two distinct components (here, simply take the first two)
        comp1, comp2 = components[0], components[1]
        # Randomly choose one node from each component
        node1 = random.choice(list(comp1))
        node2 = random.choice(list(comp2))
        # Add an edge between these two nodes with a designated edge attribute.
        graph.add_edge(node1, node2, edge_attr=[0, 1])
    return graph


def graph_generator(sizes, prob_matrix, seed=0):
    # Set the random seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)

    # Step 1: Generate the graph using the SBM model
    graph_sbm = nx.stochastic_block_model(sizes, prob_matrix, seed=seed)

    # Ensure the graph is connected by linking disconnected components
    graph_sbm = ensure_connected(graph_sbm, seed=seed)

    # Generate node features as one-hot encoding (each node gets a one-hot vector for its community)
    node_features = np.zeros((graph_sbm.number_of_nodes(), len(sizes)))
    for i, size in enumerate(sizes):
        start = sum(sizes[:i])
        end = start + size
        node_features[start:end, i] = 1

    # Assign node features to the graph (PyG expects them under the attribute 'x')
    for i, node in enumerate(graph_sbm.nodes()):
        graph_sbm.nodes[node]['x'] = node_features[i]

    # Assign edge features (weights) to all edges (for edges added in ensure_connected, this ensures the attribute is set)
    for u, v in graph_sbm.edges():
        if 'edge_attr' not in graph_sbm.edges[u, v]:
            graph_sbm.edges[u, v]['edge_attr'] = [0, 1]

    # Convert the NetworkX graph to a PyG Data object
    data = from_networkx(graph_sbm)
    data.n_nodes = graph_sbm.number_of_nodes()
    return data, graph_sbm


def generate_graphs_for_sizes(min_size, max_size, prob_matrix, num_graphs, seed=0):
    graphs = []
    random.seed(seed)
    for _ in range(num_graphs-1):
        size = random.randint(min_size, max_size)
        sizes = [size // 2, size // 2]  # Example: Two communities of equal size
        data, graph_sbm = graph_generator(sizes, prob_matrix, seed)
        graphs.append(data)
    data, graph_sbm = graph_generator([max_size // 2, max_size // 2], prob_matrix, seed)
    graphs.append(data)
    batched_data = Batch.from_data_list(graphs)
    return batched_data, graphs

def main():
    sizes = [30, 20, 20]  # Two communities of sizes 200 and 50
    prob_matrix = [[0.3, 0.01, 0.01], [0.01, 0.3, 0.01], [0.01,0.01, 0.3]]  # Edge probabilities within and between communities
    graph, _ = graph_generator(sizes, prob_matrix)
    print(graph)


if __name__ == '__main__':
    main()