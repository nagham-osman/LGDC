###############################################################################
#
# Adapted from https://github.com/lrjconan/GRAN/ which in turn is adapted from https://github.com/JiaxuanYou/graph-generation
#
###############################################################################
import graph_tool.all as gt
##Navigate to the ./util/orca directory and compile orca.cpp
# g++ -O2 -std=c++11 -o orca orca.cpp
import os
import copy
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import subprocess as sp
import concurrent.futures

import pygsp as pg
import secrets
from string import ascii_uppercase, digits
from datetime import datetime
from scipy.linalg import eigvalsh
from scipy.stats import chi2
from src.analysis.dist_helper import compute_mmd, gaussian_emd, gaussian, emd, gaussian_tv, disc
from torch_geometric.utils import to_networkx
import wandb

from torch_geometric.data import Data
from torch_sparse import SparseTensor

from src.datasets.coarsen_spectre_dataset import DeterministicCoarsen

from networkx.algorithms import approximation as apx
from networkx.algorithms import community as nxcom
from networkx.algorithms.cuts import conductance

PRINT_TIME = False
__all__ = ['degree_stats', 'clustering_stats', 'orbit_stats_all', 'spectral_stats', 'eval_acc_lobster_graph',
           'diameter_stats', 'aspl_stats', 'components_stats', 'edge_connectivity_stats',
           'modularity_stats', 'conductance_stats']


def degree_worker(G):
    return np.array(nx.degree_histogram(G))


def degree_stats(graph_ref_list, graph_pred_list, is_parallel=True, compute_emd=False):
    ''' Compute the distance between the degree distributions of two unordered sets of graphs.
        Args:
            graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
        '''
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_ref_list):
                sample_ref.append(deg_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_pred_list_remove_empty):
                sample_pred.append(deg_hist)
    else:
        for i in range(len(graph_ref_list)):
            degree_temp = np.array(nx.degree_histogram(graph_ref_list[i]))
            sample_ref.append(degree_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            degree_temp = np.array(
                nx.degree_histogram(graph_pred_list_remove_empty[i]))
            sample_pred.append(degree_temp)

    if compute_emd:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    else:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing degree mmd: ', elapsed)
    return mmd_dist


###############################################################################

def spectral_worker(G, n_eigvals=-1):
    # eigs = nx.laplacian_spectrum(G)
    try:
        eigs = eigvalsh(nx.normalized_laplacian_matrix(G).todense())
    except:
        eigs = np.zeros(G.number_of_nodes())
    if n_eigvals > 0:
        eigs = eigs[1:n_eigvals + 1]
    spectral_pmf, _ = np.histogram(eigs, bins=200, range=(-1e-5, 2), density=False)
    spectral_pmf = spectral_pmf / spectral_pmf.sum()
    return spectral_pmf


def get_spectral_pmf(eigs, max_eig):
    spectral_pmf, _ = np.histogram(np.clip(eigs, 0, max_eig), bins=200, range=(-1e-5, max_eig), density=False)
    spectral_pmf = spectral_pmf / spectral_pmf.sum()
    return spectral_pmf


def eigval_stats(eig_ref_list, eig_pred_list, max_eig=20, is_parallel=True, compute_emd=False):
    ''' Compute the distance between the degree distributions of two unordered sets of graphs.
        Args:
            graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
        '''
    sample_ref = []
    sample_pred = []

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(get_spectral_pmf, eig_ref_list,
                                                 [max_eig for i in range(len(eig_ref_list))]):
                sample_ref.append(spectral_density)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(get_spectral_pmf, eig_pred_list,
                                                 [max_eig for i in range(len(eig_ref_list))]):
                sample_pred.append(spectral_density)
    else:
        for i in range(len(eig_ref_list)):
            spectral_temp = get_spectral_pmf(eig_ref_list[i])
            sample_ref.append(spectral_temp)
        for i in range(len(eig_pred_list)):
            spectral_temp = get_spectral_pmf(eig_pred_list[i])
            sample_pred.append(spectral_temp)

    if compute_emd:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
    else:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing eig mmd: ', elapsed)
    return mmd_dist


def eigh_worker(G):
    L = nx.normalized_laplacian_matrix(G).todense()
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
    except:
        eigvals = np.zeros(L[0, :].shape)
        eigvecs = np.zeros(L.shape)
    return (eigvals, eigvecs)


def compute_list_eigh(graph_list, is_parallel=False):
    eigval_list = []
    eigvec_list = []
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for e_U in executor.map(eigh_worker, graph_list):
                eigval_list.append(e_U[0])
                eigvec_list.append(e_U[1])
    else:
        for i in range(len(graph_list)):
            e_U = eigh_worker(graph_list[i])
            eigval_list.append(e_U[0])
            eigvec_list.append(e_U[1])
    return eigval_list, eigvec_list


def get_spectral_filter_worker(eigvec, eigval, filters, bound=1.4):
    ges = filters.evaluate(eigval)
    linop = []
    for ge in ges:
        linop.append(eigvec @ np.diag(ge) @ eigvec.T)
    linop = np.array(linop)
    norm_filt = np.sum(linop ** 2, axis=2)
    hist_range = [0, bound]
    hist = np.array([np.histogram(x, range=hist_range, bins=100)[0] for x in norm_filt])  # NOTE: change number of bins
    return hist.flatten()


def spectral_filter_stats(eigvec_ref_list, eigval_ref_list, eigvec_pred_list, eigval_pred_list, is_parallel=False,
                          compute_emd=False):
    ''' Compute the distance between the eigvector sets.
        Args:
            graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
        '''
    prev = datetime.now()

    class DMG(object):
        """Dummy Normalized Graph"""
        lmax = 2

    n_filters = 12
    filters = pg.filters.Abspline(DMG, n_filters)
    bound = np.max(filters.evaluate(np.arange(0, 2, 0.01)))
    sample_ref = []
    sample_pred = []
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(get_spectral_filter_worker, eigvec_ref_list, eigval_ref_list,
                                                 [filters for i in range(len(eigval_ref_list))],
                                                 [bound for i in range(len(eigval_ref_list))]):
                sample_ref.append(spectral_density)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(get_spectral_filter_worker, eigvec_pred_list, eigval_pred_list,
                                                 [filters for i in range(len(eigval_ref_list))],
                                                 [bound for i in range(len(eigval_ref_list))]):
                sample_pred.append(spectral_density)
    else:
        for i in range(len(eigval_ref_list)):
            try:
                spectral_temp = get_spectral_filter_worker(eigvec_ref_list[i], eigval_ref_list[i], filters, bound)
                sample_ref.append(spectral_temp)
            except:
                pass
        for i in range(len(eigval_pred_list)):
            try:
                spectral_temp = get_spectral_filter_worker(eigvec_pred_list[i], eigval_pred_list[i], filters, bound)
                sample_pred.append(spectral_temp)
            except:
                pass

    if compute_emd:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    else:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing spectral filter stats: ', elapsed)
    return mmd_dist


def spectral_stats(graph_ref_list, graph_pred_list, is_parallel=True, n_eigvals=-1, compute_emd=False):
    ''' Compute the distance between the degree distributions of two unordered sets of graphs.
        Args:
            graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
        '''
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(spectral_worker, graph_ref_list, [n_eigvals for i in graph_ref_list]):
                sample_ref.append(spectral_density)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(spectral_worker, graph_pred_list_remove_empty,
                                                 [n_eigvals for i in graph_ref_list]):
                sample_pred.append(spectral_density)
    else:
        for i in range(len(graph_ref_list)):
            spectral_temp = spectral_worker(graph_ref_list[i], n_eigvals)
            sample_ref.append(spectral_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            spectral_temp = spectral_worker(graph_pred_list_remove_empty[i], n_eigvals)
            sample_pred.append(spectral_temp)

    if compute_emd:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    else:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing degree mmd: ', elapsed)
    return mmd_dist


###############################################################################

def clustering_worker(param):
    G, bins = param
    clustering_coeffs_list = list(nx.clustering(G).values())
    hist, _ = np.histogram(
        clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
    return hist


def clustering_stats(graph_ref_list,
                     graph_pred_list,
                     bins=100,
                     is_parallel=True, compute_emd=False):
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(clustering_worker,
                                                [(G, bins) for G in graph_ref_list]):
                sample_ref.append(clustering_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(
                    clustering_worker, [(G, bins) for G in graph_pred_list_remove_empty]):
                sample_pred.append(clustering_hist)
    else:
        for i in range(len(graph_ref_list)):
            clustering_coeffs_list = list(nx.clustering(graph_ref_list[i]).values())
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_ref.append(hist)

        for i in range(len(graph_pred_list_remove_empty)):
            clustering_coeffs_list = list(
                nx.clustering(graph_pred_list_remove_empty[i]).values())
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_pred.append(hist)

    if compute_emd:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd, sigma=1.0 / 10, distance_scaling=bins)
    else:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv, sigma=1.0 / 10)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing clustering mmd: ', elapsed)
    return mmd_dist


# maps motif/orbit name string to its corresponding list of indices from orca output
motif_to_indices = {
    '3path': [1, 2],
    '4cycle': [8],
}
COUNT_START_STR = 'orbit counts:'


def edge_list_reindexed(G):
    idx = 0
    id2idx = dict()
    for u in G.nodes():
        id2idx[str(u)] = idx
        idx += 1

    edges = []
    for (u, v) in G.edges():
        edges.append((id2idx[str(u)], id2idx[str(v)]))
    return edges


def orca(graph):
    # tmp_fname = f'analysis/orca/tmp_{"".join(secrets.choice(ascii_uppercase + digits) for i in range(8))}.txt'
    tmp_fname = f'orca/tmp_{"".join(secrets.choice(ascii_uppercase + digits) for i in range(8))}.txt'
    tmp_fname = os.path.join(os.path.dirname(os.path.realpath(__file__)), tmp_fname)
    # print(tmp_fname, flush=True)
    f = open(tmp_fname, 'w')
    f.write(
        str(graph.number_of_nodes()) + ' ' + str(graph.number_of_edges()) + '\n')
    for (u, v) in edge_list_reindexed(graph):
        f.write(str(u) + ' ' + str(v) + '\n')
    f.close()
    output = sp.check_output(
        [str(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'orca/orca')), 'node', '4', tmp_fname, 'std'])
    output = output.decode('utf8').strip()
    idx = output.find(COUNT_START_STR) + len(COUNT_START_STR) + 2
    output = output[idx:]
    node_orbit_counts = np.array([
        # list(map(int,
        #          node_cnts.strip().split(' ')))
        list(map(int,
                 node_cnts.strip().split()))
        for node_cnts in output.strip('\n').split('\n')
    ])

    try:
        os.remove(tmp_fname)
    except OSError:
        pass

    return node_orbit_counts


def motif_stats(graph_ref_list, graph_pred_list, motif_type='4cycle', ground_truth_match=None,
                bins=100, compute_emd=False):
    # graph motif counts (int for each graph)
    # normalized by graph size
    total_counts_ref = []
    total_counts_pred = []

    num_matches_ref = []
    num_matches_pred = []

    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]
    indices = motif_to_indices[motif_type]

    for G in graph_ref_list:
        orbit_counts = orca(G)
        motif_counts = np.sum(orbit_counts[:, indices], axis=1)

        if ground_truth_match is not None:
            match_cnt = 0
            for elem in motif_counts:
                if elem == ground_truth_match:
                    match_cnt += 1
            num_matches_ref.append(match_cnt / G.number_of_nodes())

        # hist, _ = np.histogram(
        #        motif_counts, bins=bins, density=False)
        motif_temp = np.sum(motif_counts) / G.number_of_nodes()
        total_counts_ref.append(motif_temp)

    for G in graph_pred_list_remove_empty:
        orbit_counts = orca(G)
        motif_counts = np.sum(orbit_counts[:, indices], axis=1)

        if ground_truth_match is not None:
            match_cnt = 0
            for elem in motif_counts:
                if elem == ground_truth_match:
                    match_cnt += 1
            num_matches_pred.append(match_cnt / G.number_of_nodes())

        motif_temp = np.sum(motif_counts) / G.number_of_nodes()
        total_counts_pred.append(motif_temp)

    total_counts_ref = np.array(total_counts_ref)[:, None]
    total_counts_pred = np.array(total_counts_pred)[:, None]


    if compute_emd:
        mmd_dist = compute_mmd(total_counts_ref, total_counts_pred, kernel=gaussian, is_hist=False)
    else:
        mmd_dist = compute_mmd(total_counts_ref, total_counts_pred, kernel=gaussian, is_hist=False)
    return mmd_dist


def orbit_stats_all(graph_ref_list, graph_pred_list, compute_emd=False):
    total_counts_ref = []
    total_counts_pred = []

    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    for G in graph_ref_list:
        orbit_counts = orca(G)
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_ref.append(orbit_counts_graph)

    for G in graph_pred_list:
        orbit_counts = orca(G)
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_pred.append(orbit_counts_graph)

    total_counts_ref = np.array(total_counts_ref)
    total_counts_pred = np.array(total_counts_pred)

    if compute_emd:
        mmd_dist = compute_mmd(total_counts_ref, total_counts_pred, kernel=gaussian, is_hist=False, sigma=30.0)
    else:
        mmd_dist = compute_mmd(total_counts_ref, total_counts_pred, kernel=gaussian_tv, is_hist=False, sigma=30.0)
    return mmd_dist


def eval_acc_lobster_graph(G_list):
    G_list = [copy.deepcopy(gg) for gg in G_list]
    count = 0
    for gg in G_list:
        if is_lobster_graph(gg):
            count += 1
    return count / float(len(G_list))


def eval_acc_tree_graph(G_list):
    count = 0
    for gg in G_list:
        if nx.is_tree(gg):
            count += 1
    return count / float(len(G_list))


def eval_acc_grid_graph(G_list, grid_start=10, grid_end=20):
    count = 0
    for gg in G_list:
        if is_grid_graph(gg):
            count += 1
    return count / float(len(G_list))


def eval_acc_sbm_graph(G_list, p_intra=0.3, p_inter=0.005, strict=True, refinement_steps=1000, is_parallel=True):
    count = 0.0
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for prob in executor.map(is_sbm_graph,
                                     [gg for gg in G_list], [p_intra for i in range(len(G_list))],
                                     [p_inter for i in range(len(G_list))],
                                     [strict for i in range(len(G_list))],
                                     [refinement_steps for i in range(len(G_list))]):
                count += prob
    else:
        for gg in G_list:
            count += is_sbm_graph(gg, p_intra=p_intra, p_inter=p_inter, strict=strict,
                                  refinement_steps=refinement_steps)
    return count / float(len(G_list))


def eval_acc_planar_graph(G_list):
    count = 0
    for gg in G_list:
        if is_planar_graph(gg):
            count += 1
    return count / float(len(G_list))


def is_planar_graph(G):
    return nx.is_connected(G) and nx.check_planarity(G)[0]


def is_lobster_graph(G):
    """
        Check a given graph is a lobster graph or not

        Removing leaf nodes twice:

        lobster -> caterpillar -> path

    """
    ### Check if G is a tree
    if nx.is_tree(G):
        G = G.copy()
        ### Check if G is a path after removing leaves twice
        leaves = [n for n, d in G.degree() if d == 1]
        G.remove_nodes_from(leaves)

        leaves = [n for n, d in G.degree() if d == 1]
        G.remove_nodes_from(leaves)

        num_nodes = len(G.nodes())
        num_degree_one = [d for n, d in G.degree() if d == 1]
        num_degree_two = [d for n, d in G.degree() if d == 2]

        if sum(num_degree_one) == 2 and sum(num_degree_two) == 2 * (num_nodes - 2):
            return True
        elif sum(num_degree_one) == 0 and sum(num_degree_two) == 0:
            return True
        else:
            return False
    else:
        return False


def is_grid_graph(G):
    """
    Check if the graph is grid, by comparing with all the real grids with the same node count
    """
    all_grid_file = f"data/all_grids.pt"
    if os.path.isfile(all_grid_file):
        all_grids = torch.load(all_grid_file)
    else:
        all_grids = {}
        for i in range(2, 20):
            for j in range(2, 20):
                G_grid = nx.grid_2d_graph(i, j)
                n_nodes = f"{len(G_grid.nodes())}"
                all_grids[n_nodes] = all_grids.get(n_nodes, []) + [G_grid]
        torch.save(all_grids, all_grid_file)

    n_nodes = f"{len(G.nodes())}"
    if n_nodes in all_grids:
        for G_grid in all_grids[n_nodes]:
            if nx.faster_could_be_isomorphic(G, G_grid):
                if nx.is_isomorphic(G, G_grid):
                    return True
        return False
    else:
        return False


def is_sbm_graph(G, p_intra=0.3, p_inter=0.005, strict=True, refinement_steps=1000):
    """
    Check if how closely given graph matches a SBM with given probabilites by computing mean probability of Wald test statistic for each recovered parameter
    """

    adj = nx.adjacency_matrix(G).toarray()
    idx = adj.nonzero()
    g = gt.Graph()
    g.add_edge_list(np.transpose(idx))
    try:
        state = gt.minimize_blockmodel_dl(g)
    except ValueError:
        if strict:
            return False
        else:
            return 0.0

    # Refine using merge-split MCMC
    for i in range(refinement_steps):
        state.multiflip_mcmc_sweep(beta=np.inf, niter=10)

    b = state.get_blocks()
    b = gt.contiguous_map(state.get_blocks())
    state = state.copy(b=b)
    e = state.get_matrix()
    n_blocks = state.get_nonempty_B()
    node_counts = state.get_nr().get_array()[:n_blocks]
    edge_counts = e.todense()[:n_blocks, :n_blocks]
    if strict:
        if (node_counts > 40).sum() > 0 or (node_counts < 20).sum() > 0 or n_blocks > 5 or n_blocks < 2:
            return False

    max_intra_edges = node_counts * (node_counts - 1)
    est_p_intra = np.diagonal(edge_counts) / (max_intra_edges + 1e-6)

    max_inter_edges = node_counts.reshape((-1, 1)) @ node_counts.reshape((1, -1))
    np.fill_diagonal(edge_counts, 0)
    est_p_inter = edge_counts / (max_inter_edges + 1e-6)

    W_p_intra = (est_p_intra - p_intra) ** 2 / (est_p_intra * (1 - est_p_intra) + 1e-6)
    W_p_inter = (est_p_inter - p_inter) ** 2 / (est_p_inter * (1 - est_p_inter) + 1e-6)

    W = W_p_inter.copy()
    np.fill_diagonal(W, W_p_intra)
    p = 1 - chi2.cdf(abs(W), 1)
    p = p.mean()
    if strict:
        return p > 0.9  # p value < 10 %
    else:
        return p


def eval_fraction_isomorphic(fake_graphs, train_graphs):
    count = 0
    for fake_g in fake_graphs:
        for train_g in train_graphs:
            if nx.faster_could_be_isomorphic(fake_g, train_g):
                if nx.is_isomorphic(fake_g, train_g):
                    count += 1
                    break
    return count / float(len(fake_graphs))


def eval_fraction_unique(fake_graphs, precise=False):
    count_non_unique = 0
    fake_evaluated = []
    for fake_g in fake_graphs:
        unique = True
        if not fake_g.number_of_nodes() == 0:
            for fake_old in fake_evaluated:
                if precise:
                    if nx.faster_could_be_isomorphic(fake_g, fake_old):
                        if nx.is_isomorphic(fake_g, fake_old):
                            count_non_unique += 1
                            unique = False
                            break
                else:
                    if nx.faster_could_be_isomorphic(fake_g, fake_old):
                        if nx.could_be_isomorphic(fake_g, fake_old):
                            count_non_unique += 1
                            unique = False
                            break
            if unique:
                fake_evaluated.append(fake_g)

    frac_unique = (float(len(fake_graphs)) - count_non_unique) / float(
        len(fake_graphs))  # Fraction of distinct isomorphism classes in the fake graphs

    return frac_unique


def eval_fraction_unique_non_isomorphic_valid(fake_graphs, train_graphs, validity_func=(lambda x: True)):
    count_valid = 0
    count_isomorphic = 0
    count_non_unique = 0
    fake_evaluated = []
    for fake_g in fake_graphs:
        unique = True

        for fake_old in fake_evaluated:
            if nx.faster_could_be_isomorphic(fake_g, fake_old):
                if nx.is_isomorphic(fake_g, fake_old):
                    count_non_unique += 1
                    unique = False
                    break
        if unique:
            fake_evaluated.append(fake_g)
            non_isomorphic = True
            for train_g in train_graphs:
                if nx.faster_could_be_isomorphic(fake_g, train_g):
                    if nx.is_isomorphic(fake_g, train_g):
                        count_isomorphic += 1
                        non_isomorphic = False
                        break
            if non_isomorphic:
                if validity_func(fake_g):
                    count_valid += 1

    frac_unique = (float(len(fake_graphs)) - count_non_unique) / float(
        len(fake_graphs))  # Fraction of distinct isomorphism classes in the fake graphs
    frac_unique_non_isomorphic = (float(len(fake_graphs)) - count_non_unique - count_isomorphic) / float(
        len(fake_graphs))  # Fraction of distinct isomorphism classes in the fake graphs that are not in the training set
    frac_unique_non_isomorphic_valid = count_valid / float(
        len(fake_graphs))  # Fraction of distinct isomorphism classes in the fake graphs that are not in the training set and are valid
    return frac_unique, frac_unique_non_isomorphic, frac_unique_non_isomorphic_valid

#############################################################################################################
# Global Metrics
# ---------- helpers ----------
def _largest_cc(G):
    if G.number_of_nodes() == 0 or nx.is_connected(G):
        return G
    nodes = max(nx.connected_components(G), key=len)
    return G.subgraph(nodes).copy()

def diameter_worker(G, use_lcc=True, approx_when_n_gt=2000):
    if G.number_of_nodes() <= 1:
        return 0.0
    H = _largest_cc(G) if use_lcc else G
    if H.number_of_nodes() <= 1:
        return 0.0
    try:
        if approx_when_n_gt is not None and H.number_of_nodes() > approx_when_n_gt:
            return float(apx.diameter(H))
        return float(nx.diameter(H))
    except Exception:
        return 0.0

def aspl_worker(G, use_lcc=True):
    if G.number_of_nodes() <= 1:
        return 0.0
    H = _largest_cc(G) if use_lcc else G
    if H.number_of_nodes() <= 1:
        return 0.0
    try:
        return float(nx.average_shortest_path_length(H))
    except Exception:
        return 0.0

def components_worker(G):
    try:
        return float(nx.number_connected_components(G))
    except Exception:
        return 0.0

def edge_connectivity_worker(G):
    n = G.number_of_nodes()
    if n <= 1:
        return 0.0
    if not nx.is_connected(G):
        return 0.0
    mindeg = min(dict(G.degree()).values()) if n > 0 else 0
    if mindeg == 0:
        return 0.0
    try:
        return float(nx.edge_connectivity(G))
    except Exception:
        return float(min(1, mindeg))

def modularity_worker(G):
    try:
        if G.number_of_edges() == 0 or G.number_of_nodes() == 0:
            return 0.0
        comms = nxcom.greedy_modularity_communities(G, seed=42)
        return float(nxcom.modularity(G, comms))
    except Exception:
        return 0.0

def conductance_worker(G):
    try:
        if G.number_of_edges() == 0 or G.number_of_nodes() == 0:
            return 0.0
        comms = list(nxcom.greedy_modularity_communities(G, seed=42))
        if len(comms) <= 1:
            return 0.0
        vals, weights = [], []
        for S in comms:
            S = set(S)
            if 0 < len(S) < G.number_of_nodes():
                try:
                    c = conductance(G, S)
                    vals.append(float(c))
                    weights.append(len(S))
                except Exception:
                    pass
        if not vals:
            return 0.0
        return float(np.average(vals, weights=weights))
    except Exception:
        return 0.0

def _scalar_stat_mmd(graph_ref_list, graph_pred_list, worker_fn, is_parallel=False, compute_emd=False):
    sample_ref, sample_pred = [], []
    graph_pred_list_remove_empty = [G for G in graph_pred_list if G.number_of_nodes() != 0]

    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as ex:
            for v in ex.map(worker_fn, graph_ref_list):
                sample_ref.append(np.asarray([v], dtype=float))
        with concurrent.futures.ThreadPoolExecutor() as ex:
            for v in ex.map(worker_fn, graph_pred_list_remove_empty):
                sample_pred.append(np.asarray([v], dtype=float))
    else:
        for G in graph_ref_list:
            sample_ref.append(np.asarray([worker_fn(G)], dtype=float))
        for G in graph_pred_list_remove_empty:
            sample_pred.append(np.asarray([worker_fn(G)], dtype=float))

    return compute_mmd(sample_ref, sample_pred, kernel=gaussian, is_hist=False)

# ---------- stats functions ----------
def diameter_stats(graph_ref_list, graph_pred_list, use_lcc=True, is_parallel=False, compute_emd=False):
    return _scalar_stat_mmd(graph_ref_list, graph_pred_list,
                            lambda G: diameter_worker(G, use_lcc=use_lcc),
                            is_parallel=is_parallel, compute_emd=compute_emd)

def aspl_stats(graph_ref_list, graph_pred_list, use_lcc=True, is_parallel=False, compute_emd=False):
    return _scalar_stat_mmd(graph_ref_list, graph_pred_list,
                            lambda G: aspl_worker(G, use_lcc=use_lcc),
                            is_parallel=is_parallel, compute_emd=compute_emd)

def components_stats(graph_ref_list, graph_pred_list, is_parallel=False, compute_emd=False):
    return _scalar_stat_mmd(graph_ref_list, graph_pred_list,
                            components_worker,
                            is_parallel=is_parallel, compute_emd=compute_emd)

def edge_connectivity_stats(graph_ref_list, graph_pred_list, is_parallel=False, compute_emd=False):
    return _scalar_stat_mmd(graph_ref_list, graph_pred_list,
                            edge_connectivity_worker,
                            is_parallel=is_parallel, compute_emd=compute_emd)

def modularity_stats(graph_ref_list, graph_pred_list, is_parallel=False, compute_emd=False):
    return _scalar_stat_mmd(graph_ref_list, graph_pred_list,
                            modularity_worker,
                            is_parallel=is_parallel, compute_emd=compute_emd)

def conductance_stats(graph_ref_list, graph_pred_list, is_parallel=False, compute_emd=False):
    return _scalar_stat_mmd(graph_ref_list, graph_pred_list,
                            conductance_worker,
                            is_parallel=is_parallel, compute_emd=compute_emd)


class SpectreSamplingMetrics(nn.Module):
    def __init__(self, datamodule, compute_emd, metrics_list):
        super().__init__()

        self.train_graphs = self.loader_to_nx(datamodule.train_dataset, apply_transform=False)
        self.val_graphs = self.loader_to_nx(datamodule.val_dataset, apply_transform=False)
        self.test_graphs = self.loader_to_nx(datamodule.test_dataset, apply_transform=False)

        # self.train_graphs_coarse = self.build_deterministic_train_coarse_for_baseline(
        #     datamodule, seed=42)
        self.train_graphs_coarse = self.loader_to_nx(datamodule.train_dataset, apply_transform=True)
        self.val_graphs_coarse = self.loader_to_nx(datamodule.val_dataset, apply_transform=True)
        self.test_graphs_coarse = self.loader_to_nx(datamodule.test_dataset, apply_transform=True)

        self.num_graphs_test = len(self.test_graphs)
        self.num_graphs_val = len(self.val_graphs)
        self.compute_emd = compute_emd
        self.metrics_list = metrics_list


    # def materialize_coarse_graphs_with_transform(self, dataset, transform):
    #     old_tf = getattr(dataset, "transform", None)
    #     try:
    #         dataset.transform = transform
    #         graphs = self.loader_to_nx(dataset, apply_transform=True)
    #     finally:
    #         dataset.transform = old_tf
    #     return graphs
    #
    # def build_deterministic_train_coarse_for_baseline(self, datamodule, seed=42):
    #     det_tf = DeterministicCoarsen(
    #         datamodule.red_factory,
    #         datamodule.spectrum_extractor,
    #         datamodule.cfg.reduction.latent_graph_size,
    #         seed=seed,
    #     )
    #     return self.materialize_coarse_graphs_with_transform(datamodule.train_dataset, det_tf)

    def loader_to_nx(self, dataset, apply_transform=False):
        networkx_graphs = []
        if apply_transform:
            for i in range(len(dataset)):
                d = dataset[i]
                if hasattr(d, "adj_reduced") and isinstance(d.adj_reduced, SparseTensor):
                    row, col, _ = d.adj_reduced.coo()
                    ei = torch.stack([row, col], dim=0)
                    num_nodes = d.adj_reduced.size(0)
                else:
                    if hasattr(d, "expansion_matrix"):
                        num_nodes = d.expansion_matrix.size(1)
                        ei = torch.empty(2, 0, dtype=torch.long, device=d.x.device)
                    else:
                        raise RuntimeError("No coarsened topology: expected adj_reduced or expansion_matrix.")
                data_for_nx = Data(edge_index=ei, num_nodes=num_nodes)
                networkx_graphs.append(
                    to_networkx(data_for_nx, node_attrs=None, edge_attrs=None,
                                to_undirected=True, remove_self_loops=True)
                )
        else:
            for i in range(len(dataset)):
                d = dataset.get(i)
                if hasattr(d, "edge_index") and d.edge_index is not None:
                    data_for_nx = d
                elif hasattr(d, "adj") and isinstance(d.adj, SparseTensor):
                    row, col, _ = d.adj.coo()
                    ei = torch.stack([row, col], dim=0)
                    data_for_nx = Data(edge_index=ei, num_nodes=d.adj.size(0))
                elif hasattr(d, "adj_t") and isinstance(d.adj_t, SparseTensor):
                    row, col, _ = d.adj_t.t().coo()
                    ei = torch.stack([row, col], dim=0)
                    data_for_nx = Data(edge_index=ei, num_nodes=d.adj_t.size(0))
                else:
                    raise RuntimeError("No topology found for raw item.")
                networkx_graphs.append(
                    to_networkx(data_for_nx, node_attrs=None, edge_attrs=None,
                                to_undirected=True, remove_self_loops=True)
                )
        return networkx_graphs

    def compute_and_log_metric(self, metric_name, reference_graphs, networkx_graphs, to_log, name, wandb, local_rank):
        if metric_name in self.metrics_list:
            if local_rank == 0:
                print(f"Computing {metric_name} stats...")

            if metric_name == 'degree':
                value = degree_stats(reference_graphs, networkx_graphs, compute_emd=self.compute_emd)
            elif metric_name == 'spectre':
                value = spectral_stats(reference_graphs, networkx_graphs, n_eigvals=-1,
                                       compute_emd=self.compute_emd)
            elif metric_name == 'clustering':
                value = clustering_stats(reference_graphs, networkx_graphs, bins=100,
                                         compute_emd=self.compute_emd)
            elif metric_name == 'motif':
                value = motif_stats(reference_graphs, networkx_graphs, motif_type='4cycle', ground_truth_match=None,
                                    bins=100, compute_emd=self.compute_emd)
            elif metric_name == 'orbit':
                value = orbit_stats_all(reference_graphs, networkx_graphs, compute_emd=self.compute_emd)
            elif metric_name == 'wavelet':
                reference_eigvals, reference_eigvecs = compute_list_eigh(reference_graphs, is_parallel=False)
                predicted_eigvals, predicted_eigvecs = compute_list_eigh(networkx_graphs, is_parallel=False)
                value = spectral_filter_stats(reference_eigvecs, reference_eigvals, predicted_eigvecs,
                                              predicted_eigvals, compute_emd=self.compute_emd)
            elif metric_name == 'sbm':
                value = eval_acc_sbm_graph(networkx_graphs, refinement_steps=100, strict=True)
            elif metric_name == 'planar':
                value = eval_acc_planar_graph(networkx_graphs)
            elif metric_name == 'tree':
                value = eval_acc_tree_graph(networkx_graphs)
            elif metric_name == 'diameter':
                value = diameter_stats(reference_graphs, networkx_graphs, use_lcc=True,
                                       is_parallel=False, compute_emd=self.compute_emd)
            elif metric_name == 'aspl':
                value = aspl_stats(reference_graphs, networkx_graphs, use_lcc=True,
                                   is_parallel=False, compute_emd=self.compute_emd)
            elif metric_name == 'components':
                value = components_stats(reference_graphs, networkx_graphs,
                                         is_parallel=False, compute_emd=self.compute_emd)
            elif metric_name == 'edge_connectivity':
                value = edge_connectivity_stats(reference_graphs, networkx_graphs,
                                                is_parallel=False, compute_emd=self.compute_emd)
            elif metric_name == 'modularity':
                value = modularity_stats(reference_graphs, networkx_graphs,
                                         is_parallel=False, compute_emd=self.compute_emd)
            elif metric_name == 'conductance':
                value = conductance_stats(reference_graphs, networkx_graphs,
                                          is_parallel=False, compute_emd=self.compute_emd)
            else:
                return

            to_log[f'{name}/{metric_name}'] = float(value)
            if wandb.run:
                wandb.run.summary[metric_name] = float(value)


    def compute_ratios(self, preds, reference_graphs, train_graphs, to_log, name):
        ratios = {}
        for metric in self.metrics_list:
            if metric in ('sbm', 'planar', 'tree'):
                continue
            if metric not in preds:
                continue

            if metric == 'degree':
                baseline = degree_stats(reference_graphs, train_graphs, is_parallel=False,
                                        compute_emd=self.compute_emd)
            elif metric == 'spectre':
                baseline = spectral_stats(reference_graphs, train_graphs, is_parallel=False, n_eigvals=-1,
                                          compute_emd=self.compute_emd)
            elif metric == 'clustering':
                baseline = clustering_stats(reference_graphs, train_graphs, bins=100, is_parallel=False,
                                            compute_emd=self.compute_emd)
            elif metric == 'motif':
                baseline = motif_stats(reference_graphs, train_graphs, motif_type='4cycle', ground_truth_match=None,
                                       bins=100, compute_emd=self.compute_emd)
            elif metric == 'orbit':
                baseline = orbit_stats_all(reference_graphs, train_graphs, compute_emd=self.compute_emd)
            elif metric == 'wavelet':
                ref_eigvals, ref_eigvecs = compute_list_eigh(reference_graphs, is_parallel=False)
                train_eigvals, train_eigvecs = compute_list_eigh(train_graphs, is_parallel=False)
                baseline = spectral_filter_stats(ref_eigvecs, ref_eigvals, train_eigvecs, train_eigvals,
                                                 is_parallel=False, compute_emd=self.compute_emd)
            elif metric == 'diameter':
                baseline = diameter_stats(reference_graphs, train_graphs, use_lcc=True,
                                          is_parallel=False, compute_emd=self.compute_emd)
            elif metric == 'aspl':
                baseline = aspl_stats(reference_graphs, train_graphs, use_lcc=True,
                                      is_parallel=False, compute_emd=self.compute_emd)
            elif metric == 'components':
                baseline = components_stats(reference_graphs, train_graphs,
                                            is_parallel=False, compute_emd=self.compute_emd)
            elif metric == 'edge_connectivity':
                baseline = edge_connectivity_stats(reference_graphs, train_graphs,
                                                   is_parallel=False, compute_emd=self.compute_emd)
            elif metric == 'modularity':
                baseline = modularity_stats(reference_graphs, train_graphs,
                                            is_parallel=False, compute_emd=self.compute_emd)
            elif metric == 'conductance':
                baseline = conductance_stats(reference_graphs, train_graphs,
                                             is_parallel=False, compute_emd=self.compute_emd)
            else:
                continue

            gen_val = float(preds[metric])
            baseline = float(baseline)

            if baseline != 0.0 and np.isfinite(baseline):
                ratios[f'{metric}_ratio'] = gen_val / baseline
            else:
                print(f"WARNING: Reference baseline for {metric} is 0 or non-finite. Skipping its ratio.")

        if ratios:
            avg = float(np.mean(list(ratios.values())))
        else:
            avg = -1.0
            print("WARNING: No valid ratios computed.")

        core_metrics = ["degree", "clustering", "orbit", "spectre", "wavelet"]
        partial_keys = [f'{m}_ratio' for m in core_metrics if f'{m}_ratio' in ratios]
        if partial_keys:
            partial_avg = float(np.mean([ratios[k] for k in partial_keys]))
        else:
            partial_avg = -1.0
            print("WARNING: No valid partial ratios computed (none of the core metrics had ratios).")

        local_metrics = ["degree", "clustering", "orbit", "motif"]
        local_keys = [f'{m}_ratio' for m in local_metrics if f'{m}_ratio' in ratios]
        if local_keys:
            local_avg = float(np.mean([ratios[k] for k in local_keys]))
        else:
            local_avg = -1.0
            print("WARNING: No valid partial ratios computed (none of the core metrics had ratios).")

        global_metrics = ["diameter", "modularity", "conductance", "spectre"]
        global_keys = [f'{m}_ratio' for m in global_metrics if f'{m}_ratio' in ratios]
        if global_keys:
            global_avg = float(np.mean([ratios[k] for k in global_keys]))
        else:
            global_avg = -1.0
            print("WARNING: No valid partial ratios computed (none of the core metrics had ratios).")

        to_log[f'{name}/average_ratio'] = avg
        to_log[f'{name}/partial_average_ratio'] = partial_avg
        to_log[f'{name}/local_average_ratio'] = local_avg
        to_log[f'{name}/global_average_ratio'] = global_avg
        for k, v in ratios.items():
            to_log[f'{name}/{k}'] = float(v)


    def forward(self, generated_graphs: list, name, current_epoch, val_counter, local_rank, test=False, coarse=False):
        train_graphs = self.train_graphs_coarse if coarse else self.train_graphs
        if coarse:
            reference_graphs = self.test_graphs_coarse if test else self.val_graphs_coarse
        else:
            reference_graphs = self.test_graphs if test else self.val_graphs

        if local_rank == 0:
            print(
                f"Computing sampling metrics between {len(generated_graphs)} generated graphs and {len(reference_graphs)} test graphs -- emd computation: {self.compute_emd}")

        networkx_graphs = [nx.from_numpy_array(edge_types.bool().cpu().numpy()) for graph in generated_graphs for
                           node_types, edge_types in [graph]]
        adjacency_matrices = [edge_types.bool().cpu().numpy() for graph in generated_graphs for node_types, edge_types
                              in [graph]]

        np.savez('generated_adjs.npz', *adjacency_matrices)

        to_log = {}
        for metric in self.metrics_list:
            self.compute_and_log_metric(metric, reference_graphs, networkx_graphs, to_log, name, wandb, local_rank)

        if any(m in self.metrics_list for m in ('sbm', 'planar', 'tree')):
            print("Computing all fractions...")
            test_fn = None
            if "sbm" in self.metrics_list:
                test_fn = is_sbm_graph
            elif "planar" in self.metrics_list:
                test_fn = is_planar_graph
            elif "tree" in self.metrics_list:
                test_fn = nx.is_tree

            frac_unique, frac_unique_non_isomorphic, fraction_unique_non_isomorphic_valid = eval_fraction_unique_non_isomorphic_valid(
                networkx_graphs, train_graphs, test_fn)
            frac_non_isomorphic = 1.0 - eval_fraction_isomorphic(networkx_graphs, train_graphs)
            to_log.update({
                'sampling/frac_unique': frac_unique,
                'sampling/frac_unique_non_iso': frac_unique_non_isomorphic,
                'sampling/frac_unic_non_iso_valid': fraction_unique_non_isomorphic_valid,
                'sampling/frac_non_iso': frac_non_isomorphic
            })
            unique = eval_fraction_unique(networkx_graphs, precise=True)
            to_log[f'{name}/VUN'] = fraction_unique_non_isomorphic_valid
            to_log[f'{name}/Novelty'] = frac_non_isomorphic
            to_log[f'{name}/Unique'] = unique

        if local_rank == 0:
            print("Computing ratio metric...")

        preds = {
            m: to_log[f'{name}/{m}']
            for m in self.metrics_list
            if f'{name}/{m}' in to_log and np.isfinite(to_log[f'{name}/{m}'])
        }

        self.compute_ratios(preds, reference_graphs, train_graphs, to_log, name)

        if local_rank == 0:
            print("Sampling statistics", to_log)

        if wandb.run:
            wandb.log(to_log, commit=False)

        if not test:
            return to_log

    # def forward(self, generated_graphs: list, name, current_epoch, val_counter, local_rank, test=False, coarse=False):
    #     if coarse:
    #         reference_graphs = self.test_graphs_coarse if test else self.val_graphs_coarse
    #     else:
    #         reference_graphs = self.test_graphs if test else self.val_graphs
    #
    #     if local_rank == 0:
    #         print(f"Computing sampling metrics between {len(generated_graphs)} generated graphs and {len(reference_graphs)}"
    #               f" test graphs -- emd computation: {self.compute_emd}")
    #     networkx_graphs = []
    #     adjacency_matrices = []
    #     if local_rank == 0:
    #         print("Building networkx graphs...")
    #     for graph in generated_graphs:
    #         node_types, edge_types = graph
    #         A = edge_types.bool().cpu().numpy()
    #         adjacency_matrices.append(A)
    #
    #         nx_graph = nx.from_numpy_array(A)
    #         networkx_graphs.append(nx_graph)
    #
    #     np.savez('generated_adjs.npz', *adjacency_matrices)
    #
    #     to_log = {}
    #     if 'degree' in self.metrics_list:
    #         if local_rank == 0:
    #             print("Computing degree stats..")
    #         degree = degree_stats(reference_graphs, networkx_graphs, is_parallel=False,
    #                               compute_emd=self.compute_emd)
    #         to_log[f'{name}/degree'] = degree
    #         if wandb.run:
    #             wandb.run.summary['degree'] = degree
    #
    #     if 'spectre' in self.metrics_list:
    #         if local_rank == 0:
    #             print("Computing spectre stats...")
    #         spectre = spectral_stats(reference_graphs, networkx_graphs, is_parallel=False, n_eigvals=-1,
    #                                  compute_emd=self.compute_emd)
    #
    #         to_log[f'{name}/spectre'] = spectre
    #         if wandb.run:
    #           wandb.run.summary['spectre'] = spectre
    #
    #     if 'clustering' in self.metrics_list:
    #         if local_rank == 0:
    #             print("Computing clustering stats...")
    #         clustering = clustering_stats(reference_graphs, networkx_graphs, bins=100, is_parallel=False,
    #                                       compute_emd=self.compute_emd)
    #         to_log[f'{name}/clustering'] = clustering
    #         if wandb.run:
    #             wandb.run.summary['clustering'] = clustering
    #
    #     if 'motif' in self.metrics_list:
    #         if local_rank == 0:
    #             print("Computing motif stats")
    #         motif = motif_stats(reference_graphs, networkx_graphs, motif_type='4cycle', ground_truth_match=None, bins=100,
    #                             compute_emd=self.compute_emd)
    #         to_log['motif'] = motif
    #         if wandb.run:
    #             wandb.run.summary['motif'] = motif
    #
    #     if 'orbit' in self.metrics_list:
    #         if local_rank == 0:
    #             print("Computing orbit stats...")
    #         orbit = orbit_stats_all(reference_graphs, networkx_graphs, compute_emd=self.compute_emd)
    #         to_log[f'{name}/orbit'] = orbit
    #         if wandb.run:
    #             wandb.run.summary['orbit'] = orbit
    #
    #     if 'wavelet' in self.metrics_list:
    #         if local_rank == 0:
    #             print("Computing wavelet stats...")
    #         reference_eigvals, reference_eigvecs = compute_list_eigh(self.train_graphs)
    #         predicted_eigvals, predicted_eigvecs = compute_list_eigh(networkx_graphs)
    #         wavelet = spectral_filter_stats(
    #             reference_eigvecs, reference_eigvals, predicted_eigvecs, predicted_eigvals
    #         )
    #         to_log[f'{name}/wavelet'] = wavelet
    #         if wandb.run:
    #             wandb.run.summary['wavelet'] = wavelet
    #
    #     if 'sbm' in self.metrics_list:
    #         if local_rank == 0:
    #             print("Computing accuracy...")
    #         acc = eval_acc_sbm_graph(networkx_graphs, refinement_steps=100, strict=True)
    #         to_log[f'{name}/sbm_acc'] = acc
    #         if wandb.run:
    #             wandb.run.summary['sbmacc'] = acc
    #
    #     if 'planar' in self.metrics_list:
    #         if local_rank ==0:
    #             print('Computing planar accuracy...')
    #         planar_acc = eval_acc_planar_graph(networkx_graphs)
    #         to_log[f'{name}/planar_acc'] = planar_acc
    #         if wandb.run:
    #             wandb.run.summary['planar_acc'] = planar_acc
    #
    #     if 'tree' in self.metrics_list:
    #         if local_rank ==0:
    #             print('Computing tree accuracy...')
    #         tree_acc = eval_acc_tree_graph(networkx_graphs)
    #         to_log[f'{name}/tree_acc'] = tree_acc
    #         if wandb.run:
    #             wandb.run.summary['tree_acc'] = tree_acc
    #
    #     if any(m in self.metrics_list for m in ('sbm', 'planar', 'tree')):
    #         if local_rank == 0:
    #             print("Computing all fractions...")
    #         if "sbm" in self.metrics_list:
    #             test_fn = is_sbm_graph
    #         elif "planar" in self.metrics_list:
    #             test_fn = is_planar_graph
    #         elif "tree" in self.metrics_list:
    #             test_fn = nx.is_tree
    #         else:
    #             raise ValueError('Error: Not a type of graph data')
    #
    #         frac_unique, frac_unique_non_isomorphic, fraction_unique_non_isomorphic_valid = eval_fraction_unique_non_isomorphic_valid(
    #             networkx_graphs, self.train_graphs, test_fn)
    #         frac_non_isomorphic = 1.0 - eval_fraction_isomorphic(networkx_graphs, self.train_graphs)
    #         to_log.update({'sampling/frac_unique': frac_unique,
    #                        'sampling/frac_unique_non_iso': frac_unique_non_isomorphic,
    #                        'sampling/frac_unic_non_iso_valid': fraction_unique_non_isomorphic_valid,
    #                        'sampling/frac_non_iso': frac_non_isomorphic})
    #         unique = eval_fraction_unique(networkx_graphs, precise=True)
    #         to_log[f'{name}/VUN'] = fraction_unique_non_isomorphic_valid
    #         to_log[f'{name}/Novelty'] = frac_non_isomorphic
    #         to_log[f'{name}/Unique'] = unique
    #
    #         if local_rank == 0:
    #             print("Computing ratio metric...")
    #
    #         preds = np.array([
    #             to_log[f'{name}/degree'],
    #             to_log[f'{name}/clustering'],
    #             to_log[f'{name}/orbit'],
    #             to_log[f'{name}/spectre'],
    #             to_log[f'{name}/wavelet'],
    #         ], dtype=float)
    #
    #         base_deg = degree_stats(reference_graphs, self.train_graphs,
    #                                 is_parallel=False, compute_emd=self.compute_emd)
    #         base_clus = clustering_stats(reference_graphs, self.train_graphs,
    #                                      bins=100, is_parallel=False,compute_emd=self.compute_emd)
    #         base_orb = orbit_stats_all(reference_graphs, self.train_graphs, compute_emd=self.compute_emd)
    #         base_spec = spectral_stats(reference_graphs, self.train_graphs,
    #                                    is_parallel=False, n_eigvals=-1, compute_emd=self.compute_emd)
    #         ref_eigvals, ref_eigvecs = compute_list_eigh(reference_graphs, is_parallel=False)
    #         train_eigvals, train_eigvecs = compute_list_eigh(self.train_graphs, is_parallel=False)
    #         base_wave = spectral_filter_stats(ref_eigvecs, ref_eigvals,
    #                                           train_eigvecs, train_eigvals,
    #                                           is_parallel=False, compute_emd=self.compute_emd)
    #         baselines = np.array([base_deg, base_clus, base_orb, base_spec, base_wave], dtype=float)
    #
    #         ratios = {}
    #         metrics_keys = ['degree', 'clustering', 'orbit', 'spectre', 'wavelet']
    #         for i, key in enumerate(metrics_keys):
    #             try:
    #                 ref_metric = baselines[i]
    #                 gen_metric = preds[i]
    #
    #                 if ref_metric != 0.0:
    #                     ratios[f'{key}_ratio'] = gen_metric / ref_metric
    #                 else:
    #                     print(f"WARNING: Reference {key} is 0. Skipping its ratio.")
    #
    #             except IndexError:
    #                 print(f"ERROR: Index out of range for metric: {key}")
    #
    #         if len(ratios) > 0:
    #             ratios["average_ratio"] = np.mean(list(ratios.values()))
    #         else:
    #             ratios["average_ratio"] = -1
    #             print("WARNING: No valid ratios computed.")
    #
    #         to_log[f'{name}/average_ratio'] = ratios["average_ratio"]
    #
    #         for key, ratio in ratios.items():
    #             if key != "average_ratio":
    #                 to_log[f'{name}/{key}'] = ratio
    #
    #     if 'comm20' in self.metrics_list:
    #         if local_rank == 0:
    #             print("Computing ratio metric…")
    #
    #         preds = np.array([
    #             to_log[f'{name}/degree'],
    #             to_log[f'{name}/clustering'],
    #             to_log[f'{name}/orbit'],
    #             to_log[f'{name}/spectre'],
    #             to_log[f'{name}/wavelet'],
    #         ], dtype=float)
    #
    #         # compute the same baselines against your reference set
    #         base_deg = degree_stats(reference_graphs, self.train_graphs, is_parallel=False,
    #                                 compute_emd=self.compute_emd)
    #         base_clus = clustering_stats(reference_graphs, self.train_graphs, bins=100, is_parallel=False,
    #                                      compute_emd=self.compute_emd)
    #         base_orb = orbit_stats_all(reference_graphs, self.train_graphs, compute_emd=self.compute_emd)
    #         base_spec = spectral_stats(reference_graphs, self.train_graphs, is_parallel=False, n_eigvals=-1,
    #                                    compute_emd=self.compute_emd)
    #         ref_eigvals, ref_eigvecs = compute_list_eigh(reference_graphs, is_parallel=False)
    #         train_eigvals, train_eigvecs = compute_list_eigh(self.train_graphs, is_parallel=False)
    #         base_wave = spectral_filter_stats(ref_eigvecs, ref_eigvals, train_eigvecs, train_eigvals,
    #                                           is_parallel=False, compute_emd=self.compute_emd)
    #
    #         baselines = np.array([base_deg, base_clus, base_orb, base_spec, base_wave], dtype=float)
    #
    #         mean_gen = np.mean(preds)
    #         mean_base = np.mean(baselines)
    #         ratio = float(mean_gen / (mean_base + 1e-8))
    #
    #         to_log[f'{name}/ratio'] = ratio
    #         if wandb.run:
    #             wandb.run.summary['ratio'] = ratio
    #
    #     if local_rank == 0:
    #         print("Sampling statistics", to_log)
    #     if wandb.run:
    #         wandb.log(to_log, commit=False)
    #     if not test:
    #         return to_log

    def reset(self):
        pass


class Comm20SamplingMetrics(SpectreSamplingMetrics):
    def __init__(self, datamodule):
        super().__init__(datamodule=datamodule,
                         compute_emd=True,
                         metrics_list=['degree', 'clustering', 'orbit', 'spectre', 'wavelet', 'comm20',
                                       'motif', 'diameter', 'aspl', 'components', 'edge_connectivity',
                                       'modularity', 'conductance'
                                       ])


class PlanarSamplingMetrics(SpectreSamplingMetrics):
    def __init__(self, datamodule):
        super().__init__(datamodule=datamodule,
                         compute_emd=False,
                         metrics_list=['degree', 'clustering', 'orbit', 'spectre', 'wavelet', 'planar',
                                       'motif', 'diameter', 'aspl', 'components', 'edge_connectivity',
                                       'modularity', 'conductance'
                                       ])


class SBMSamplingMetrics(SpectreSamplingMetrics):
    def __init__(self, datamodule):
        super().__init__(datamodule=datamodule,
                         compute_emd=False,
                         metrics_list=['degree', 'clustering', 'orbit', 'spectre', 'wavelet', 'sbm',
                                       'motif', 'diameter', 'aspl', 'components', 'edge_connectivity',
                                       'modularity', 'conductance'
                                       ])


class TreeSamplingMetrics(SpectreSamplingMetrics):
    def __init__(self, datamodule):
        super().__init__(datamodule=datamodule,
                         compute_emd=False,
                         metrics_list=['degree', 'clustering', 'orbit', 'spectre', 'wavelet', 'tree',
                                       'motif', 'diameter', 'aspl', 'components', 'edge_connectivity',
                                       'modularity', 'conductance'
                                       ])
