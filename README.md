# LGDC: Latent Graph Diffusion via Spectrum-Preserving Coarsening

Official implementation of **[LGDC](https://arxiv.org/pdf/2512.01190)**, accepted at the **NeurIPS 2025 New Perspectives in Advancing Graph Machine Learning Workshop**.

> **LGDC: Latent Graph Diffusion via Spectrum-Preserving Coarsening**  
> Nagham Osman, Keyue Jiang, Davide Buffelli, Xiaowen Dong, Laura Toni  

## Overview

Graph generation methods fall into two categories: autoregressive models (iterative local expansion) and one-shot models (diffusion). Each paradigm has complementary strengths:

- **Autoregressive** models excel at fine-grained **local** structure (degree, clustering)
- **One-shot / diffusion** models excel at **global** structure (spectral distributions, community organization)

**LGDC** is a hybrid framework that combines both. It uses spectrum-preserving coarsening to map a graph into a compact latent space, runs discrete diffusion there for efficient global modeling, then expands back to the original size via a single autoregressive-inspired refinement step.

```
G  ──coarsen──►  G_c  ──diffuse/denoise──►  Ĝ_c  ──expand/refine──►  Ĝ
```

## Installation

This code was tested with Python 3.9, PyTorch 2.0.1, CUDA 11.8, and torch_geometric 2.3.1.

**1. Create a conda environment with RDKit:**
```bash
conda create -c conda-forge -n lgdc rdkit=2023.03.2 python=3.9
conda activate lgdc
```

**2. Install graph-tool:**
```bash
conda install -c conda-forge graph-tool=2.45
```

**3. Install CUDA and PyTorch:**
```bash
conda install -c "nvidia/label/cuda-11.8.0" cuda
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
```

**4. Install remaining dependencies:**
```bash
pip install -r requirements.txt
pip install -e .
```

**5. Compile ORCA (graph orbit statistics):**
```bash
cd src/analysis/orca
g++ -O2 -std=c++11 -o orca orca.cpp
```

> Note: graph_tool and torch_geometric may conflict on macOS.

## Running experiments

All experiments are launched through `src/main.py` using [Hydra](https://hydra.cc/) for configuration.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{osman2025lgdc,
  title     = {{LGDC}: Latent Graph Diffusion via Spectrum-Preserving Coarsening},
  author    = {Osman, Nagham and Jiang, Keyue and Buffelli, Davide and Dong, Xiaowen and Toni, Laura},
  booktitle = {NeurIPS 2025 New Perspectives in Advancing Graph Machine Learning Workshop},
  year      = {2025}
}
```

This codebase builds on [DiGress](https://github.com/cvignac/DiGress) (Vignac et al., ICLR 2023) and [HSpectre](https://github.com/AndreaBe99/gnn_graph_unpool) (Bergmeister et al., ICLR 2024).

## License

See [LICENSE](LICENSE).
