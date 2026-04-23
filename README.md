# LGDC: Latent Graph Diffusion via Spectrum-Preserving Coarsening

Official implementation of **LGDC**, accepted at the **NeurIPS 2025 New Perspectives in Advancing Graph Machine Learning Workshop**.

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

### Key properties
- **Complexity**: O(n² + T·n_c²) vs O(T·n²) for one-shot and O(T·n²/3) for autoregressive
- **Spectrum-preserving**: coarsening keeps principal Laplacian eigenvalues/eigenspaces close between original and coarsened graphs (restricted spectral similarity)
- **Single-step decoding**: one coarse-to-fine pass at both training and test time

## Results

| Model | Class | Planar V.U.N.↑ | Planar A.Ratio↓ | Tree V.U.N.↑ | Tree A.Ratio↓ | Comm-20 Deg.↓ | Comm-20 Clus.↓ | Comm-20 Orb.↓ |
|-------|-------|---------------|----------------|-------------|--------------|--------------|----------------|--------------|
| HSpectre | Autoregressive | 62.5 | 2.90 | 82.5 | 2.10 | — | — | — |
| DeFoG | One-shot | 77.5 | 4.07 | 73.1 | 1.50 | 0.071 | 0.115 | 0.037 |
| **LGDC (Ours)** | **Hybrid** | **82.5** | **3.06** | **86.0** | **1.70** | **0.037** | **0.027** | **0.007** |

## Repository structure

```
LGDC/
├── src/
│   ├── main.py                        # Training entry point (Hydra + PyTorch Lightning)
│   ├── diffusion_model_coarsen.py     # CoarsenedDDM — main LGDC model
│   ├── diffusion_model_discrete.py    # DiscreteDenoisingDiffusion base class
│   ├── diffusion_model.py             # LiftedDenoisingDiffusion (continuous variant)
│   ├── decoarse_model.py              # Stand-alone decoarsening/refinement model
│   ├── utils.py
│   ├── analysis/                      # Evaluation metrics & visualization
│   ├── datasets/                      # Dataset implementations
│   │   └── coarsen_spectre_dataset.py # Coarsened graph datasets (main dataset class)
│   ├── diffusion/                     # Noise schedules, distributions, utils
│   ├── expansion/                     # Graph expansion & one-shot decode
│   ├── graph_coarsen/                 # Spectral coarsening (REC algorithm)
│   ├── metrics/                       # Training and evaluation metrics
│   └── models/                        # Neural network architectures
│       ├── graphformer_uncon.py       # Graph Transformer (diffusion backbone)
│       ├── sparse_ppgn.py             # Sparse PPGN (refinement model)
│       ├── ppgn.py                    # PPGN
│       └── ...
└── configs/                           # Hydra configuration files
    ├── config.yaml
    ├── experiment/                    # Ready-to-run experiment configs
    │   ├── coarsen_planar.yaml
    │   ├── coarsen_sbm.yaml
    │   ├── coarsen_tree.yaml
    │   └── coarsen_comm20.yaml
    ├── dataset/
    ├── model/
    ├── train/
    ├── decoarse/
    └── reduction/
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

```bash
cd src

# Debug run (recommended first)
python main.py +experiment=debug.yaml

# Planar graphs (coarsening pipeline)
python main.py +experiment=coarsen_planar.yaml

# Tree graphs
python main.py +experiment=coarsen_tree.yaml

# SBM / Community graphs
python main.py +experiment=coarsen_sbm.yaml
python main.py +experiment=coarsen_comm20.yaml
```

Override any config parameter on the command line:
```bash
python main.py +experiment=coarsen_planar.yaml train.batch_size=16 train.lr=0.0002
```

## Data

Datasets are generated/downloaded automatically the first time you run an experiment. Processed files are cached under `data/` (excluded from version control due to size).

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
