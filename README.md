# ProDo-GNN: Centrality-Driven Training for Fast GNN

## Overview

Graph Neural Networks (GNNs) have emerged as a powerful framework for learning from graph-structured data. However, training GNNs on large-scale graphs remains challenging due to high computational cost, excessive memory consumption, and neighborhood explosion.

**ProDo-GNN** is a centrality-aware framework designed to accelerate GNN training while maintaining predictive performance. The framework combines:

- Dominating-set-based training node reduction
- Eigenvector-centrality-guided node selection
- Centrality-driven neighborhood sampling
- GPU-accelerated preprocessing using CUDA

The implementation is built on top of DGL and leverages CUDA-based GPU acceleration for preprocessing and graph analysis.

---

## Publication

**Title:** ProDo-GNN: Centrality-Driven Schemes for Fast GNN

This repository contains the artifact and implementation accompanying a submitted research paper currently under review.

---

## Abstract

Graph Neural Networks (GNNs) have become a popular machine learning toolbox for solving complex problems dealing with graph-structured data. Over the years, GNNs have achieved remarkable progress; nevertheless, they continue to face critical challenges, including limited scalability, high training costs, and substantial memory demands when applied to large-scale real-world graphs.

Several graph sparsification and sampling strategies have been proposed in the literature. However, most approaches focus either on accuracy or on improving performance, but not both. Therefore, designing effective sampling strategies that control neighborhood expansion is essential to improve training efficiency while preserving predictive accuracy.

To address these challenges, we propose **ProDo-GNN**, a centrality-aware framework for scalable GNN training. First, we reduce computational overhead by reducing the training set using eigenvector centrality in conjunction with the dominating set. Subsequently, we introduce a neighborhood sampling scheme based on node centrality.

We implemented ProDo-GNN using the DGL framework and parallelized it on GPUs using CUDA. Experimental evaluation on multiple GNN datasets demonstrates that ProDo-GNN achieves up to **3× speedup** and an average **1.8× speedup** over state-of-the-art sampling techniques while maintaining comparable accuracy.

---

## Repository Location

All scripts required to reproduce the results are located in:

```text
examples/pytorch/graphsage
```

All commands below should be executed from this directory.

---

## Requirements

### Software

```text
Python 3.8.20
DGL 2.1
PyTorch 1.13.0
CUDA 12.1
GCC 12.3.0
CuPy
NumPy
OGB
```

### Hardware Used in Evaluation

```text
CPU  : Intel Xeon Gold 5218 (16 Cores)
RAM  : 512 GB
GPU  : NVIDIA RTX A6000 (48 GB)
OS   : Ubuntu 24.04
```

---

## Supported Datasets

- Reddit
- OGBN-Arxiv
- OGBN-Products
- Yelp
- IGB-Small

---

## Preprocessing

The preprocessing stage computes weighted eigenvector centrality and generates the data structures required by ProDo-GNN.

Run:

```bash
./preprocessing_ProDo_GNN.sh
```

Generated files are stored in:

```text
dissimilarity-eigenvector-centrality/
```

---

## Running ProDo-GNN

After preprocessing, execute:

```bash
./run_ProDo-GNN.sh
```

The execution statistics and generated outputs are stored in:

```text
time_centrality/
```

---

## Experimental Workflow

```text
Dataset
   |
   v
Preprocessing
   |
   +--> Dominating Set Construction
   |
   +--> Eigenvector Centrality Computation
   |
   +--> Neighbor Reordering
   |
   v
ProDo-GNN Training
   |
   v
Accuracy and Performance Evaluation
```

---

## Expected Results

ProDo-GNN reduces training overhead by:

- Reducing the number of training nodes
- Prioritizing important neighbors using centrality
- Controlling neighborhood expansion
- Leveraging GPU parallelism during preprocessing

Across evaluated datasets, ProDo-GNN achieves:

- Up to 3× training speedup
- Average 1.8× speedup
- Comparable predictive accuracy to existing sampling approaches

---

## Contact

For questions regarding the implementation, please open an issue in this repository.
> **Note:** This repository contains the implementation of ProDo-GNN built on top of the DGL framework. The original DGL source code is retained, while all scripts required to reproduce the results are located in `examples/pytorch/graphsage`.
