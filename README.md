# mapct-v2-to-v13
# MAPCT v2–v13: Mechanism‑Aware Progressive Clinical Transformer for Drug Regimen Prediction and Generation

[![DOI](https://zenodo.org/badge/DOI/yourzenododoi.svg)](https://doi.org/yourzenododoi)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the complete source code, pre‑processing scripts, and trained model weights for the **MAPCT** framework, a controlled evolutionary study of 13 deep learning architectures for **drug regimen prediction and generation** under strict leakage‑free splitting.

**Key features**:
- 13 architectures: contrastive dual encoders, autoencoders, GANs, diffusion models, VAEs, normalizing flows, graph neural networks, and hybrid flow‑GNNs.
- Leakage‑free regimen splitting (no drug combination appears in both training and test sets).
- Comprehensive evaluation: drug prediction (F1, precision, recall), clinical‑regimen latent alignment (cosine similarity), and generative diversity (Jaccard distance).
- Extensive visualisations: PCA, t‑SNE, UMAP, alignment distributions, training curves.

## 📁 Repository Structure
