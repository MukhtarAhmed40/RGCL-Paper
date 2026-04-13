# RGCL: Robust Graph Contrastive Learning for Adversarial Network Intrusion Detection

This repository contains the official implementation of RGCL, a robust graph contrastive learning framework for adversarial network intrusion detection.

## Overview

RGCL is designed to learn resilient and transferable representations from evolving network traffic graphs without relying on extensive labeled data. The framework integrates three key components:

1. **Ensemble Graph Attention (EGA)**: Stabilizes feature aggregation under noisy and perturbed graph structures
2. **Adaptive Contrastive Optimization**: Preserves local node-level consistency and global graph-level semantics
3. **Dynamic Graph Update**: Incrementally adapts to evolving traffic patterns

## Requirements

```bash
pip install -r requirements.txt
