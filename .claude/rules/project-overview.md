# Project Overview

**Title:** FedMAQ: Communication-Efficient Federated Learning via Multi-Adaptive Quantization and Knowledge Distillation

**Status:** Proposal-stage — details may change; update this rule when objectives shift.

## Objectives

1. Simulate FL with uniform bandwidth and compute, heterogeneous per-client memory (the Tier-1 hard-clamp resource signal), and statistical heterogeneity (non-IID data).
2. Formulate multi-adaptive gradient quantization (resource, data, and state/gradient-norm awareness).
3. Devise server-side aggregation with KD to mitigate drift and quantization error.
4. Benchmark against SOTA communication-efficient FL methods on accuracy, communication overhead, and convergence stability.

## Core Hypothesis

Integrating multiple dimensions of awareness into a unified quantization formula yields a superior communication-efficiency tradeoff versus single-factor adaptive methods.
