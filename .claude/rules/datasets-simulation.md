# Datasets and Simulation

**Task:** Discrete image classification only.

**Benchmark datasets (§4.1):** CIFAR-10, CIFAR-100, FEMNIST.
MNIST and FMNIST are supported in the codebase but excluded from the benchmark scope.

**Statistical heterogeneity:**

- CIFAR-10 / CIFAR-100: Dirichlet ($\alpha$) partitioning.
- FEMNIST: Writer-based natural partitioning (`partition: writer`) — no Dirichlet; use `heterogeneity=femnist experiment=femnist`.
- Benchmark $\alpha \in \{0.1, 1.0\}$ (high and moderate skew). $\alpha = 10.0$ is excluded.
- Pilot formulation study: CIFAR-10 only, $\alpha = 0.1$, variable memory $\mathcal{U}(2048, 16384)$ MB.

**System configuration:** Bandwidth and compute are uniform across all clients
(10 Mbps bandwidth, 200 samples/sec compute speed) to isolate algorithmic factors.
Memory is heterogeneous by default: $\mathcal{U}(2048, 16384)$ MB.
Control group uses `heterogeneity=uniform_memory` (8192 MB fixed).
