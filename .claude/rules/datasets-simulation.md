# Datasets and Simulation

**Task:** Discrete image classification only.

**Benchmark datasets (§4.1):** CIFAR-10, CIFAR-100, FEMNIST.
MNIST and FMNIST are supported in the codebase but excluded from the benchmark scope.

**Statistical heterogeneity:**

- CIFAR-10 / CIFAR-100: Dirichlet ($\alpha$) partitioning.
- FEMNIST: Writer-based natural partitioning (`partition: writer`), not Dirichlet; use `heterogeneity=femnist experiment=femnist`.
- Exact alpha values, memory/bandwidth/compute settings, and the control-group config: see `conf/heterogeneity/` and `conf/experiment/default.yaml`.
