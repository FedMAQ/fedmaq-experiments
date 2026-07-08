# Evaluation Metrics

Log to WandB for every experiment run:

1. **Top-1 test accuracy (%)**
2. **Cross-entropy loss** and **distillation loss** (when KD active)
3. **Auxiliary classification metrics** — Precision, Recall, and F1-Score (macro-averaged)
4. **Cumulative communication overhead** (MB/GB per client and aggregate)
5. **Wall-clock runtime** (seconds)
6. **Convergence stability** — curves of accuracy vs. rounds and accuracy vs. transmitted bytes
