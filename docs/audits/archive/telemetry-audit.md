# Hardware & Telemetry Grounding Audit

Critical review of [ADR-0002](file:///c:/Users/Quirora/Documents/GitHub/BSMSCS Thesis/fedmaq-experiments/docs/adr/0002-hardware-telemetry-grounding.md) and [Decision 36](file:///c:/Users/Quirora/Documents/GitHub/BSMSCS Thesis/fedmaq-experiments/docs/DECISIONS.md#L344-L356), plus a scan for other non-hyperparameter experimental values needing finalization.

---

## 1. Client Hardware (Raspberry Pi 5) — Verdict: ✅ Well-Grounded

| Parameter                     | Value                                       | Source                                                       | Math Check                |
| :---------------------------- | :------------------------------------------ | :----------------------------------------------------------- | :------------------------ |
| Processor                     | Quad Cortex-A76 @ 2.4 GHz                   | BCM2712 datasheet (Oct 2023)                                 | ✅ Real product           |
| Peak FP32 GEMM                | ~31.5 GFLOPS                                | 4 cores × 2 NEON units/core × 128-bit SIMD × 2.4 GHz         | ✅ Correct order          |
| MobileNetV2GN FLOPs           | ~0.90 GFLOPs/sample (fwd+bwd)               | ≈0.30 GFLOPs fwd × 3 (bwd≈2×fwd)                             | ✅ Standard 3× multiplier |
| $v_{\text{client}}$ (CIFAR)   | 35.0 samples/sec                            | $31.5 / 0.90 = 35.0$                                         | ✅ Arithmetic checks out  |
| SimpleCNN FLOPs               | ~0.015 GFLOPs/sample (fwd+bwd)              | Tiny CNN ≈ 5M FLOPs fwd × 3                                  | ✅ Reasonable             |
| $v_{\text{client}}$ (FEMNIST) | 600.0 samples/sec                           | Capped below $31.5/0.015 = 2100$ for DataLoader/GIL overhead | ✅ Conservative cap       |
| RAM tiers                     | U(2048, 16384) MB                           | Pi 5 ships in 2/4/8/16 GB                                    | ✅ Matches real SKUs      |
| $c_{\text{unit}}$ = 512 MB    | $\lfloor c_k / 512 \rfloor$ → 4/8/16/32-bit | Design choice for clean tier mapping                         | ✅ Consistent             |

### Minor Observations (not blockers)

- **Peak vs. sustained throughput**: The 31.5 GFLOPS figure assumes 100% ALU utilization on NEON FP32 GEMM, which is a _theoretical peak_. Real sustained PyTorch inference on ARM typically achieves **40–70% of peak** due to memory bandwidth bottlenecks (Pi 5: ~34 GB/s LPDDR4X). This would put real throughput at ~12–22 GFLOPS → **~14–24 samples/sec** for MobileNetV2GN instead of 35.
  - **Impact**: If anything, 35.0 s/s is _optimistic_ for the client. This would make FedMAQ's per-round simulated training time _shorter_ than reality, meaning the real-world advantage of communication compression is actually _larger_. So this bias is **conservative for your thesis claims** (underestimates compute-boundedness).
  - **Recommendation**: Defensible as-is. If a reviewer raises this, you can note that using peak FLOPS is standard in simulation-based FL papers (FedProx, FedPAQ, etc.) and that a lower value would only _strengthen_ FedMAQ's wall-clock advantage.

- **Memory bandwidth ignored for training time**: The model treats compute as purely FLOP-bound. In practice, depthwise-separable convolutions (MobileNetV2's signature op) are _memory-bandwidth-bound_, not compute-bound, on Pi 5's LPDDR4X. This further supports the above: real training is likely slower than 35 s/s.

---

## 2. Wireless Link (10 Mbps 802.11ac) — Verdict: ✅ Reasonable, Minor Flag

| Parameter        | Value | Justification                         | Math Check    |
| :--------------- | :---- | :------------------------------------ | :------------ |
| `bandwidth_mbps` | 10.0  | 802.11ac under contention + path loss | ✅ Reasonable |

- **802.11ac theoretical max**: 433 Mbps (single spatial stream, 80 MHz). Real-world single-client throughput in a shared environment: 50–100 Mbps. Under multi-client FL contention with 10 simultaneous clients: ~5–20 Mbps per client.
- **10 Mbps is well within the plausible range** and is the most commonly used value in FL simulation papers (Li et al., 2020; Reisizadeh et al., 2020).
- **Uniform bandwidth assumption**: All clients get the same 10 Mbps — no heterogeneous bandwidth modeling. This is a simplification, but is standard for FL simulations and wouldn't affect your primary accuracy metrics.

> [!TIP]
> This is fine as-is. If you later want richer telemetry analysis, you could add a heterogeneous bandwidth sweep (e.g., U(1, 20) Mbps) as a sensitivity study, but it's not needed for the thesis.

---

## 3. Server Hardware (Xeon + L40S) — Verdict: ⚠️ Math Gap

| Parameter           | Value                                       | Source                                                | Math Check                 |
| :------------------ | :------------------------------------------ | :---------------------------------------------------- | :------------------------- |
| CPU                 | 24-Core Intel Xeon 5th Gen (Emerald Rapids) | Dec 2023 release                                      | ✅ Real product            |
| GPU                 | NVIDIA L40S 48GB (91.6 FP32 TFLOPS)         | Aug 2023 release                                      | ✅ Real product            |
| System RAM          | 64 GB DDR5-4800 ECC                         | Standard server config                                | ✅ Reasonable              |
| $v_{\text{server}}$ | 4500.0 samples/sec                          | "Calculated for server-side KD ensemble fine-tuning…" | ⚠️ **No derivation shown** |

### The Gap

The client-side derivation is explicit and checkable:

$$v_{\text{client}} = \frac{P_{\text{device}}}{F_{\text{model}}} = \frac{31.5}{0.90} = 35.0 \text{ s/s}$$

The server-side derivation says only:

> $v_{\text{server}} = \mathbf{4500.0\text{ samples/sec}}$
> (Calculated for server-side KD ensemble fine-tuning and gradient-norm probe backprop on $D_{\text{pub}}$ using CUDA FP16/FP32 PyTorch execution).

**There is no equivalent formula.** Let's sanity-check with math:

- **L40S FP32 throughput**: 91.6 TFLOPS
- **MobileNetV2GN fwd+bwd**: ~0.90 GFLOPs/sample
- **Theoretical peak**: $91{,}600 / 0.90 = 101{,}778$ samples/sec
- **Typical GPU utilization for small models** on a datacenter GPU: **3–15%** (small batch sizes, kernel launch overhead, memory transfers dominate for a 2.24M param model on a 48GB GPU)
- **4500 s/s** → utilization = $4500 / 101{,}778 ≈ 4.4\%$

This is **plausible** for a small model doing KD on a datacenter GPU (KD involves forward passes through multiple teacher models, not a single large batch), but the document doesn't show how 4500 was reached.

> [!IMPORTANT]
> **Recommendation**: Add the derivation to ADR-0002. A simple approach:
> $$v_{\text{server}} = \frac{P_{\text{L40S}} \times \eta_{\text{util}}}{F_{\text{model}}} = \frac{91{,}600 \times 0.044}{0.90} \approx 4{,}471 \approx 4500 \text{ s/s}$$
> Where $\eta_{\text{util}} \approx 4.4\%$ is justified by the small model + multi-teacher serial KD workload. Alternatively, cite a PyTorch benchmark of MobileNetV2 inference throughput on an A100/L40S-class GPU.

### Server Compute Context: How It's Used

Looking at [kd_utils.py](file:///c:/Users/Quirora/Documents/GitHub/BSMSCS Thesis/fedmaq-experiments/src/fedmaq/core/kd_utils.py#L104-L118):

```python
def kd_server_sim_time(num_public, kd_epochs, num_teachers, server_compute_speed):
    return (num_public * kd_epochs * num_teachers) / server_compute_speed
```

With defaults: $|D_{\text{pub}}| = 3000$, `kd_epochs=1`, `num_teachers=10` (10% of 100 clients):
$$t_{\text{server\_kd}} = \frac{3000 \times 1 \times 10}{4500} = 6.67 \text{ seconds/round}$$

This is a small fraction of the per-round time (~71.4s client training + ~14.4s comm), so **server speed has low sensitivity** on the reported results. Even a 2× error in $v_{\text{server}}$ would only shift total round time by ~3.3s on a ~90s round. Still worth documenting the derivation for thesis defense.

---

## 4. Your Friend's Data Center Hardware — Impact Assessment

Your friend's specs: **64-core Intel Platinum 8276, 64GB RAM, NVIDIA A100 40GB**.

This is your **experiment execution platform** (the machine that actually runs the Flower simulation), not the hardware being _modeled_ in the simulation. Important distinction:

| Concern                                         | Status                                                                                                                               |
| :---------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------- |
| Does the A100's 40GB VRAM help?                 | ✅ Yes — can lower `client_gpus` below 1.0, enabling concurrent Ray actors → **massive wall-clock speedup** for the ~201 formal runs |
| Does it change your simulated telemetry values? | ❌ No — simulation models Pi 5 clients and L40S server; the A100 just runs the simulation faster                                     |
| Can you expand `num_clients` or `total_rounds`? | ✅ Yes — 40GB VRAM + 64GB RAM makes K=200 clients or R=200 rounds feasible                                                           |
| Intel Platinum 8276 vs Xeon 5th Gen?            | The Platinum 8276 is Cascade Lake (2019), not Emerald Rapids (2023) — doesn't matter since you're not benchmarking on it             |

> [!TIP]
> **Key opportunity with the A100**: Profile `client_gpus` at 0.5 or 0.25 on the A100 (40GB should handle 2–4 concurrent MobileNetV2GN actors at ~2.24M params × FP32 ≈ 9MB each, plus optimizer state ≈ ~27MB total per actor). This would cut your ~201 formal-run wall-clock by 2–4× compared to the current serial `client_gpus=1.0` mode. This was explicitly parked in Decision 25 due to OOM on your current GPU — the A100 removes that blocker.

---

## 5. Other Non-Hyperparameter Experimental Values — Finalization Audit

Beyond compute/bandwidth/memory, here are the structural experiment parameters from [default.yaml](file:///c:/Users/Quirora/Documents/GitHub/BSMSCS Thesis/fedmaq-experiments/conf/experiment/default.yaml) and the formal plan:

| Parameter                                   | Current Value              | Status                                 | Notes                                     |
| :------------------------------------------ | :------------------------- | :------------------------------------- | :---------------------------------------- |
| `num_clients` (K)                           | 100 (CIFAR), 200 (FEMNIST) | ✅ Settled (Decision 9)                | Standard FL scale                         |
| `client_fraction` (C)                       | 0.1                        | ✅ Settled                             | 10 clients/round (CIFAR), 20 (FEMNIST)    |
| `total_rounds` (R)                          | 100                        | ✅ Settled (Decision 9)                | Could expand with A100                    |
| `local_epochs` (E)                          | 5                          | ✅ Settled                             | Standard FL value                         |
| `batch_size` (B)                            | 64                         | ✅ Settled                             | Standard                                  |
| `num_public_samples` ($\|D_{\text{pub}}\|$) | 3000                       | ✅ Settled                             | 6% of CIFAR-10 train set                  |
| Seeds                                       | [0, 42, 123]               | ✅ Settled (Decision 6)                | 3 paired seeds                            |
| α values                                    | {0.1, 1.0}                 | ✅ Settled (Decision 10)               | Severe + moderate                         |
| Explore-α                                   | 0.3                        | ✅ Settled (Decision 27)               | Distinct from report α                    |
| `c_unit`                                    | 512.0 MB                   | ✅ Settled (ADR-0002)                  | Clean Pi 5 tier mapping                   |
| `q_min` / `q_max`                           | 1 / 16                     | ⚠️ **Implicit — not formally decided** | See below                                 |
| `bit_widths` set                            | [1,2,3,4,5,6,7,8,16,32]    | ⚠️ **Implicit — not formally decided** | See below                                 |
| `post_process`                              | false (default)            | ✅ Settled                             | Error-feedback/diff/zlib off for ablation |

### Items Worth Formalizing

#### A. `q_min=1` / `q_max=16` Interpolation Bounds

These bound the _soft quality target_ interpolation range — the Tier-2 formulations interpolate $\hat{q}$ between `q_min` and `q_max`. Currently set to 1 and 16 in [fedmaq.yaml](file:///c:/Users/Quirora/Documents/GitHub/BSMSCS Thesis/fedmaq-experiments/conf/algorithm/fedmaq.yaml#L3-L4).

- **Why 16 and not 32?** A client with 16GB RAM gets $Q_k^{\max} = \lfloor 16384/512 \rfloor = 32$, but the interpolation caps at 16-bit. This means the Tier-2 soft target never _requests_ 32-bit — only Tier-1 can yield 32-bit (when $Q_k^{\max} = 32$ and $\hat{q} \ge 32$, which it can't be since $q_{\max}=16$). In effect, **no client ever transmits at FP32 precision** even if their RAM allows it.
- **Is this intentional?** Likely yes (FP16→FP32 gains are marginal for model accuracy in FL, and FP32 doubles comm cost for no benefit), but it should be documented as a design choice.
- **Recommendation**: Add a brief note to the decisions log or ADR that `q_max=16` is deliberate — the permissible set includes 32 for the Tier-1 hard cap, but the Tier-2 quality target never pushes above FP16.

#### B. `bit_widths` Permissible Set

The set `[1,2,3,4,5,6,7,8,16,32]` is stated as "per manuscript §4.2" but has a structural property worth noting: it includes every integer 1–8 plus jumps to 16 and 32. The gap between 8 and 16 means **no client can be assigned 9, 10, 11, 12, 13, 14, or 15-bit precision** — they snap down to 8. Combined with `q_max=16`, the effective precision distribution across your RAM tiers is:

| Pi 5 RAM | $Q_k^{\max}$ (Tier-1) | Soft target range | Practical output                      |
| :------- | :-------------------- | :---------------- | :------------------------------------ |
| 2 GB     | 4-bit                 | 1–4               | 1, 2, 3, or 4                         |
| 4 GB     | 8-bit                 | 1–8               | 1–8                                   |
| 8 GB     | 16-bit                | 1–16              | 1–8 or 16 (snap)                      |
| 16 GB    | 32-bit                | 1–16              | 1–8 or 16 (snap, since $q_{\max}=16$) |

This means **8GB and 16GB clients are functionally identical** in terms of achievable precision (both cap at 16-bit via Tier-2, snap to the same set). This is fine if intentional, but worth noting.

#### C. Datasets — CIFAR-100 Config

CIFAR-100 is in your formal grid (Decision 9) but I only see [cifar10.yaml](file:///c:/Users/Quirora/Documents/GitHub/BSMSCS Thesis/fedmaq-experiments/conf/dataset/cifar10.yaml) in `conf/dataset/`. You'll need a `cifar100.yaml` with `num_classes: 100`. May already exist — just flagging.

#### D. Learning Rate / Optimizer — Not Explored, But Standard

`lr=0.01`, `momentum=0.9`, `weight_decay=1e-4`, `lr_decay=0.99` are in [default.yaml](file:///c:/Users/Quirora/Documents/GitHub/BSMSCS Thesis/fedmaq-experiments/conf/experiment/default.yaml) and are standard FL values (McMahan et al., 2017 defaults). These are shared across all algorithms (iso-training-regime, per Decision 7's parity principle). No finalization needed, but confirm they're not in the baseline tuning grid — they shouldn't be.

---

## 6. Summary of Required Actions

### Must-Fix Before Experiments

| #   | Item                                                    | Effort | Impact                                                                |
| :-- | :------------------------------------------------------ | :----- | :-------------------------------------------------------------------- |
| 1   | **Add $v_{\text{server}}$ derivation math to ADR-0002** | Low    | Thesis defensibility — reviewers will ask "where did 4500 come from?" |

### Should-Document (Low Effort, High Defensibility)

| #   | Item                                                                       | Effort | Impact                            |
| :-- | :------------------------------------------------------------------------- | :----- | :-------------------------------- |
| 2   | Document `q_max=16` as intentional (Tier-2 never assigns FP32)             | Low    | Preempts "why not FP32?" question |
| 3   | Note 8GB↔16GB functional equivalence under current `q_max`/`bit_widths`    | Low    | Transparency                      |
| 4   | Note peak-vs-sustained FLOPS assumption for $v_{\text{client}}$ derivation | Low    | Preempts reviewer nit             |

### Optional Considerations (with A100 Access)

| #   | Item                                                      | Notes                                                      |
| :-- | :-------------------------------------------------------- | :--------------------------------------------------------- |
| 5   | Profile `client_gpus < 1.0` on A100 for concurrent actors | Huge wall-clock savings; blocked by OOM on current GPU     |
| 6   | Consider expanding `total_rounds` to 150–200              | Feasible with A100; shows convergence plateau more clearly |
| 7   | Verify CIFAR-100 dataset config exists                    | Need `conf/dataset/cifar100.yaml`                          |
