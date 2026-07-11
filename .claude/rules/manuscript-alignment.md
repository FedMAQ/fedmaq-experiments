# Codebase-to-Manuscript Alignment Rules

All agents and developers modifying this codebase must adhere strictly to the following mathematical and hyperparameter constraints from the LaTeX thesis manuscript (located in the `fedmaq-manuscript` sibling repository).

## 1. Hyperparameter Synchronization

Every benchmark experiment's hyperparameters must match [conf/experiment/default.yaml](../../conf/experiment/default.yaml), derived from **Table 4.1** of the manuscript; do not hardcode hyperparameters elsewhere. The permissible discrete bit-width set is $\mathcal{Q} = \{1,2,3,4,5,6,7,8,16,32\}$ — not an arbitrary continuous integer.

## 2. Quantization Soft Quality Target Formulas

Any precision-scaling evaluation logic must stay synchronized with the five mathematical formulations in **Section 3.3** of the manuscript — code and manuscript are a single source of truth; if one changes, update the other.

## 3. Decoupled Simulated Time & Overheads

The simulation time logged in `TelemetryFedAvg` must stay synchronized with the round/edge/server timing model in manuscript **Section 3.3** — code and manuscript are a single source of truth; if one changes, update the other.

## 4. Testing Rigor

Any mathematical or simulation configuration updates must be accompanied by comprehensive tests verifying correctness and determinism, particularly for:

- Stochastic rounding (DAdaQuant); symmetric uniform quantization (FedPAQ and FedMAQ, which share bit-width semantics)
- Formulation output values
- Decoupled telemetry calculations
- Deterministic data partitioning (Dirichlet skew and FEMNIST writer chunking)
