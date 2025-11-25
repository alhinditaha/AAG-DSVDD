````markdown
# Anomaly-Aware Graph-Based Semi-Supervised Deep SVDD (AAG-DSVDD)

This repository implements **Anomaly-Aware Graph-Based Semi-Supervised Deep SVDD (AAG-DSVDD)**, a DeepSVDD-style one-class method for semi-supervised anomaly detection with:

- A **bias-free MLP encoder** (no biases, no batch-norm) to reduce center collapse.
- A **soft-boundary hinge** on labeled normals (`y = +1`), keeping them inside a hypersphere of radius `R`.
- A **push-out hinge** for a single labeled anomaly class (`y = −1`) with margin `m`, scaled by `Ω`, pushing anomalies outside the hypersphere.
- A **label-aware graph Laplacian regularizer** on squared distances \(d^2(x) = \lVert z - c\rVert^2\), built from a kNN graph in the latent space:
  - Edge policy: labeled nodes connect only to unlabeled nodes; unlabeled nodes connect to both labeled and unlabeled.
- A **DeepSAD-style unlabeled pull-in term** on unlabeled points (`y = 0`), with optional capping for robustness.

With \(z = \phi_\theta(x)\) the embedding, \(c\) the center, and \(d^2(x) = \lVert z - c\rVert_2^2\), the anomaly score is

\[
f(x) = d^2(x) - R^2,
\]

so larger scores correspond to more anomalous samples.

> To approximate vanilla soft-boundary DeepSVDD (without graph regularization and anomaly push-out), set `Omega = 0`, `lambda_u = 0`, and ensure only labeled normals are used.

**Short name:** AAG-SS-DeepSVDD (or AAG-DSVDD).

> **Status:** Research code for a manuscript that is currently in preparation / under review. The API and configuration may change. Once the paper is accepted, this repository will be updated with a stable release and the final citation.

---

## Repository structure

Top level:

- `pyproject.toml` — build metadata (setuptools; installable via `pip install -e .`).
- `LICENSE` — MIT license.
- `README.md` — this file.
- `.gitignore` — standard Python / editor ignores.
- `scripts/` — small entry-point scripts.
- `src/` — core implementation, baselines, and experiment runners.
- `tests/` — initial smoke-test scaffolding for the method.

Core source layout (under `src/`):

```text
src/
├─ models/
│   ├─ aag_dsvdd.py          # AAGDeepSVDDTrainer + TrainConfig + bias-free MLP encoder
│   ├─ deep_svdd.py          # DeepSVDD baseline (center-based encoder)
│   └─ svdd_linear.py        # Linear SVDD variant
├─ baselines/
│   ├─ deep_sad.py           # DeepSAD baseline
│   ├─ deep_svdd.py          # DeepSVDD baseline wrapper
│   ├─ ocsvm.py              # OC-SVM baseline
│   └─ svdd_rbf.py           # SVDD with RBF kernel baseline
├─ data/
│   └─ sim_banana_moons.py   # Synthetic banana / two-moons generators + semi-supervised labeling
├─ datasets/
│   ├─ banana.py             # Banana distribution utilities
│   └─ data.py               # synthetic_blobs, split helpers, CSV loading (with pandas)
├─ optim/
│   ├─ deepSVDD_trainer.py   # Training loop for DeepSVDD-style models
│   └─ svdd_linear_trainer.py
├─ plots/                    # Placeholder for plotting utilities
├─ run_methods.py            # CLI runner for AAG-DSVDD + baselines on 2D banana / moons
└─ run_suite.py              # Orchestrator around run_methods.py; configured to reproduce
                             # the simulated banana experiment results used in the manuscript
````

Helper scripts:

```text
scripts/
├─ run_banana.py             # Simple banana-dataset example (to be kept in sync with src/)
├─ run_demo.py               # Minimal synthetic demo
└─ train_from_csv.py         # Train on a labeled CSV with triplet labels {+1,0,−1}
```

Tests:

```text
tests/
└─ test_smoke.py             # Initial smoke-test scaffolding (may still mirror older naming)
```

---

## Method overview

### Labels and embeddings

Training assumes **triplet labels**:

* `y = +1` — labeled normal samples,
* `y = 0`  — unlabeled samples (mixture of normals and anomalies),
* `y = −1` — labeled anomaly samples (single anomaly class).

A bias-free MLP encoder (\phi_\theta) maps inputs to latent space:

[
z_i = \phi_\theta(x_i) \in \mathbb{R}^d.
]

A center (c \in \mathbb{R}^d) and radius (R) parameterize a hypersphere in the latent space, and the squared distance

[
d_i^2 = \lVert z_i - c\rVert_2^2
]

drives both the loss and the anomaly score.

### Loss components (high-level)

The training objective combines:

1. **Radius / soft-boundary term**

   * Encourages a compact hypersphere, similar to soft-boundary DeepSVDD.
   * Controlled by `nu`: approximate fraction of normals allowed to lie outside the sphere.

2. **Soft-boundary hinge for labeled normals (`y = +1`)**

   * Penalizes labeled normals that fall outside the hypersphere:
     [
     \max(0, d_i^2 - R^2)^p
     ]
   * Typically with (p = 2) (squared hinge).

3. **Push-out hinge for labeled anomalies (`y = −1`)**

   * Encourages anomalies to lie outside the sphere with a margin `m`:
     [
     \max(0, R^2 + m - d_i^2)^p
     ]
   * Scaled by `Omega (Ω)`, controlling the importance of anomaly separation.

4. **Graph Laplacian regularizer on scores**

   * Build a latent kNN graph over embeddings, with a **label-aware edge policy**:

     * Labeled nodes connect only to unlabeled nodes.
     * Unlabeled nodes connect to both labeled and unlabeled nodes.
   * Penalize differences in anomaly scores between connected nodes:
     [
     \sum_{(i,j)\in E} w_{ij},(f_i - f_j)^2
     ]
   * Edges touching labeled anomalies can be upweighted via `gamma_anom_edges ≥ 1`.
   * Overall strength set by `lambda_u` in `TrainConfig`.

5. **DeepSAD-style unlabeled pull-in**

   * Averages squared distances for unlabeled points and pulls them towards the center:
     [
     \eta_{\text{unl}} \cdot \mathbb{E}_{y_i=0}[d_i^2]
     ]
   * Controlled by `eta_unl` and optionally capped (`cap_unlabeled`, `cap_offset`) to maintain robustness under contamination.

The radius (R^2) is updated using a DeepSVDD-style quantile rule on the distances of labeled normals, with update frequency controlled by `r2_update_every` and `r2_start_epoch`.

### Key hyperparameters (TrainConfig)

Some of the main fields in `TrainConfig` (see `models/aag_dsvdd.py`):

* Encoder:

  * `in_dim`: input dimensionality.
  * `hidden`: hidden layer sizes for the MLP (tuple, e.g., `(128, 64)`).
  * `out_dim`: latent dimensionality.

* Hinge / boundary:

  * `p`: hinge power (usually `2`).
  * `nu`: soft-boundary trade-off parameter.
  * `Omega`: anomaly push-out weight.
  * `margin_m`: anomaly margin `m`.

* Graph:

  * `lambda_u`: weight for the graph Laplacian regularizer.
  * `k`: number of neighbors in the kNN graph.
  * `gamma_anom_edges`: upweight edges involving anomalies.
  * `graph_refresh`: epochs between graph rebuilds.

* Unlabeled pull-in:

  * `eta_unl`: weight for unlabeled mean distance.
  * `cap_unlabeled`: if `True`, cap unlabeled distances by `R^2 + cap_offset`.
  * `cap_offset`: cap margin added to `R^2`.

* Radius update:

  * `r2_update_every`: update (R^2) every N epochs.
  * `r2_start_epoch`: start updating (R^2) from this epoch.

* Optimization / system:

  * `epochs`, `batch_size`, `lr`, `wd`, `warmup_epochs`.
  * `device`: `"cpu"` or `"cuda"` (if available).
  * `print_every`: logging frequency in epochs.

---

## Installation

It is recommended to use a virtual environment (Python ≥ 3.9):

```bash
python -m venv .venv
source .venv/bin/activate      # on Windows: .venv\Scripts\activate
pip install -e .
```

The core dependencies are managed via `pyproject.toml` and include:

* `torch`
* `numpy`
* `scikit-learn`
* `scipy`
* `tqdm`

For plotting and CSV-based workflows, you may also need:

* `matplotlib`
* `pandas`
* `pytest` (for tests)

You can optionally create a `requirements.txt` such as:

```text
torch>=2.1
numpy>=1.24
scikit-learn>=1.3
scipy>=1.10
tqdm>=4.66
matplotlib>=3.7
pandas>=2.0
pytest>=7.0
```

and run:

```bash
pip install -r requirements.txt
```

---

## Quick start: Python API

A minimal end-to-end example (2D synthetic data, triplet labels):

```python
import torch
from models.aag_dsvdd import AAGDeepSVDDTrainer, TrainConfig

# Example: synthetic data (replace with your own)
N = 500
X = torch.randn(N, 2)
y = torch.randint(low=-1, high=2, size=(N,))  # values in {-1, 0, +1}

cfg = TrainConfig(
    in_dim=2,
    hidden=(128, 64),
    out_dim=16,
    epochs=20,
    batch_size=128,
    lr=1e-3,
    wd=1e-4,
    nu=0.1,
    Omega=2.0,
    margin_m=1.0,
    lambda_u=0.1,
    k=15,
    graph_refresh=2,
    eta_unl=1.0,
    device="cpu",
)

trainer = AAGDeepSVDDTrainer(cfg)
trainer.fit(X, y)
scores = trainer.score(X)  # shape: [N], higher = more anomalous
```

You can then threshold `scores` to produce binary anomaly predictions, or feed them into evaluation / calibration pipelines.

---

## Quick start: scripts

> Note: the helper scripts under `scripts/` are provided as starting points and may still reflect earlier import paths. They are useful as templates for constructing your own experiment scripts.

### 1. Minimal synthetic demo

From the repository root:

```bash
python scripts/run_demo.py
```

Typical behavior:

* Generates small synthetic blobs with labeled normals, anomalies, and unlabeled samples.
* Trains an AAG-DSVDD-style model with default hyperparameters.
* Prints basic information and example scores.

### 2. Banana example

```bash
python scripts/run_banana.py
```

Typical behavior:

* Uses a 2D banana-shaped dataset plus anomaly clusters (via `data/sim_banana_moons.py`).
* Converts raw labels into triplet form `{+1, 0, −1}` with a chosen unlabeled fraction.
* Trains AAG-DSVDD and prints summary information.

### 3. Train from CSV

```bash
python scripts/train_from_csv.py \
    --csv path/to/data.csv \
    --label-col label
```

Assumptions:

* Your CSV has a label column (default `label`) with integer values in `{+1, 0, −1}`:

  * `+1` — labeled normal,
  * `0`  — unlabeled,
  * `−1` — labeled anomaly.
* All other columns are numeric features.

Internally, this uses `datasets.data.load_csv_dataset`, which relies on `pandas` to read the CSV.

---

## Reproducing simulated banana results

The file `src/run_suite.py` is an **orchestrator** around `src/run_methods.py`. It is configured to reproduce the **simulated banana dataset experiments** used in the AAG-DSVDD manuscript.

`run_suite.py` controls:

* Dataset choice (`banana` / `moons`) and generator settings (sample counts, shapes, noise).
* Semi-supervised labeling policy:

  * Numbers / fractions of labeled normals, labeled anomalies, and unlabeled points.
* Which methods to run:

  * **AAG-DSVDD (proposed)**
  * **DeepSVDD**
  * **DeepSAD**
  * **OC-SVM**
  * **SVDD-RBF**
* Hyperparameters for each model.
* Seeds, plotting flags, and CSV logging of metrics.
* Grid sweeps and multi-seed evaluations.

Example usage (default configuration, banana only):

```bash
cd src
python run_suite.py
```

By default:

* `DATASETS = ["banana"]`.
* `SEEDS` is a small list (e.g., `[0]` by default).
* Sample counts and hyperparameters are aligned with the synthetic banana experiments used in the manuscript.

You can edit `run_suite.py` to:

* Switch to `["moons"]` or include both datasets.
* Add additional seeds.
* Enable / disable specific baselines.
* Expand hyperparameter lists and set `GRID_EXPLODE = True` for full grid sweeps.

---

## Testing

An initial smoke-test scaffold is provided in `tests/test_smoke.py`. Once the imports are updated to the new package layout, you will be able to run:

```bash
pytest
```

to verify that a small synthetic experiment runs end-to-end and returns anomaly scores of the expected shape.

Because this is research code under active development, the test file may temporarily lag behind the latest internal refactors.

---

---

## License

This project is released under the **MIT License**. See `LICENSE` for the full text.

```

```
