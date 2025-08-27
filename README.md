# MUGEN — MUltimodal GENeralization Tool

*A personal study project on multimodal regression & generalization.*

MUGEN explores how different ML approaches learn from **heterogeneous inputs** (e.g., synchronized timestamps + vector features) to predict a **single scalar target**. It contains a lightweight training harness, a small model zoo (from linear baselines to CNNs/Transformers/PointNet‑style models), utilities for cross‑validation, and a **dummy dataset generator** so you can run everything end‑to‑end without any proprietary data.

> **Why?** The goal is to compare modeling strategies and test **generalization** under different dataset splits and feature representations, not to ship a single production model.

---

## Highlights

* **End‑to‑end pipeline**: data loading → train/val/test folding → model training → evaluation → plots.
* **Model zoo**: linear regressors on time signals, MLPs/CNNs on vector inputs, autoencoders (pretrain & freeze encoder → regression head), Transformers, and PointNet‑like architectures.
* **Training harness**: early stopping, stuck‑training retries, consistent output structure, and ECDF plots for error distributions.
* **Cross‑validation utilities**: simple K‑folds, leave‑group/file‑out style splits, plus helpers for packing/unpacking fold data.
* **Runs on CPU**: GPU is optional; threading is configured to behave well on CPUs.
* **Dummy data**: generate a synthetic dataset with the same schema and try everything locally.

---

## Repository Structure

```
project-root/
│
├─ lib/
│  ├─ data/              # Data loaders, caches, and dummy-data maker
│  ├─ dsp/               # Classical baselines / spectral-style helpers
│  ├─ pipelines/         # Unpackers (X/y builders) & Constructors (result packers)
│  ├─ training/          # Training harnesses (BaseModelTester, FineTune, etc.)
│  └─ utils/             # Plotting & small utilities
│
├─ models/               # Model zoo (MLP/CNN/Transformer/PointNet/Autoencoders)
├─ scripts/              # One-off scripts (e.g., build_cache / dummy-data)
├─ tests/                # Minimal end-to-end examples
├─ data/                 # (created locally) input caches / dummy data
└─ output/               # (created automatically) metrics & plots per run
```

> Names and exact contents may evolve; check the modules for the most current API.

---

## Installation

Tested with **Python 3.10+**.

```bash
# 1) Clone
git clone https://github.com/MirumeYato/mugen.git
cd mugen

# 2) (Optional) Create a fresh environment
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# 3) Install dependencies
pip install -U pip wheel
pip install numpy pandas scipy scikit-learn matplotlib seaborn tqdm statsmodels joblib tensorflow
```

> If you prefer CPU‑only installs, the standard `tensorflow` package is fine. GPU is optional.

---

## Data Schema

MUGEN expects a **tabular** dataset (e.g., a pandas DataFrame serialized to `.pkl`) where each row is **one sample** with:

* `features`: nested list/array of **one or more fixed‑length float vectors**. Common shapes are **(2, 256)** or **(4, 256)**.
* `target`: float — the scalar label to predict.
* `slope_est` *(optional)*: float baseline estimate from a trend‑based heuristic.
* `music_est` *(optional)*: float baseline estimate from a subspace/peak‑finding heuristic.
* `rtt` *(optional)*: auxiliary reference value used by some evaluation flows.
* `time_stamp_1..4`: int64 — synchronized timestamps or markers.
* `source_file`: string — an identifier used by some split strategies.

You can create a fully compatible **dummy dataset** in one command (see Quickstart).

---

## Quickstart (Dummy Data)

1. **Generate dummy data** (saved under `data/dummy_data.pkl`):

```bash
python scripts/build_cache.py
```

2. **Run a minimal end‑to‑end example** (trains a small model and writes plots/metrics to `output/`):

```bash
python tests/anaSimple1.py
```

This demonstrates a simple fold, training, predictions, and ECDF plots of the error distribution.

---

## Training & Evaluation

The high‑level flow is:

1. **Choose a fold strategy** (e.g., simple K‑fold or leave‑file‑out) and build index splits.
2. **Unpack** the DataFrame into tensors via *Unpackers* (builds `(train_ds, val_ds, test_ds, X_test, y_test, ...)`).
3. **Train** a model via `BaseModelTester` (with early stopping & retry‑on‑stuck).
4. **Construct** results via *Constructors* (assemble per‑fold DataFrames and compute metrics like MAE, ECDF, etc.).

Key pieces you’ll see in code:

* `lib/training/BaseModelTester.py` — the main training loop wrapper.
* `lib/pipelines/Unpackers.py` — turn a split of rows into model‑ready data.
* `lib/pipelines/Constructors.py` — convert predictions + references into tidy results tables & errors.
* `lib/utils/plotting.py` — e.g., `plot_learning_curves(...)` and ECDF helpers.

### Example: swap in a different model

Most examples take a **model‑builder function**. To try another architecture, import a builder from `models/` and plug it in. A few that exist:

* **Linear/Time baselines** (timestamps only)
* **MLP / 1D‑CNN** (vector inputs)
* **Autoencoder + regression head** (pretrain encoder, freeze, then fine‑tune)
* **Transformer‑style** encoders (positional encoding, attention blocks)
* **PointNet‑like** models for per‑position features

> Many builders live in `models/model_configurations.py`, plus auxiliary modules like `hybrid_model.py`, `autoencoder.py`, or `Insane_model.py`. Some are experimental and may need small refactors.

---

## Finetuning Encoders

There’s a convenience module for pretraining an **autoencoder encoder** and then reusing it for scalar regression:

* `lib/training/FineTune.py`: helper routines for encoder pretraining & transfer to downstream regression.

Typical recipe:

1. Train an autoencoder on your feature vectors.
2. Freeze the encoder and attach a small MLP/CNN head for the scalar target.
3. Train with a cosine‑decay learning rate schedule + early stopping.

---

## Outputs

Each run creates a unique subfolder under `output/`, e.g. `output/MT_<run_name>_<epochs>/`, containing:

* **Learning curves** (`loss`, `val_loss`, etc.).
* **Aggregated ECDF plots** of MAE across folds.
* **Per‑fold CSVs / DataFrames** summarizing predictions, targets, and any baselines provided.
* *(Optionally)* **saved models** in Keras format (`.keras`).

Note: Units in figures are generic — they reflect whatever units your `target` uses.

---

## Tips & Notes

* **CPU friendly**: the harness disables GPU by default and tunes thread counts; remove those lines if you prefer GPU.
* **Baselines optional**: `slope_est` / `music_est` / `rtt` aren’t required; if absent, related comparisons are skipped.
* **Feature shapes**: Unpackers can select which branches of `features` to use (e.g., the first two vectors from a 4×256 blob).
* **Reproducibility**: set seeds at your script entrypoint when needed; stochasticity is intentional in some demos.

---

## Roadmap

* [ ] Unify feature‑shape handling across Unpackers (2×256 vs 4×256).
* [ ] Clean up the experimental models; promote stable ones to a documented API.
* [ ] Add a small CLI (YAML/JSON configs) for repeatable experiments.
* [ ] More fold strategies (group‑aware, temporal splits, stratified by source).
* [ ] Richer visualizations (residual plots, per‑source breakdowns, calibration curves).

---

## Contributing / Using This Repo

This is an evolving research playground. Feel free to open issues with questions or small PRs. If you fork it for your own experiments, please keep the **data schema** compatible so the Unpackers/Constructors continue to work.

---

## Acknowledgements

This repository started as a personal exploration of multimodal scalar regression and generalization. Any similarity to other domains is coincidental; no external datasets are included.
