# Blown Film MPC

> **Data-driven System Identification and Model Predictive Control
> for a Co-Extrusion Blown Film Manufacturing Line**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/Code%20Style-PEP%208-orange)](https://peps.python.org/pep-0008/)
[![CVXPY](https://img.shields.io/badge/Solver-CVXPY%20%2F%20OSQP-purple)](https://www.cvxpy.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?logo=numpy)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.11%2B-8CAAE6?logo=scipy)](https://scipy.org/)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [System Description](#-system-description)
- [Mathematical Background](#-mathematical-background)
- [Project Structure](#-project-structure)
- [Module Descriptions](#-module-descriptions)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Pipeline Stages](#-pipeline-stages)
- [Outputs](#-outputs)
- [Dataset Format](#-dataset-format)
- [OOP Design Principles](#-oop-design-principles)
- [Dependencies](#-dependencies)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔍 Overview

This repository implements a complete **data-driven control engineering
pipeline** for an industrial co-extrusion blown film manufacturing line.
Starting from raw SCADA/PLC time-series data, the pipeline:

1. **Identifies** a discrete-time linear state space model from
   input-output data using the **N4SID subspace algorithm**.
2. **Reduces** the model order via **Balanced Truncation** and
   **POD/Galerkin projection** to make it suitable for real-time
   optimisation.
3. **Designs** a **steady-state Kalman filter** for augmented state
   estimation with offset-free tracking.
4. **Validates** the identified model against held-out test data
   with quantitative metrics (R², NRMSE, MSE).
5. **Designs and optimises** a **Model Predictive Controller (MPC)**
   that enforces input/output constraints and minimises a
   multi-objective tracking cost.
6. **Simulates** the closed-loop system and generates comprehensive
   diagnostic reports and figures.

The entire codebase is written in **object-oriented Python** following
PEP 8, SOLID principles, and modern type-annotation standards.

---

## 🏭 System Description

A **co-extrusion blown film line** produces multi-layer plastic film
by simultaneously extruding several polymer melts through a circular
die, inflating the resulting tube into a bubble, cooling it, and
winding the collapsed film onto rolls.

### Monitored Stations

| Station | Description |
|---------|-------------|
| `ST0`   | Production management (job tracking) |
| `ST110` | Extruders, die head, IBC cooling, dosing, blowers |
| `ST112` | Haul-off tower |
| `ST113` | Winder 1 |
| `ST114` | Winder 2 |

### Key Process Variables

| Category | Variables |
|----------|-----------|
| **Controlled outputs** | Layer thickness (`SDickeIst`), melt temperature (`Massetemperatur`), melt pressure (`druck_IstP`), IBC speed, cooling device temperature, haul-off speed, roll diameter, web tension |
| **Manipulated inputs** | Heating zone setpoints (`SollTemp`), thickness setpoints (`SDickeSoll`), IBC speed setpoints, cooling setpoints, haul-off speed setpoint, winder speed/tension setpoints |
| **Disturbances** | Material bulk density, dosing proportions, blower load |

---

## 📐 Mathematical Background

### State Space Model

The linearised discrete-time model takes the form:

```
x_{k+1} = A x_k + B u_k + w_k
y_k     = C x_k + D u_k + v_k
```

where `w_k ~ N(0, Q_n)` and `v_k ~ N(0, R_n)` are process and
measurement noise respectively.

### Model Order Reduction

The N4SID identification step already produces a **discrete-time** model
(sampled directly at `Ts`), so the reduction pipeline stays entirely in
the discrete-time domain — no re-discretisation step is needed:

```
Full identified model (n = n_id, e.g. 20)
        │
        ▼  Balanced Truncation (BT, HSV-based truncation)
n ≈ n_red (e.g. 12)
        │
        ▼  POD / Galerkin Projection (snapshot-energy-based, may reduce further)
n ≤ n_red
        │
        ▼  Integrator Augmentation (one integrator per measured output)
n_aug = n + n_y
```

For example, with the default `--n_id 20 --n_red 12` on the 22-output
dataset, the augmented MPC model has `n_aug ≈ 12 + 22 = 34` states.

### MPC Optimisation Problem

At each time step `k`, the controller solves:

```
min   Σ_{j=0}^{Np-1} ||y_{k+j} - y_ref||²_Q + ||Δu_{k+j}||²_R

s.t.  x_{j+1} = A_aug x_j + B_aug u_j
      u_min  ≤ u_j  ≤ u_max
      Δu_min ≤ Δu_j ≤ Δu_max
      y_min  ≤ y_j  ≤ y_max        (soft, via slack variables)
```

The QP is solved using **OSQP** via **CVXPY**.

---

## 📁 Project Structure

```
blown_film_mpc/
├── extrusion.csv               # Example real dataset (optional, not required)
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
├── README.md                   # This file
│
└── Code/
    ├── main.py                     # Entry point & pipeline orchestrator
    ├── config.py                   # All configuration, constants & column defs
    ├── data_manager.py              # Data loading, preprocessing & splitting
    ├── system_identification.py     # N4SID subspace identification
    ├── model_reduction.py           # Balanced truncation, POD, integrator augmentation
    ├── estimation.py                # Kalman filter & state estimation
    ├── mpc_controller.py            # MPC design, QP solver & weight optimisation
    ├── simulation.py                # Closed-loop simulator & model validation
    ├── utils.py                     # Plotting, reporting & helper utilities
    │
    └── outputs/                     # Auto-generated figures and reports
        ├── singular_values.png
        ├── hsv_balanced_truncation.png
        ├── validation_predictions.png
        ├── residuals.png
        ├── metrics_heatmap.png
        ├── mpc_weight_optimisation.png
        ├── mpc_output_tracking.png
        ├── mpc_inputs.png
        ├── mpc_cost.png
        └── blown_film_sysid_mpc_report.txt
```

---

## 📦 Module Descriptions

### `config.py`
Central configuration hub. Contains:
- **`ALL_COLUMNS`** — complete list of all SCADA dataset column names.
- **`INPUT_COLS`** — manipulated variable column names.
- **`OUTPUT_COLS`** — controlled variable column names.
- **Typed dataclasses** — `DataConfig`, `IdentificationConfig`,
  `ReductionConfig`, `KalmanConfig`, `MPCConfig`, `SimulationConfig`,
  `ProjectConfig` — all frozen and documented.

> ⚙️ **All tunable parameters live here. No magic numbers elsewhere.**

---

### `data_manager.py`
Handles the full data lifecycle.

| Class | Responsibility |
|-------|---------------|
| `IODataset` | Immutable container for scaled train/test I/O arrays |
| `SyntheticDataGenerator` | Generates stable LTI-driven test data |
| `DataManager` | Load → validate → clean → scale → split |

**Preprocessing steps:**
1. Sort by timestamp
2. Drop fully-NaN columns
3. Forward-fill / backward-fill short gaps (configurable limit)
4. Drop remaining NaN rows
5. Z-score outlier removal (configurable threshold)
6. `RobustScaler` normalisation
7. Chronological 70/30 train-test split

---

### `system_identification.py`
Implements data-driven model identification.

| Class | Responsibility |
|-------|---------------|
| `StateSpaceModel` | Immutable (A, B, C, D) container with simulation & stability utilities |
| `SubspaceIdentifier` | N4SID algorithm: Hankel matrices → SVD → observability → (A, B, C, D) |
| `ParameterOptimiser` | L-BFGS-B refinement of (A, B, C, D) with stability penalty |

---

### `model_reduction.py`
Three-stage reduction pipeline.

| Class | Responsibility |
|-------|---------------|
| `ReducedModel` | Container for reduced + augmented matrices |
| `ModelReducer` | Balanced truncation → POD/Galerkin → integrator augmentation |

> The identified model is already discrete-time, so no ZOH
> re-discretisation step is performed (or needed) here.

**Guaranteed error bound (balanced truncation):**
‖G(z) - G_r(z)‖∞ ≤ 2 · Σ{i > r} σ_i

---

### `estimation.py`
Kalman filter for real-time state estimation.

| Class | Responsibility |
|-------|---------------|
| `KalmanFilter` | Steady-state discrete Kalman filter on augmented state z = [x_r; d] |

Features:
- Steady-state gain via Lyapunov equation
- Time-varying update for robustness
- `clone()` method for safe use in simulation loops
- `batch_filter()` for offline processing

---

### `mpc_controller.py`
MPC design and real-time QP solving.

| Class | Responsibility |
|-------|---------------|
| `PredictionMatrices` | Immutable (F, G) prediction matrices |
| `MPCResult` | Immutable result of a single QP solve |
| `MPCController` | Builds and solves the MPC QP at each step |
| `MPCWeightOptimiser` | Nelder-Mead tuning of Q, R in log-space |

**Priority weighting (default):**

```
q_thickness   = 10.0  (layer thickness — highest priority)
q_temperature =  5.0  (melt temperature)
q_default     =  1.0  (all other outputs)
```

---

### `simulation.py`
Closed-loop simulation and open-loop validation.

| Class | Responsibility |
|-------|---------------|
| `ValidationMetrics` | Per-output MSE, R², NRMSE container |
| `SimulationResult` | Full closed-loop result with ISE, ITAE properties |
| `ModelValidator` | Open-loop test-set validation with metrics |
| `ClosedLoopSimulator` | MPC + Kalman filter + plant in closed loop |

---

### `utils.py`
Visualisation and reporting.

| Class | Responsibility |
|-------|---------------|
| `Plotter` | All matplotlib/seaborn figures, auto-saved to `outputs/` |
| `ReportWriter` | Structured plain-text report builder |

---

### `main.py`
Top-level pipeline orchestrator.

| Class | Responsibility |
|-------|---------------|
| `BlownFilmPipeline` | Runs all 8 stages in order, wires modules together |

CLI entry point with `argparse` — see [Usage](#-usage).

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-org/blown-film-mpc.git
cd blown-film-mpc
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Requires **Python 3.10+**. See [Dependencies](#-dependencies) for details.

---

## 🚀 Quick Start

Run the full pipeline with synthetic data (no CSV required) straight
from the `Code/` directory:

```bash
cd Code
python main.py
```

This generates all diagnostic figures and a text report under
`Code/outputs/`.

To run against your own dataset:

```bash
python main.py --data ../extrusion.csv
```

---

## 📖 Usage

`main.py` is the single CLI entry point. All behaviour is controlled
via command-line flags — there is no separate flag for synthetic data;
it is used automatically whenever `--data` is omitted.

```bash
# Synthetic data, all defaults
python main.py

# Real data
python main.py --data ../extrusion.csv

# Custom model orders and MPC horizons
python main.py --data ../extrusion.csv --n_id 20 --n_red 12 --Np 20 --Nc 8

# Skip MPC weight optimisation (much faster)
python main.py --no_weight_opt

# Skip N4SID parameter refinement
python main.py --no_param_opt

# Headless run (no interactive figure windows), custom output folder
python main.py --no_show --output_dir results
```

### CLI Arguments

| Flag | Default | Description |
|------|---------|--------------|
| `--data` | `None` | Path to CSV/Excel data file (omit for synthetic data) |
| `--output_dir` | `outputs` | Directory for figures and report |
| `--n_id` | `20` | N4SID model order |
| `--n_red` | `12` | Target order after balanced truncation |
| `--Ts` | `3.0` | Sampling time in seconds |
| `--Np` | `20` | MPC prediction horizon |
| `--Nc` | `8` | MPC control horizon |
| `--T_sim` | `300` | Closed-loop simulation steps |
| `--no_param_opt` | `False` | Skip parameter optimisation after N4SID |
| `--no_weight_opt` | `False` | Skip MPC weight optimisation |
| `--no_show` | `False` | Do not display figures interactively |

---

## ⚙️ Configuration

All tunable parameters are centralised in [`Code/config.py`](Code/config.py)
as frozen dataclasses, so nothing else in the codebase hardcodes a
magic number. The CLI only exposes the parameters most commonly
changed between runs; everything else (noise scales, constraint
bounds, priority weights, etc.) is edited directly in `config.py`.

| Dataclass | Key fields |
|-----------|-----------|
| `DataConfig` | `sampling_time`, `train_fraction`, `outlier_zscore`, `synthetic_samples`, `random_seed` |
| `IdentificationConfig` | `n_states`, `n_block_rows`, `optimise_params`, `optimisation_method` |
| `ReductionConfig` | `n_states_bt`, `bt_energy_tolerance`, `pod_energy_tolerance` |
| `KalmanConfig` | `process_noise_scale`, `measurement_noise_scale` |
| `MPCConfig` | `prediction_horizon`, `control_horizon`, `u_bound`, `du_bound`, `y_bound`, `q_thickness_weight`, `q_temperature_weight`, `r_weight`, `optimise_weights`, `weight_opt_iterations` |
| `SimulationConfig` | `n_steps`, `noise_std`, `ref_step_time_1/2/3`, `ref_amplitude_1/2` |

`ProjectConfig` bundles all of the above and is what `main.py` builds
from the parsed CLI arguments before constructing `BlownFilmPipeline`.

---

## 🔄 Pipeline Stages

`BlownFilmPipeline.run()` executes eight stages in order:

1. **Data** — load real data or generate synthetic data, clean, scale, split
2. **System Identification** — N4SID subspace identification (+ optional L-BFGS-B refinement)
3. **Model Order Reduction** — balanced truncation → POD/Galerkin → integrator augmentation
4. **Kalman Filter** — steady-state gain design on the augmented state
5. **Model Validation** — open-loop test-set metrics (R², NRMSE, MSE) per output
6. **MPC Design** — builds the QP and, if enabled, tunes Q/R weights via Nelder-Mead
7. **Closed-Loop Simulation** — MPC + Kalman filter + plant simulated together against a step-change reference
8. **Report** — prints and saves a structured text report

---

## 📊 Outputs

Every run writes to `output_dir` (default `Code/outputs/`):

| File | Description |
|------|-------------|
| `singular_values.png` | N4SID Hankel singular value spectrum |
| `hsv_balanced_truncation.png` | Balanced truncation HSV energy plot |
| `validation_predictions.png` | Predicted vs. actual test-set outputs |
| `residuals.png` | Validation residuals per output |
| `metrics_heatmap.png` | R² / NRMSE / MSE heatmap across outputs |
| `mpc_weight_optimisation.png` | Nelder-Mead cost history (if weight optimisation enabled) |
| `mpc_output_tracking.png` | Closed-loop output tracking vs. reference |
| `mpc_inputs.png` | Closed-loop manipulated variable trajectories |
| `mpc_cost.png` | QP cost per simulation step |
| `blown_film_sysid_mpc_report.txt` | Full plain-text summary of all 8 stages |

---

## 🗂️ Dataset Format

Real data is supplied as a CSV/Excel file with one row per sample and
one column per SCADA tag, matching the names in `ALL_COLUMNS`
(`Code/config.py`):

- A time column (`Datum`) used for chronological sorting.
- 26 manipulated-variable columns (`INPUT_COLS`).
- 22 controlled-variable columns (`OUTPUT_COLS`).

`DataManager` handles missing tags/columns gracefully during cleaning
(dropping fully-NaN columns, gap-filling, outlier removal), but the
column **names** must match what `config.py` expects for the
corresponding signal to be picked up. If no `--data` path is given,
`SyntheticDataGenerator` produces a stable random LTI-driven dataset
with the same input/output dimensions so the full pipeline can be
exercised without real plant data.

---

## 🧱 OOP Design Principles

- **Single Responsibility** — each module owns one pipeline concern
  (data, identification, reduction, estimation, control, simulation,
  reporting).
- **Immutability** — data containers (`StateSpaceModel`, `ReducedModel`,
  `IODataset`, `MPCResult`, `ValidationMetrics`) are frozen/immutable
  wherever practical to avoid accidental in-place mutation of shared
  state.
- **Composition over inheritance** — `BlownFilmPipeline` composes
  independent, testable classes rather than building a deep class
  hierarchy.
- **No magic numbers** — every tunable constant lives in `config.py`.
- **Type hints throughout** — all public methods are annotated for
  static-analysis friendliness.

---

## 📦 Dependencies

See [`requirements.txt`](requirements.txt):

- `numpy`, `scipy` — numerical linear algebra, Lyapunov/Riccati solvers
- `pandas` — data loading and preprocessing
- `matplotlib`, `seaborn` — figure generation
- `scikit-learn` — validation metrics (R², MSE), `RobustScaler`
- `cvxpy` — MPC QP modelling (bundles the OSQP solver used by default)
- `tqdm` — progress bars
- `openpyxl` — Excel file support

---

## 🩺 Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| `UnicodeEncodeError` when running on Windows | Fixed in `main.py` by forcing UTF-8 stdout/stderr; ensure you're running the current version of the script. |
| Validation metrics are `NaN` | Usually an unstable reduced model. Confirm `ReducedModel.A_d` has spectral radius < 1 (logged at Stage 3); try a smaller `--n_red`. |
| Poor validation R² with real data | Increase `--n_id`/`--n_red`, provide more training samples, or check that CSV column names match `config.py`. |
| Slow runs with weight optimisation | Use `--no_weight_opt`, or reduce `--T_sim`, `--Np`, `--Nc` for faster smoke tests. |
| `Files above 50MB cannot be...` type errors when inspecting data | Large CSVs should be read by `pandas`/the pipeline itself, not by generic text-file tools. |

---

## 🤝 Contributing

Contributions are welcome. Please:

1. Follow the existing OOP structure and PEP 8 style.
2. Keep configuration values in `config.py` — avoid hardcoding constants.
3. Add/update tests or a smoke-test run (`python main.py --no_show --T_sim 5 --n_id 8 --n_red 5 --no_weight_opt`) before submitting a PR.
4. Open an issue describing the change before large refactors.

---

## 📄 License

Distributed under the **MIT License**. See [LICENSE](LICENSE) for details.