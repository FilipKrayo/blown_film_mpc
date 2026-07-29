# 🎬 Blown Film MPC

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
x_{k+1} = A x_k + B u_k + w_k y_k = C x_k + D u_k + v_k

where `w_k ~ N(0, Q_n)` and `v_k ~ N(0, R_n)` are process and
measurement noise respectively.

### Model Order Reduction

The reduction pipeline proceeds as:
Full model (n=146) → Singular Perturbation → n ≈ 90 (fast states eliminated) → Balanced Truncation (BT) → n ≈ 40 (HSV-based truncation) → POD / Galerkin Projection → n ≈ 22 (snapshot-based) → ZOH Discretisation → n ≈ 22 (Ts = 3 s) → Integrator Augmentation → n ≈ 80 (offset-free MPC)

### MPC Optimisation Problem

At each time step `k`, the controller solves:
min Σ_{j=0}^{Np-1} ||y_{k+j} - y_ref||²_Q + ||Δu_{k+j}||²_R s.t. x_{j+1} = A_aug x_j + B_aug u_j u_min ≤ u_j ≤ u_max Δu_min ≤ Δu_j ≤ Δu_max y_min ≤ y_j ≤ y_max (soft, via slack variables)

The QP is solved using **OSQP** (with ECOS fallback) via **CVXPY**.

---

## 📁 Project Structure
blown_film_mpc/ │ ├── main.py # Entry point & pipeline orchestrator ├── config.py # All configuration, constants & column defs ├── data_manager.py # Data loading, preprocessing & splitting ├── system_identification.py # N4SID subspace identification ├── model_reduction.py # Balanced truncation, POD, discretisation ├── estimation.py # Kalman filter & state estimation ├── mpc_controller.py # MPC design, QP solver & weight optimisation ├── simulation.py # Closed-loop simulator & model validation ├── utils.py # Plotting, reporting & helper utilities │ ├── outputs/ # Auto-generated figures and reports │ ├── singular_values.png │ ├── hsv_balanced_truncation.png │ ├── validation_predictions.png │ ├── residuals.png │ ├── metrics_heatmap.png │ ├── mpc_weight_optimisation.png │ ├── mpc_output_tracking.png │ ├── mpc_inputs.png │ ├── mpc_cost.png │ └── blown_film_sysid_mpc_report.txt │ ├── requirements.txt # Python dependencies ├── LICENSE # MIT License └── README.md # This file

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
Four-stage reduction pipeline.

| Class | Responsibility |
|-------|---------------|
| `ReducedModel` | Container for reduced + augmented matrices |
| `ModelReducer` | Balanced truncation → POD/Galerkin → ZOH → augmentation |

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
q_thickness = 10.0 (layer thickness — highest priority) q_temperature = 5.0 (melt temperature) q_default = 1.0 (all other outputs)

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