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
  - [Monitored Stations](#monitored-stations)
  - [Key Process Variables](#key-process-variables)
- [Mathematical Background](#-mathematical-background)
- [Original PDE/ODE System Model](#-original-pdeode-system-model)
  - [1. Extruder Dynamics](#1-extruder-dynamics)
  - [2. Dosing / Feeder Dynamics](#2-dosing--feeder-dynamics)
  - [3. Die Head Thermal Dynamics](#3-die-head-thermal-dynamics)
  - [4. Blown Film Bubble Dynamics](#4-blown-film-bubble-dynamics)
  - [5. Cooling Dynamics](#5-cooling-dynamics)
  - [6. Haul-Off Tower Dynamics](#6-haul-off-tower-dynamics)
  - [7. Winder Dynamics](#7-winder-dynamics)
  - [8. Layer Thickness & Output Coupling](#8-layer-thickness--output-coupling)
  - [9. Production Management](#9-production-management)
  - [10. Full-Order State Vector](#10-full-order-state-vector)
  - [11. Coupled System Summary](#11-coupled-system-summary)
  - [12. From PDE Model to Data-Driven State Space](#12-from-pde-model-to-data-driven-state-space)
- [State Space Linearisation from the Nonlinear PDE Model](#-state-space-linearisation-from-the-nonlinear-pde-model)
  - [1. Spatial Discretisation of PDEs](#1-spatial-discretisation-of-pdes)
  - [2. Operating Point Definition](#2-operating-point-definition)
  - [3. Jacobian Linearisation](#3-jacobian-linearisation)
  - [4. Continuous-Time Linearised System](#4-continuous-time-linearised-system)
  - [5. Discretisation to Discrete-Time State Space](#5-discretisation-to-discrete-time-state-space)
  - [6. Validity Region of the Linearisation](#6-validity-region-of-the-linearisation)
  - [7. From Linearised Model to Data-Driven Identification](#7-from-linearised-model-to-data-driven-identification)
  - [8. Summary: Linearisation Pipeline](#8-summary-linearisation-pipeline)
  - [State Space Model](#state-space-model)
  - [Model Order Reduction](#model-order-reduction)
  - [MPC Optimisation Problem](#mpc-optimisation-problem)
- [Project Structure](#-project-structure)
- [Module Descriptions](#-module-descriptions)
  - [`config.py`](#configpy)
  - [`data_manager.py`](#data_managerpy)
  - [`system_identification.py`](#system_identificationpy)
  - [`model_reduction.py`](#model_reductionpy)
  - [`accuracy.py`](#accuracypy)
  - [`estimation.py`](#estimationpy)
  - [`mpc_controller.py`](#mpc_controllerpy)
  - [`simulation.py`](#simulationpy)
  - [`persistence.py`](#persistencepy)
  - [`utils.py`](#utilspy)
  - [`main.py`](#mainpy)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
  - [CLI Arguments](#cli-arguments)
- [Configuration](#-configuration)
- [Pipeline Stages](#-pipeline-stages)
- [Outputs](#-outputs)
- [Dataset Format](#-dataset-format)
- [Performance Notes](#-performance-notes)
- [Model & Weight Caching](#-model--weight-caching)
- [Accuracy Gate](#-accuracy-gate)
- [OOP Design Principles](#-oop-design-principles)
- [Dependencies](#-dependencies)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

This repository implements a complete **data-driven control engineering
pipeline** for an industrial co-extrusion blown film manufacturing line.
Starting from raw SCADA/PLC time-series data, the pipeline:

1. **Identifies** a discrete-time linear state space model from
   input-output data using either the **N4SID subspace algorithm** (black-box)
   or **grey-box linearisation** (physics-based).
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

## 📐 Original PDE/ODE System Model

Before data-driven system identification is applied, the blown film
line is described by a **first-principles mathematical model** derived
from conservation laws, constitutive relations, and empirical
correlations. This section documents the full physics-based model that
motivates the state space structure used in the identification and
MPC pipeline.

The model spans **nine coupled physical domains**, each contributing
states to the full-order system of dimension $`n_x = 146`$.

---

### 1. Extruder Dynamics

#### 1.1 Melt Flow Rate

The volumetric throughput of extruder $`k`$ is governed by the
classical drag-pressure flow decomposition for a single-screw
extruder:

```math
Q_k = \alpha_k N_k - \beta_k \frac{\Delta P_k}{\mu_k(T_k,\,\dot{\gamma}_k)}
```

| Symbol | Description | Unit |
|--------|-------------|------|
| $`Q_k`$ | Volumetric flow rate | m³/s |
| $`\alpha_k`$ | Drag flow coefficient (screw geometry) | m³/rev |
| $`N_k`$ | Screw rotational speed | rpm |
| $`\beta_k`$ | Pressure flow coefficient | m³·s/kg |
| $`\Delta P_k`$ | Pressure drop across screw | Pa |
| $`\mu_k`$ | Non-Newtonian melt viscosity | Pa·s |

#### 1.2 Non-Newtonian Viscosity

The melt follows a **power-law (Ostwald–de Waele)** model with
Arrhenius temperature dependence:

```math
\mu_k(T_k,\,\dot{\gamma}_k)
= m_k(T_k)\cdot\dot{\gamma}_k^{\,n_k - 1}
```

```math
m_k(T_k)
= m_{0,k}\exp\!\left(\frac{E_{a,k}}{R\,T_k}\right)
```

| Symbol | Description | Unit |
|--------|-------------|------|
| $`\dot{\gamma}_k`$ | Shear rate | s⁻¹ |
| $`n_k`$ | Power-law index | — |
| $`m_{0,k}`$ | Reference consistency index | Pa·sⁿ |
| $`E_{a,k}`$ | Flow activation energy | J/mol |
| $`R`$ | Universal gas constant | J/(mol·K) |

#### 1.3 Melt Pressure ODE

```math
\frac{dP_k}{dt}
= \frac{K_k}{\rho_k}
\bigl(\dot{m}_{in,k} - \dot{m}_{out,k}\bigr)
```

where $`K_k`$ is the bulk modulus of the polymer melt.

#### 1.4 Barrel Energy Balance PDE

The temperature field along the screw axis $`z`$ evolves according
to the **convection–diffusion–reaction** equation:

```math
\rho_k c_{p,k}
\frac{\partial T_k}{\partial t} +
\rho_k c_{p,k}\,v_{z,k}
\frac{\partial T_k}{\partial z}
= \lambda_k
\frac{\partial^2 T_k}{\partial z^2} +
\underbrace{\eta_k\,\dot{\gamma}_k^2}_{\text{viscous dissipation}} +
\underbrace{\dot{q}_{wall,k}(z,t)}_{\text{barrel heating}}
```

| Symbol | Description | Unit |
|--------|-------------|------|
| $`v_{z,k}`$ | Axial melt velocity | m/s |
| $`\lambda_k`$ | Melt thermal conductivity | W/(m·K) |
| $`\eta_k\dot{\gamma}_k^2`$ | Viscous dissipation (→ `DissipationPwr`) | W/m³ |
| $`\dot{q}_{wall,k}`$ | Wall heat flux from barrel zones | W/m³ |

#### 1.5 Barrel Zone Heat Flux

Each of the $`j = 1,\ldots,8`$ heating zones contributes:

```math
\dot{q}_{wall,k,j}
= h_{k,j}\,A_{k,j}
\bigl(T_{sp,k,j} - T_k(z_j,t)\bigr)
\cdot P_{eff,k,j}(t)
```

```math
\frac{dT_{set,k,j}}{dt}
= \frac{1}{\tau_{k,j}}
\bigl(T_{sp,k,j} - T_{set,k,j}\bigr) -
K_{P,k,j}\,e_{k,j}(t)
```

where $`e_{k,j} = T_{set,k,j} - T_{act,k,j}`$ is the zone PID
error signal (mapped from SCADA tags `Regler_X`, `Regler_Y`,
`ActEffectPower`).

---

### 2. Dosing / Feeder Dynamics

For component $`i`$ in extruder $`k`$ (up to 5 components per
extruder, mapped from `Dos_2` through `Dos_5`):

#### 2.1 Component Mass Balance

```math
\frac{dm_{i,k}}{dt}
= \dot{m}_{feed,i,k} -
\rho_{bulk,i,k}\,Q_{dos,i,k}(N_{dos,i,k})
```

#### 2.2 Actual Proportion

```math
\phi_{i,k}(t)
= \frac{\dot{m}_{out,i,k}}
       {\displaystyle\sum_{i=1}^{n}\dot{m}_{out,i,k}}
```

→ `Dos_i_IstAnteil`

#### 2.3 PI Dosing Control Law

```math
\frac{dN_{dos,i,k}}{dt}
= K_{c,i,k}
\bigl(\phi_{i,k}^{set} - \phi_{i,k}\bigr) +
\frac{K_{c,i,k}}{\tau_{I,i,k}}
\int_0^t
\bigl(\phi_{i,k}^{set} - \phi_{i,k}\bigr)\,d\tau
```

#### 2.4 Mix Density

```math
\rho_{mix,k}(t)
= \sum_{i=1}^{n}
\phi_{i,k}(t)\cdot\rho_{bulk,i,k}
```

→ `MischDicht`

---

### 3. Die Head Thermal Dynamics

#### 3.1 Die Zone Temperature PDE

The annular die body is modelled in cylindrical coordinates
$`(r, z)`$:

```math
\rho_{die}\,c_{p,die}
\frac{\partial T_{die}}{\partial t}
= \lambda_{die}
\left(
  \frac{1}{r}
  \frac{\partial}{\partial r}
  \!\left(r\frac{\partial T_{die}}{\partial r}\right) +
  \frac{\partial^2 T_{die}}{\partial z^2}
\right) +
\dot{q}_{heater}(r,z,t)
```

Zones $`j = 3,\ldots,9`$ are monitored via
`HeizungZone_j_SollTemp`, `HeizungZone_j_Regler_Y`.

#### 3.2 Zone Heater Power Control

```math
P_{eff,j}(t)
= \frac{P_{eff,j}^{max}}{100}\cdot u_j(t),
\qquad u_j \in [0,100]\,\%
```

```math
\frac{du_j}{dt}
= K_{P,j}\frac{de_j}{dt} +
\frac{K_{P,j}}{\tau_{I,j}}\,e_j(t),
\qquad
e_j = T_{sp,j} - T_{act,j}
```

#### 3.3 Annular Melt Flow (Stokes)

```math
\frac{\partial v_z}{\partial t}
= -\frac{1}{\rho}\frac{\partial P}{\partial z} +
\frac{1}{r}\frac{\partial}{\partial r}
\!\left(r\,\mu\frac{\partial v_z}{\partial r}\right)
```

with no-slip boundary conditions:
$`v_z(R_{inner},t) = 0`$,
$`v_z(R_{outer},t) = 0`$.

---

### 4. Blown Film Bubble Dynamics

The free-surface bubble is the most geometrically complex
subsystem, requiring a **moving-boundary** formulation.

#### 4.1 Bubble Radius PDE

```math
\frac{\partial R}{\partial t} +
v_z\frac{\partial R}{\partial z}
= \frac{R}{2}
\bigl(\dot{\varepsilon}_\theta - \dot{\varepsilon}_z\bigr)
```

#### 4.2 Film Thickness Evolution PDE

```math
\frac{\partial h}{\partial t} +
v_z\frac{\partial h}{\partial z}
= -h
\bigl(\dot{\varepsilon}_z + \dot{\varepsilon}_\theta\bigr)
```

| Symbol | Description | SCADA Tag |
|--------|-------------|-----------|
| $`h(z,t)`$ | Local film thickness | `SDickeIst` |
| $`\dot{\varepsilon}_z`$ | Axial strain rate | — |
| $`\dot{\varepsilon}_\theta`$ | Hoop strain rate | — |

#### 4.3 Axial Force Balance

```math
F_z = 2\pi R\,h\,\sigma_{zz} + \pi R^2\,\Delta P_{bubble}
```

```math
\frac{dF_z}{dz} = 0
\qquad\text{(quasi-steady axial tension)}
```

#### 4.4 Bubble Pressure ODE (IBC Control)

The internal bubble pressure is governed by the net mass flow
through the three IBC units ($`l = 1,2,3`$):

```math
\frac{dP_{bubble}}{dt}
= \frac{\gamma_{air}}{V_{bubble}}
\bigl(\dot{m}_{in,IBC} - \dot{m}_{out,IBC}\bigr)
\frac{R_{gas}\,T_{air}}{M_{air}}
```

```math
\dot{m}_{in,IBC,l}
= f\!\bigl(N_{IBC,l},\,P_{bubble}\bigr)
```

→ `VARIBC_l_Ist_n_Calc`

---

### 5. Cooling Dynamics

#### 5.1 Film Temperature PDE

The thin-film energy balance couples external air cooling and
internal IBC cooling:

```math
\rho_f\,c_{p,f}\,h
\frac{\partial T_f}{\partial t}
= -h_{conv,ext}(T_f - T_{air,ext}) -
h_{conv,int}(T_f - T_{IBC}) +
\dot{q}_{rad}
```

#### 5.2 IBC Air Temperature ODE (unit $`l`$)

```math
\rho_{air}\,V_{IBC,l}\,c_{p,air}
\frac{dT_{IBC,l}}{dt}
= \dot{m}_{in,l}\,c_{p,air}(T_{in,l} - T_{IBC,l}) -
UA_l(T_{IBC,l} - T_{film})
```

`KuehlGeraet_l_IstTemp` ↔ $`T_{IBC,l}`$

#### 5.3 IBC Speed Control ODE

```math
\frac{dN_{IBC,l}}{dt}
= \frac{1}{\tau_{IBC}}
\bigl(N_{IBC,l}^{set} - N_{IBC,l}\bigr)
```

→ `VARIBC_l_Soll_n_Visu`

#### 5.4 Cooling Device Temperature ODE

```math
M_{cool,l}\,c_{p,cool}
\frac{dT_{cool,l}}{dt}
= \dot{Q}_{in,l} -
UA_{cool,l}(T_{cool,l} - T_{amb})
```

---

### 6. Haul-Off Tower Dynamics

#### 6.1 Haul-Off Speed ODE

```math
\frac{dv_{haul}}{dt}
= \frac{1}{J_{haul}}
\bigl[
  K_v(v_{haul}^{set} - v_{haul}) -
  \sigma_{haul}\,h_{film}\,W_{film} -
  F_{fric}
\bigr]
```

`VARAbzug_1_IstZu` ↔ $`v_{haul}`$

#### 6.2 Film Tension ODE

```math
\frac{d\sigma_{haul}}{dt}
= \frac{E_f\,h_{film}\,W_{film}}{L_{haul}}
\bigl(v_{haul} - v_{bubble,FLH}\bigr)
```

| Symbol | Description | Unit |
|--------|-------------|------|
| $`E_f`$ | Film elastic modulus | Pa |
| $`W_{film}`$ | Lay-flat film width | m |
| $`L_{haul}`$ | Distance from frost line to nip | m |
| $`v_{bubble,FLH}`$ | Film velocity at frost line height | m/s |

---

### 7. Winder Dynamics

For winders $`w = 1`$ (`ST113`) and $`w = 2`$ (`ST114`):

#### 7.1 Roll Build-Up ODE

```math
\frac{dR_{roll,w}}{dt}
= \frac{h_{film}\,v_{haul}}
       {2\pi\,R_{roll,w}\,\eta_{pack}}
```

→ `VARDiaRollRc`

#### 7.2 Accumulated Length ODE

```math
\frac{dL_w}{dt} = v_{haul}(t),
\qquad
L_{rem,w}(t) = L_{target,w} - L_w(t)
```

`VARActLen`, `VARRemainingLen` ↔ $`L_w`$, $`L_{rem,w}`$

#### 7.3 Winder Drum Dynamics

```math
J_w\frac{d\omega_{drum,w}}{dt}
= T_{drive,w} -
\sigma_{web,w}\,h_{film}\,W_{film}\,R_{roll,w} -
B_w\,\omega_{drum,w}
```

```math
\dot{R}_{roll,w}
= \frac{h_{film}\,v_{haul}}{2\pi\,R_{roll,w}}
```

#### 7.4 Web Tension ODE

```math
\frac{d\sigma_{web,w}}{dt}
= \frac{E_f\,h_{film}\,W_{film}}{L_{span,w}}
\bigl(\omega_{drum,w}\,R_{roll,w} - v_{haul}\bigr)
```

→ `VARWdSpTens`

#### 7.5 Tension Taper (Clp Curve)

The tension setpoint is tapered as the roll builds up to prevent
telescoping:

```math
\sigma_{web,w}^{set}(R)
= \sigma_{0,w}
\left(
  1 - \kappa_w\cdot
  \frac{R_{roll,w} - R_{core,w}}
       {R_{max,w} - R_{core,w}}
\right)
```

where $`\kappa_w`$ is the taper factor
(→ `VARClpTens`, `VARWdTapeReductVal`).

---

### 8. Layer Thickness & Output Coupling

#### 8.1 Total Mass Flow Rate

```math
\dot{m}_{total}(t)
= \sum_{k=0}^{K-1}
\rho_{mix,k}(t)\cdot Q_k(t)
```

→ `VAREx_k_GesamtDS`

#### 8.2 Individual Layer Thickness Fraction

```math
\delta_k(t)
= \frac{Q_k(t)}{\displaystyle\sum_j Q_j(t)}
\cdot h_{total}(t)
```

→ `VAREx_k_SDickeProz`

#### 8.3 Global Film Thickness (Mass Balance)

```math
h_{total}(z,t)
= \frac{\dot{m}_{total}}
       {2\pi\,R(z,t)\cdot\rho_{mix}\cdot v_z(z,t)}
```

→ `VAREx_k_SDickeIst`

---

### 9. Production Management

```math
\frac{dL_{job}}{dt} = v_{haul}(t)
```

```math
\frac{dm_{job}}{dt}
= \rho_{film}\cdot h_{total}\cdot W_{film}\cdot v_{haul}(t)
```

`VAREx_0_Dos_0_IstLMGewicht` ↔ $`m_{job}`$

---

### 10. Full-Order State Vector

The complete first-principles model has state dimension
$`n_x = 146`$, partitioned as:

| Subsystem | States | Dimension |
|-----------|--------|-----------|
| Extruder barrel temperatures ($`T_{k,j}`$, 4 extruders × 8 zones) | $`T_{k,j}`$, $`P_k`$, $`\dot{m}_{out,k}`$, $`T_{melt,k}`$ | 44 |
| Dosing (4 extruders × 5 components) | $`\phi_{i,k}`$, $`N_{dos,i,k}`$, $`m_{i,k}`$ | 60 |
| Die head (7 zones, temp + actuator) | $`T_{die,j}`$, $`u_{die,j}`$ | 14 |
| Bubble | $`R_{bub}`$, $`h_{film}`$, $`P_{bub}`$, $`v_z`$, $`T_f`$, $`\sigma_{zz}`$ | 6 |
| Cooling (3 IBC + 3 devices) | $`T_{IBC,l}`$, $`N_{IBC,l}`$, $`T_{cool,l}`$ | 9 |
| Haul-off | $`v_{haul}`$, $`\sigma_{haul}`$, $`L_{job}`$ | 3 |
| Winders 1 & 2 | $`\omega_{drum,w}`$, $`R_{roll,w}`$, $`L_w`$, $`\sigma_{web,w}`$, $`T_{drive,w}`$ | 10 |
| **Total** | | **146** |

---

### 11. Coupled System Summary

The full nonlinear system can be written compactly as:

```math
\dot{\mathbf{x}}(t)
= \mathbf{f}\!\bigl(\mathbf{x}(t),\,\mathbf{u}(t)\bigr),
\qquad
\mathbf{x} \in \mathbb{R}^{146},\;
\mathbf{u} \in \mathbb{R}^{96}
```

```math
\mathbf{y}(t)
= \mathbf{g}\!\bigl(\mathbf{x}(t),\,\mathbf{u}(t)\bigr),
\qquad
\mathbf{y} \in \mathbb{R}^{58}
```

The key coupling structure is:

```
[Dosing ODEs] ──► [Extruder PDEs] ──► [Die Head PDEs]
                        │                     │
                   [Pressure ODE]       [Melt Flow PDE]
                        │                     │
                        └──────────┬──────────┘
                                   ▼
                          [Bubble Dynamics PDEs]
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼               ▼
              [IBC Cooling]   [Air Ring]    [Film Thickness]
                    │                            │
                    └────────────────────────────┘
                                   │
                           [Haul-Off ODE]
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
              [Winder 1 ODEs]             [Winder 2 ODEs]
```

---

### 12. From PDE Model to Data-Driven State Space

The first-principles model is **not used directly** for control
because:

| Challenge | Detail |
|-----------|--------|
| **Nonlinearity** | Power-law viscosity, bubble free-surface, strain-rate coupling |
| **Unknown parameters** | $`E_{a,k}`$, $`n_k`$, $`m_{0,k}`$, $`h_{conv}`$, $`UA_l`$ vary with material grade |
| **Moving boundary** | Bubble radius $`R(z,t)`$ requires mesh adaptation |
| **Stiffness** | Time constants span 0.1 s (drives) to 1800 s (roll build-up) |
| **Distributed parameters** | PDEs require spatial discretisation → very high state dimension |

Instead, the PDE model **motivates the state space structure** and
**informs signal selection** (which tags are inputs vs outputs),
while the actual model matrices $`(A, B, C, D)`$ are identified
from SCADA data using the **N4SID subspace algorithm** described
in the [`system_identification.py`](#system_identificationpy) section.

The reduction from $`n_x = 146`$ (first-principles) to
$`\tilde{n} \approx 22`$ (MPC-ready) is achieved through the
**model order reduction pipeline**:

```
First-Principles PDE Model  →  n = 146
         ↓  Linearisation at operating point
Linearised State Space      →  n = 146
         ↓  Singular Perturbation (fast states)
Slow Subsystem              →  n ≈ 90
         ↓  Balanced Truncation (HSV)
Balanced Reduced Model      →  n ≈ 40
         ↓  POD / Galerkin Projection
POD Reduced Model           →  n ≈ 22
         ↓  ZOH Discretisation (Ts = 3 s)
Discrete MPC Model          →  n ≈ 22
         ↓  Integrator Augmentation
Offset-Free MPC Model       →  n ≈ 80  (22 dynamic + 58 disturbance)
```

> **Note:** The numerical solution of the full PDE system requires
> implicit time-stepping (e.g. Crank–Nicolson for the thermal PDEs),
> Newton–Raphson iterations for the nonlinear viscosity terms, and
> a moving-mesh or level-set method for the bubble free surface.
> These are outside the scope of this repository, which focuses on
> the data-driven identification and MPC design workflow.

---

## 🔢 State Space Linearisation from the Nonlinear PDE Model

This section documents the systematic procedure for deriving the
**linearised state space model** from the nonlinear distributed-
parameter PDE/ODE system described in the previous section.
The linearisation is performed around a **nominal operating point**
and proceeds through five formal stages:

1. Spatial discretisation of PDEs → finite-dimensional ODE system
2. Operating point definition
3. Jacobian linearisation
4. Time-scale separation (singular perturbation)
5. Assembly of the linearised $`(A, B, C, D)`$ matrices

---

### 1. Spatial Discretisation of PDEs

Before linearisation can be applied, the infinite-dimensional PDE
system must be reduced to a finite-dimensional ODE system via
**spatial discretisation**.

#### 1.1 Extruder Barrel — Finite Difference (Method of Lines)

The axial temperature PDE along screw direction $`z \in [0, L_k]`$
is discretised into $`N_z`$ equally spaced nodes
$`z_j = j\,\Delta z`$, $`\Delta z = L_k / N_z`$:

```math
\rho_k c_{p,k}
\frac{dT_{k,j}}{dt}
= \frac{\lambda_k}{\Delta z^2}
\bigl(T_{k,j+1} - 2T_{k,j} + T_{k,j-1}\bigr) -
\frac{\rho_k c_{p,k} v_{z,k}}{\Delta z}
\bigl(T_{k,j} - T_{k,j-1}\bigr) +
\eta_k\dot{\gamma}_k^2 +
\dot{q}_{wall,k,j}
```

This converts the extruder PDE into a system of $`N_z`$ coupled ODEs
per extruder, which in matrix form reads:

```math
\frac{d\mathbf{T}_k}{dt}
= \underbrace{
    \frac{\lambda_k}{\rho_k c_{p,k} \Delta z^2}\mathbf{D}_2 -
    \frac{v_{z,k}}{\Delta z}\mathbf{D}_1
  }_{\mathbf{M}_k}
\mathbf{T}_k +
\mathbf{f}_{visc,k} +
\mathbf{B}_{wall,k}\mathbf{u}_{zone,k}
```

where:
- $`\mathbf{D}_2 \in \mathbb{R}^{N_z \times N_z}`$ is the
  second-order finite difference matrix (tridiagonal)
- $`\mathbf{D}_1 \in \mathbb{R}^{N_z \times N_z}`$ is the
  first-order upwind difference matrix
- $`\mathbf{f}_{visc,k}`$ is the viscous dissipation vector
- $`\mathbf{B}_{wall,k}`$ maps zone setpoints to nodal heat fluxes

#### 1.2 Die Head — Finite Element (Radial Direction)

The 2D die PDE in $`(r, z)`$ is discretised using
**linear finite elements** in $`r`$ and finite differences in $`z`$.
The resulting ODE system is:

```math
\mathbf{M}_{die}
\frac{d\mathbf{T}_{die}}{dt}
= \mathbf{K}_{die}\mathbf{T}_{die} +
\mathbf{F}_{die}(\mathbf{u}_{die})
```

where $`\mathbf{M}_{die}`$ is the FE mass matrix,
$`\mathbf{K}_{die}`$ is the stiffness matrix, and
$`\mathbf{F}_{die}`$ is the load vector from heater inputs.

#### 1.3 Bubble — Galerkin Spectral Discretisation

The bubble radius and film thickness PDEs are projected onto
$`N_{bub}`$ spectral basis functions $`\psi_n(z)`$:

```math
R(z,t) \approx \sum_{n=1}^{N_{bub}} r_n(t)\,\psi_n(z),
\qquad
h(z,t) \approx \sum_{n=1}^{N_{bub}} h_n(t)\,\psi_n(z)
```

yielding the Galerkin ODE system:

```math
\frac{d\mathbf{r}}{dt}
= \mathbf{G}_{bub}(\mathbf{r}, \mathbf{h}, P_{bub}, v_z)
```

```math
\frac{d\mathbf{h}_{bub}}{dt}
= \mathbf{H}_{bub}(\mathbf{r}, \mathbf{h}, \dot{\varepsilon}_z,
  \dot{\varepsilon}_\theta)
```

After discretisation, the **full nonlinear ODE system** takes the
compact form:

```math
\boxed{
\dot{\mathbf{x}}(t)
= \mathbf{f}\!\bigl(\mathbf{x}(t),\,\mathbf{u}(t)\bigr),
\qquad
\mathbf{x} \in \mathbb{R}^{146},\;
\mathbf{u} \in \mathbb{R}^{96}
}
```

---

### 2. Operating Point Definition

The linearisation is performed around a **steady-state nominal
operating point** $`(\mathbf{x}_0, \mathbf{u}_0)`$ satisfying:

```math
\mathbf{f}(\mathbf{x}_0,\,\mathbf{u}_0) = \mathbf{0}
```

This is found by solving the nonlinear algebraic system using
Newton–Raphson iteration:

```math
\mathbf{x}_0^{(i+1)}
= \mathbf{x}_0^{(i)} -
\left[
    \frac{\partial \mathbf{f}}{\partial \mathbf{x}}
    \bigg|_{\mathbf{x}_0^{(i)},\,\mathbf{u}_0}
  \right]^{-1}
\mathbf{f}\!\bigl(\mathbf{x}_0^{(i)},\,\mathbf{u}_0\bigr)
```

The nominal operating point corresponds to a **typical production
recipe** with the following key values:

| Variable | Nominal Value | SCADA Tag |
|----------|--------------|-----------|
| Barrel zone temperatures | 180–240 °C | `HeizungZone_j_SollTemp` |
| Melt temperature | 220–250 °C | `Massetemperatur` |
| Melt pressure | 150–300 bar | `druck_1_IstP` |
| Layer thickness | 20–80 µm | `SDickeIst` |
| Haul-off speed | 20–60 m/min | `VARAbzug_1_IstZu` |
| IBC speed | 800–2000 rpm | `VARIBC_l_Ist_n_Calc` |
| Roll diameter | 300–600 mm | `VARDiaRollRc` |

The **perturbation variables** are defined as:

```math
\tilde{\mathbf{x}}(t) = \mathbf{x}(t) - \mathbf{x}_0,
\qquad
\tilde{\mathbf{u}}(t) = \mathbf{u}(t) - \mathbf{u}_0,
\qquad
\tilde{\mathbf{y}}(t) = \mathbf{y}(t) - \mathbf{y}_0
```

---

### 3. Jacobian Linearisation

#### 3.1 General Procedure

Expanding $`\mathbf{f}`$ in a first-order Taylor series about
$`(\mathbf{x}_0, \mathbf{u}_0)`$:

```math
\dot{\tilde{\mathbf{x}}}
= \underbrace{
    \frac{\partial \mathbf{f}}{\partial \mathbf{x}}
    \bigg|_{\mathbf{x}_0,\,\mathbf{u}_0}
  }_{\mathbf{A}_c}
\tilde{\mathbf{x}} +
\underbrace{
    \frac{\partial \mathbf{f}}{\partial \mathbf{u}}
    \bigg|_{\mathbf{x}_0,\,\mathbf{u}_0}
  }_{\mathbf{B}_c}
\tilde{\mathbf{u}} +
\underbrace{
    \mathcal{O}\!\left(\|\tilde{\mathbf{x}}\|^2,
    \|\tilde{\mathbf{u}}\|^2\right)
  }_{\text{neglected}}
```

```math
\tilde{\mathbf{y}}
= \underbrace{
    \frac{\partial \mathbf{g}}{\partial \mathbf{x}}
    \bigg|_{\mathbf{x}_0,\,\mathbf{u}_0}
  }_{\mathbf{C}_c}
\tilde{\mathbf{x}} +
\underbrace{
    \frac{\partial \mathbf{g}}{\partial \mathbf{u}}
    \bigg|_{\mathbf{x}_0,\,\mathbf{u}_0}
  }_{\mathbf{D}_c}
\tilde{\mathbf{u}}
```

#### 3.2 Analytical Jacobian Blocks

The Jacobian $`\mathbf{A}_c`$ has a **sparse block structure**
reflecting the physical coupling between subsystems:

```math
\mathbf{A}_c =
\begin{bmatrix}
\mathbf{A}_{ee} & \mathbf{0}      & \mathbf{0}        & \mathbf{0}      & \mathbf{0}      & \mathbf{0}      & \mathbf{0}      \\
\mathbf{A}_{de} & \mathbf{A}_{dd} & \mathbf{0}        & \mathbf{0}      & \mathbf{0}      & \mathbf{0}      & \mathbf{0}      \\
\mathbf{A}_{\partial e} & \mathbf{0} & \mathbf{A}_{\partial\partial} & \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{0} \\
\mathbf{A}_{be} & \mathbf{0}      & \mathbf{A}_{b\partial} & \mathbf{A}_{bb} & \mathbf{0}  & \mathbf{0}      & \mathbf{0}      \\
\mathbf{0}      & \mathbf{0}      & \mathbf{0}        & \mathbf{A}_{cb} & \mathbf{A}_{cc} & \mathbf{0}      & \mathbf{0}      \\
\mathbf{0}      & \mathbf{0}      & \mathbf{0}        & \mathbf{A}_{hb} & \mathbf{A}_{hc} & \mathbf{A}_{hh} & \mathbf{0}      \\
\mathbf{0}      & \mathbf{0}      & \mathbf{0}        & \mathbf{0}      & \mathbf{0}      & \mathbf{A}_{wh} & \mathbf{A}_{ww}
\end{bmatrix}
\in \mathbb{R}^{146 \times 146}
```

where the subscripts denote:
$`e`$ = extruder, $`d`$ = dosing, $`\partial`$ = die,
$`b`$ = bubble, $`c`$ = cooling, $`h`$ = haul-off, $`w`$ = winder.

The key **off-diagonal coupling blocks** are:

| Block | Physical Meaning |
|-------|-----------------|
| $`\mathbf{A}_{de}`$ | Dosing proportions affect extruder melt density |
| $`\mathbf{A}_{\partial e}`$ | Extruder melt temperature drives die inlet BC |
| $`\mathbf{A}_{b\partial}`$ | Die pressure drives bubble inflation |
| $`\mathbf{A}_{be}`$ | Extruder throughput sets bubble mass flow |
| $`\mathbf{A}_{cb}`$ | Bubble temperature couples to IBC cooling |
| $`\mathbf{A}_{hb}`$ | Bubble exit velocity drives haul-off tension |
| $`\mathbf{A}_{hc}`$ | Cooling affects film stiffness at haul-off |
| $`\mathbf{A}_{wh}`$ | Haul-off speed drives winder drum speed |

#### 3.3 Extruder Jacobian Block

For the discretised barrel temperature states
$`\tilde{T}_{k,j}`$, the diagonal block is:

```math
[\mathbf{A}_{ee}]_{k,j,j}
= -\frac{h_{k,j}A_{k,j}P_{eff,k,j}^0
         + \dot{m}_{out,k}^0 c_{p,k}}
        {\rho_k c_{p,k} V_{k,j}} -
\frac{v_{z,k}}{\Delta z} -
\frac{2\lambda_k}{\rho_k c_{p,k} \Delta z^2}
```

The super- and sub-diagonal entries (axial coupling):

```math
[\mathbf{A}_{ee}]_{k,j,j+1}
= \frac{\lambda_k}{\rho_k c_{p,k} \Delta z^2}
```

```math
[\mathbf{A}_{ee}]_{k,j,j-1}
= \frac{\lambda_k}{\rho_k c_{p,k} \Delta z^2} +
\frac{v_{z,k}}{\Delta z}
```

The viscosity nonlinearity contributes an additional term via
the chain rule:

```math
\frac{\partial f_{T_k}}{\partial T_k}
\bigg|_{\text{visc}}
= \frac{\partial}{\partial T_k}
\!\left(\frac{\eta_k(T_k)\,\dot{\gamma}_k^2}
              {\rho_k c_{p,k}}\right)
= \frac{\dot{\gamma}_k^2}{\rho_k c_{p,k}}
\cdot m_{0,k}\,\dot{\gamma}_k^{n_k-1}
\cdot\left(-\frac{E_{a,k}}{R\,T_{k,0}^2}\right)
e^{E_{a,k}/(R T_{k,0})}
```

#### 3.4 Bubble Jacobian Block

The bubble subsystem has the most complex Jacobian due to the
nonlinear strain-rate coupling. Linearising the thickness
evolution:

```math
\frac{\partial \dot{h}}{\partial h}
\bigg|_0
= -\bigl(\dot{\varepsilon}_{z,0} + \dot{\varepsilon}_{\theta,0}\bigr)
```

```math
\frac{\partial \dot{h}}{\partial v_z}
\bigg|_0
= -\frac{\partial h}{\partial z}\bigg|_0
```

```math
\frac{\partial \dot{R}}{\partial R}
\bigg|_0
= \frac{1}{2}
\bigl(\dot{\varepsilon}_{\theta,0} - \dot{\varepsilon}_{z,0}\bigr)
```

The stress relaxation linearisation:

```math
\frac{\partial \dot{\sigma}_{zz}}{\partial \sigma_{zz}}
\bigg|_0
= -\frac{1}{\tau_{relax}}
```

The full bubble block in linearised form:

```math
\mathbf{A}_{bb}
=
\begin{bmatrix}
  a_{RR}   & 0        & 0        & 0        & 0        & 0        \\
  0        & a_{hh}   & 0        & a_{hv}   & 0        & 0        \\
  0        & 0        & a_{PP}   & 0        & 0        & 0        \\
  0        & a_{vh}   & a_{vP}   & 0        & 0        & a_{v\sigma} \\
  0        & a_{Th}   & 0        & 0        & a_{TT}   & 0        \\
  0        & 0        & 0        & a_{\sigma v} & 0   & a_{\sigma\sigma}
\end{bmatrix}
```

where:

```math
a_{RR} = \tfrac{1}{2}(\dot{\varepsilon}_{\theta,0}
         - \dot{\varepsilon}_{z,0}), \quad
a_{hh} = -(\dot{\varepsilon}_{z,0}
         + \dot{\varepsilon}_{\theta,0}), \quad
a_{PP} = -\frac{\gamma_{air} R_{gas} T_{air}}
               {M_{air} V_{bub,0}^2}\dot{m}_{net,0}
```

```math
a_{TT} = -\frac{h_{ext} + h_{int}}
               {\rho_f c_{p,f} h_{film,0}}, \quad
a_{\sigma\sigma} = -\frac{1}{\tau_{relax}}, \quad
a_{v\sigma} = \frac{1}{\rho_f h_{film,0}}
```

#### 3.5 Winder Jacobian Block

For winder $`w`$, the $`5 \times 5`$ diagonal block is:

```math
\mathbf{A}_{ww,w}
=
\begin{bmatrix}
  -B_w/J_w
  & -\sigma_{web,0}hW/J_w
  & 0
  & -R_{roll,0}hW/J_w
  & 1/J_w \\
  0
  & 0
  & 0
  & 0
  & 0 \\
  0
  & 0
  & 0
  & 0
  & 0 \\
  \omega_{drum,0}E_f hW/L_{span}
  & v_{haul,0}E_f hW/L_{span}
  & 0
  & 0
  & 0 \\
  0
  & 0
  & 0
  & 0
  & -1/\tau_{drive,w}
\end{bmatrix}
```

#### 3.6 Input Jacobian $`\mathbf{B}_c`$

The input matrix maps manipulated variables to state derivatives.
The non-zero blocks are:

**Extruder zone setpoints** → barrel temperatures:

```math
[\mathbf{B}_c]_{T_{k,j},\,T_{sp,k,j}}
= \frac{h_{k,j}\,A_{k,j}\,P_{eff,k,j}^0}
       {\rho_k\,c_{p,k}\,V_{k,j}}
```

**IBC speed setpoints** → IBC speed states:

```math
[\mathbf{B}_c]_{N_{IBC,l},\,N_{IBC,l}^{set}}
= \frac{1}{\tau_{IBC}}
```

**Haul-off speed setpoint** → haul-off velocity:

```math
[\mathbf{B}_c]_{v_{haul},\,v_{haul}^{set}}
= \frac{K_v}{J_{haul}}
```

**Winder torque setpoint** → drum angular velocity:

```math
[\mathbf{B}_c]_{\omega_{drum,w},\,T_{drive,w}^{set}}
= \frac{1}{J_w\,\tau_{drive,w}}
```

#### 3.7 Output Jacobian $`\mathbf{C}_c`$

Since most outputs are **direct state measurements**, $`\mathbf{C}_c`$
is a **sparse selection matrix**:

```math
\mathbf{C}_c
=
\begin{bmatrix}
  \mathbf{e}_{T_{melt,k}}^\top \\
  \mathbf{e}_{P_k}^\top \\
  \mathbf{e}_{h_{film}}^\top \\
  \mathbf{e}_{N_{IBC,l}}^\top \\
  \mathbf{e}_{T_{cool,l}}^\top \\
  \mathbf{e}_{v_{haul}}^\top \\
  \mathbf{e}_{R_{roll,w}}^\top \\
  \mathbf{e}_{\sigma_{web,w}}^\top \\
  \mathbf{e}_{L_{rem,w}}^\top
\end{bmatrix}
\in \mathbb{R}^{58 \times 146}
```

where $`\mathbf{e}_i`$ denotes the unit selection vector for
state $`i`$.

The film thickness output has a **nonlinear coupling** that
produces a non-trivial $`\mathbf{C}_c`$ row:

```math
\frac{\partial h_{total}}{\partial Q_k}
\bigg|_0
= \frac{\rho_{mix,k,0}}
       {2\pi R_0 \rho_{mix,0} v_{z,0}}
```

```math
\frac{\partial h_{total}}{\partial R}
\bigg|_0
= -\frac{\dot{m}_{total,0}}
        {2\pi R_0^2 \rho_{mix,0} v_{z,0}}
```

---

### 4. Continuous-Time Linearised System

Assembling all blocks, the **continuous-time linearised system** is:

```math
\boxed{
\dot{\tilde{\mathbf{x}}}
= \mathbf{A}_c\,\tilde{\mathbf{x}} +
\mathbf{B}_c\,\tilde{\mathbf{u}},
\qquad
\tilde{\mathbf{y}}
= \mathbf{C}_c\,\tilde{\mathbf{x}} +
\mathbf{D}_c\,\tilde{\mathbf{u}}
}
```

```math
\mathbf{A}_c \in \mathbb{R}^{146 \times 146}, \quad
\mathbf{B}_c \in \mathbb{R}^{146 \times 96}, \quad
\mathbf{C}_c \in \mathbb{R}^{58 \times 146}, \quad
\mathbf{D}_c \approx \mathbf{0}
```

#### 4.1 Stability of the Linearised System

The linearised system is **asymptotically stable** if and only if
all eigenvalues of $`\mathbf{A}_c`$ have strictly negative real parts:

```math
\text{Re}\bigl(\lambda_i(\mathbf{A}_c)\bigr) < 0,
\qquad \forall\, i = 1,\ldots,146
```

The dominant (slowest) eigenvalues correspond to the **thermal
states** of the barrel and die, with time constants:

```math
\tau_i = -\frac{1}{\text{Re}(\lambda_i)}
```

| Subsystem | Dominant $`\tau`$ (s) |
|-----------|-------------------|
| Roll build-up | 300 – 1800 |
| Barrel zone temperatures | 200 – 600 |
| Die zone temperatures | 100 – 400 |
| Melt temperature | 60 – 180 |
| Film thickness | 30 – 90 |
| Bubble radius | 20 – 60 |
| Bubble pressure | 10 – 30 |
| Web tension | 5 – 20 |
| Melt pressure | 1 – 5 |
| Drive torque | 0.1 – 1 |

#### 4.2 Controllability and Observability

**Controllability** (Kalman rank condition):

```math
\text{rank}\,\mathcal{C}
= \text{rank}
\begin{bmatrix}
  \mathbf{B}_c &
  \mathbf{A}_c\mathbf{B}_c &
  \mathbf{A}_c^2\mathbf{B}_c &
  \cdots &
  \mathbf{A}_c^{145}\mathbf{B}_c
\end{bmatrix}
= 146
```

**Observability** (Kalman rank condition):

```math
\text{rank}\,\mathcal{O}
= \text{rank}
\begin{bmatrix}
  \mathbf{C}_c \\
  \mathbf{C}_c\mathbf{A}_c \\
  \mathbf{C}_c\mathbf{A}_c^2 \\
  \vdots \\
  \mathbf{C}_c\mathbf{A}_c^{145}
\end{bmatrix}
= 146
```

> ⚠️ **Practical note:** Full-rank controllability and observability
> of the 146-state system are necessary but not sufficient for
> practical control design. The **Gramian-based analysis** used in
> balanced truncation provides a quantitative measure of how well
> each mode is excited and observed, which is more informative than
> the binary rank condition alone.

---

### 5. Discretisation to Discrete-Time State Space

#### 5.1 Zero-Order Hold (ZOH) Discretisation

The continuous-time system is converted to discrete time with
sampling period $`T_s = 3\,\text{s}`$ using the **matrix exponential**
(exact for piecewise-constant inputs):

```math
\mathbf{A}_d
= e^{\mathbf{A}_c T_s}
= \sum_{k=0}^{\infty}
  \frac{(\mathbf{A}_c T_s)^k}{k!}
```

```math
\mathbf{B}_d
= \mathbf{A}_c^{-1}
  \bigl(e^{\mathbf{A}_c T_s} - \mathbf{I}\bigr)
  \mathbf{B}_c
= \int_0^{T_s} e^{\mathbf{A}_c\tau}\,d\tau\;\mathbf{B}_c
```

```math
\mathbf{C}_d = \mathbf{C}_c,
\qquad
\mathbf{D}_d = \mathbf{D}_c
```

Computed efficiently via the **Van Loan method**:

```math
\exp\!\left(
  \begin{bmatrix}
    \mathbf{A}_c & \mathbf{B}_c \\
    \mathbf{0}   & \mathbf{0}
  \end{bmatrix}
  T_s
\right)
=
\begin{bmatrix}
  \mathbf{A}_d & \mathbf{B}_d \\
  \mathbf{0}   & \mathbf{I}
\end{bmatrix}
```

#### 5.2 Sampling Time Selection

The sampling time $`T_s = 3\,\text{s}`$ is chosen to satisfy the
**Shannon–Nyquist criterion** for the fastest controlled mode:

```math
T_s \leq \frac{\tau_{min}}{10}
= \frac{30\,\text{s}}{10}
= 3\,\text{s}
```

where $`\tau_{min} = 30\,\text{s}`$ is the fastest slow-subsystem
time constant (film thickness / bubble dynamics). The fast states
(drives, pressure, dosing speed) are eliminated by singular
perturbation before discretisation.

#### 5.3 Discrete-Time Stability

The discrete-time system is stable if and only if all eigenvalues
of $`\mathbf{A}_d`$ lie strictly inside the unit circle:

```math
|\lambda_i(\mathbf{A}_d)| < 1,
\qquad \forall\, i = 1,\ldots,146
```

The ZOH mapping preserves stability:

```math
\text{Re}(\lambda_i(\mathbf{A}_c)) < 0
\;\Longleftrightarrow\;
|\lambda_i(\mathbf{A}_d)| < 1
```

The resulting **full-order discrete-time linearised system** is:

```math
\boxed{
\tilde{\mathbf{x}}_{k+1}
= \mathbf{A}_d\,\tilde{\mathbf{x}}_k +
\mathbf{B}_d\,\tilde{\mathbf{u}}_k,
\qquad
\tilde{\mathbf{y}}_k
= \mathbf{C}_d\,\tilde{\mathbf{x}}_k +
\mathbf{D}_d\,\tilde{\mathbf{u}}_k
}
```

```math
\mathbf{A}_d \in \mathbb{R}^{146 \times 146}, \quad
\mathbf{B}_d \in \mathbb{R}^{146 \times 96}, \quad
\mathbf{C}_d \in \mathbb{R}^{58 \times 146}, \quad
\mathbf{D}_d \approx \mathbf{0}
```

---

### 6. Validity Region of the Linearisation

The linearised model is valid within a **neighbourhood of the
operating point** where the neglected higher-order terms remain
small. The validity region is characterised by:

```math
\|\tilde{\mathbf{x}}\| \leq \delta_x,
\qquad
\|\tilde{\mathbf{u}}\| \leq \delta_u
```

where $`\delta_x`$ and $`\delta_u`$ are determined by the dominant
nonlinearities in the system:

| Nonlinearity | Source | Validity Bound |
|--------------|--------|---------------|
| Arrhenius viscosity | $`\mu \propto e^{E_a/RT}`$ | $`\|\tilde{T}\| \leq 15\,°\text{C}`$ |
| Power-law shear thinning | $`\mu \propto \dot{\gamma}^{n-1}`$ | $`\|\tilde{\dot{\gamma}}\| \leq 20\%`$ |
| Bubble geometry | $`R(z,t)`$ free surface | $`\|\tilde{R}\| \leq 10\%\,R_0`$ |
| Film thickness | $`h \propto 1/(R\,v_z)`$ | $`\|\tilde{h}\| \leq 15\%\,h_0`$ |
| Roll build-up | $`\dot{R}_{roll} \propto 1/R_{roll}`$ | $`\|\tilde{R}_{roll}\| \leq 20\%\,R_0`$ |

> 💡 **Implication for MPC:** The MPC input/output constraints
> (configured via `MPCConfig.u_bound` and `MPCConfig.y_bound`)
> should be set to keep the system within this validity region.
> The default values of $`\pm 3\sigma`$ in normalised space
> are chosen to respect these bounds under typical operating
> conditions.

---

### 7. From Linearised Model to Data-Driven Identification

In practice, the analytical Jacobians derived above require
**precise knowledge** of all physical parameters
($`E_{a,k}`$, $`n_k`$, $`h_{conv}`$, $`UA_l`$, etc.), which are
**material- and grade-dependent** and difficult to measure directly.

The **N4SID subspace identification** algorithm used in this
pipeline bypasses this requirement by estimating the
$`(A_d, B_d, C_d, D_d)`$ matrices **directly from SCADA data**,
implicitly capturing the linearised dynamics at the operating
point encoded in the data.

The relationship between the analytical and identified models is:

```math
\underbrace{
  \hat{\mathbf{A}}_d,\,
  \hat{\mathbf{B}}_d,\,
  \hat{\mathbf{C}}_d,\,
  \hat{\mathbf{D}}_d
}_{\text{N4SID identified}}
\;\approx\;
\underbrace{
  \mathbf{A}_d,\,
  \mathbf{B}_d,\,
  \mathbf{C}_d,\,
  \mathbf{D}_d
}_{\text{analytical linearisation}} +
\underbrace{
  \boldsymbol{\Delta}_A,\,
  \boldsymbol{\Delta}_B,\,
  \boldsymbol{\Delta}_C,\,
  \boldsymbol{\Delta}_D
}_{\text{identification error}}
```

The identification error $`\boldsymbol{\Delta}`$ arises from:
- Finite data length and measurement noise
- Unmodelled nonlinearities within the operating region
- Grade-to-grade parameter variations
- Unobserved disturbances (material changes, ambient temperature)

The **parameter optimisation** step (L-BFGS-B refinement in
`system_identification.py`) minimises the Frobenius norm of
$`\boldsymbol{\Delta}`$ subject to the stability constraint
$`\rho(\hat{\mathbf{A}}_d) < 1`$.

---

### 8. Summary: Linearisation Pipeline

```
Nonlinear PDE/ODE System  f(x, u) = 0
            │
            │  Spatial discretisation
            │  (FDM for extruder, FEM for die,
            │   Galerkin for bubble)
            ▼
Nonlinear ODE System  ẋ = f(x, u)  [n = 146]
            │
            │  Newton-Raphson steady-state solve
            │  f(x₀, u₀) = 0
            ▼
Operating Point  (x₀, u₀, y₀)
            │
            │  First-order Taylor expansion
            │  Analytical Jacobians ∂f/∂x, ∂f/∂u
            ▼
Continuous-Time LTI  ẋ̃ = Ac x̃ + Bc ũ  [146 × 146]
            │
            │  ZOH discretisation
            │  Ts = 3 s (Van Loan method)
            ▼
Discrete-Time LTI  x̃_{k+1} = Ad x̃_k + Bd ũ_k  [146 × 146]
            │
            │  In practice: replaced by N4SID
            │  identification from SCADA data
            ▼
Identified Model  (Â_d, B̂_d, Ĉ_d, D̂_d)  [n_id × n_id]
            │
            │  Model order reduction pipeline
            │  (BT → POD → ZOH → Augmentation)
            ▼
MPC-Ready Reduced Model  [n_r ≈ 22 states]
```

---

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
├── extrusion.csv                    # Example real dataset (optional, not required)
├── extrusion_data_legend.xlsx       # SCADA tag reference for extrusion.csv
├── Co_Extrusion_Blown_Film_Line_Dynamics.pdf  # Background reading on the physical process
├── requirements.txt                 # Python dependencies
├── LICENSE                          # MIT License
├── README.md                        # This file
│
└── Code/
    ├── main.py                     # Entry point & pipeline orchestrator
    ├── config.py                   # All configuration, constants & column defs
    ├── data_manager.py              # Data loading, preprocessing & splitting
    ├── system_identification.py     # N4SID subspace identification
    ├── model_reduction.py           # Balanced truncation, POD, integrator augmentation
    ├── accuracy.py                  # Shared accuracy-gate primitives (R² checks, ModelAccuracyError)
    ├── estimation.py                # Kalman filter & state estimation
    ├── mpc_controller.py            # MPC design, QP solver & weight optimisation
    ├── simulation.py                # Closed-loop simulator & model validation
    ├── persistence.py               # Save/load cache for reduced model & tuned MPC weights
    ├── utils.py                     # Plotting, reporting & helper utilities
    │
    ├── saved/                       # Cached reduced model & tuned MPC weights (git-ignored)
    │   └── reduced_model.pkl
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

### `accuracy.py`
Shared accuracy-gate primitives used by `main.py` to enforce a minimum
worst-case per-output R² after identification and again after
reduction (see [Accuracy Gate](#-accuracy-gate)).

| Class / Function | Responsibility |
|-------------------|----------------|
| `AccuracyResult` | Per-output R² result with `worst_output`/`worst_r2`/`passed` properties and a `summary()` string |
| `ModelAccuracyError` | Raised once order escalation exhausts its configured ceiling without meeting the threshold |
| `evaluate_accuracy()` | Computes per-output R² and packages it into an `AccuracyResult` |

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
| `MPCController` | Builds the MPC QP **once** as a parametrised CVXPY problem and solves it every step by updating `cp.Parameter` values (state, reference, previous input) rather than rebuilding the problem from scratch — see [Performance Notes](#-performance-notes) |
| `MPCWeightOptimiser` | Nelder-Mead tuning of 4 group-level Q/R weights in log-space |

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

### `persistence.py`
Save/load cache for the two expensive-to-compute pipeline artefacts —
see [Model & Weight Caching](#-model--weight-caching).

| Class | Responsibility |
|-------|---------------|
| `MPCWeights` | Immutable (Q, R) weight snapshot |
| `ModelStore` | Save/load a `ReducedModel` pickle |
| `ControllerWeightsStore` | Save/load an `MPCWeights` pickle, with shape validation against the current model |

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
| `BlownFilmPipeline` | Runs all pipeline stages in order (loading cached model/weights where available), wires modules together |

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
# Synthetic data, all defaults (loads cached model/weights if present,
# otherwise builds them fresh and caches them — weight optimisation
# is off by default)
python main.py

# Real data
python main.py --data ../extrusion.csv

# Custom model orders and MPC horizons
python main.py --data ../extrusion.csv --n_id 20 --n_red 12 --Np 20 --Nc 8

# Run MPC weight optimisation (off by default) and cache the result
python main.py --optimise_weights

# Force a fresh model (ignore the cached one) and cache the new result
python main.py --optimise_model

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
| `--min_r2` | `0.95` | Required worst-case per-output R² (accuracy gate) |
| `--max_n_states` | `60` | Ceiling for automatic N4SID order escalation on gate failure |
| `--max_n_states_bt` | `50` | Ceiling for automatic reduction-order escalation on gate failure |
| `--no_accuracy_gate` | `False` | Disable the minimum-accuracy gate entirely (debugging only) |
| `--Ts` | `None` (auto) | Sampling time in seconds. Auto-detected from data timestamps for real data (falls back to 3.0s); explicit values override auto-detection |
| `--Np` | `20` | MPC prediction horizon |
| `--Nc` | `8` | MPC control horizon |
| `--T_sim` | `300` | Closed-loop simulation steps |
| `--no_param_opt` | `False` | Skip parameter optimisation after N4SID |
| `--optimise_weights` | `False` | Run MPC weight optimisation (Nelder-Mead), ignoring any cached weights, and cache the result |
| `--optimise_model` | `False` | Force fresh identification + reduction, ignoring any cached model, and cache the result |
| `--model_path` | `saved/reduced_model.pkl` | Path to load/save the cached reduced model |
| `--controller_path` | `saved/mpc_weights.pkl` | Path to load/save the cached MPC weights |
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
| `AccuracyConfig` | `min_r2`, `enabled`, `max_n_states`, `n_states_step`, `max_n_states_bt`, `n_states_bt_step` |
| `KalmanConfig` | `process_noise_scale`, `measurement_noise_scale` |
| `MPCConfig` | `prediction_horizon`, `control_horizon`, `u_bound`, `du_bound`, `y_bound`, `q_thickness_weight`, `q_temperature_weight`, `r_weight`, `optimise_weights`, `weight_opt_iterations` |
| `SimulationConfig` | `n_steps`, `noise_std`, `ref_step_time_1/2/3`, `ref_amplitude_1/2` |

`ProjectConfig` bundles all of the above and is what `main.py` builds
from the parsed CLI arguments before constructing `BlownFilmPipeline`.

---

## 🔄 Pipeline Stages

`BlownFilmPipeline.run()` executes eight stages in order:

1. **Data** — load real data or generate synthetic data, clean, scale, split (sampling time auto-detected from timestamps for real data)
2. **System Identification** — N4SID subspace identification (+ optional L-BFGS-B refinement); order escalates automatically if the accuracy gate fails
3. **Model Order Reduction** — balanced truncation → POD/Galerkin → integrator augmentation; reduced order escalates automatically if the accuracy gate fails
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

**Sampling time.** `Ts` is not assumed — it's derived from the `Datum`
column. `DataManager` computes the median interval between
consecutive timestamps (median rather than mean, so occasional data
gaps don't skew it) and the pipeline adopts that value automatically
unless `--Ts` is passed explicitly, in which case a mismatch only
logs a warning. This matters because a wrong `Ts` doesn't affect the
N4SID fit itself (each row is one discrete sample regardless), but it
does silently distort every physical-time interpretation downstream —
MPC prediction-horizon duration, ITAE scaling, and reported timings.

---

## ⚡ Performance Notes

`MPCWeightOptimiser.optimise()` used to be the slowest part of the
pipeline — a single run could solve tens of thousands of QPs. Three
changes brought this down without changing the underlying QP or the
weights it converges to:

1. **Stopped rebuilding the prediction matrices on every weight
   change.** `PredictionMatrices` (`F`, `G`) depend only on the plant
   model, never on `Q`/`R`, but the `Q`/`R` setters used to trigger a
   full rebuild anyway. They no longer do.
2. **Built the QP once instead of once per solve.** `MPCController`
   now constructs a single CVXPY `Problem` with `cp.Parameter` objects
   for the state estimate, reference and previous input — the three
   quantities that actually change on every `solve()` call. Only
   `Q`/`R` changes (via `set_weights()`) trigger a rebuild, which
   happens once per closed-loop rollout during weight tuning (and
   essentially never for the deployed controller) instead of once per
   time step. This also lets OSQP's warm start carry a real
   factorisation across steps, which was previously lost because a
   brand-new `Problem` was created on every call.
3. **Reduced the weight search space from `n_y + n_u` (48) to 4
   group-level weights** — one each for layer-thickness outputs,
   melt-temperature outputs, all other outputs, and a uniform
   input-rate weight — mirroring the grouping already used for the
   fixed default weights. This shrinks Nelder-Mead's initial simplex
   and the iterations needed to converge, independent of how many
   inputs/outputs the identified model has.

All three preserve the exact QP being solved (same cost, same
constraints, same solver/tolerances), so they only remove wasted
computation rather than trading off tuning accuracy.

---

## 💾 Model & Weight Caching

Identification/reduction and MPC weight optimisation are the two most
expensive stages, so their results are cached to disk and reused by
default instead of being recomputed on every run:

| Artefact | Cache file (default) | Behaviour |
|----------|-----------------------|-----------|
| Reduced model | `saved/reduced_model.pkl` | Loaded automatically if present. Use `--optimise_model` to force a fresh N4SID + reduction run and overwrite the cache. |
| MPC weights (`Q`, `R`) | `saved/mpc_weights.pkl` | Loaded automatically if present **and** `--optimise_weights` was not passed. Use `--optimise_weights` to force a fresh Nelder-Mead tune and overwrite the cache. |

Key points:

- **Weight optimisation is off by default.** A fresh run with no cache
  and no flags builds a new model (caching it) and an MPC controller
  with the fixed config weights from `config.py` (not cached, since
  they weren't tuned). Pass `--optimise_weights` whenever you want
  tuned weights, cached or not.
- Cached MPC weights are validated against the current model's
  `(n_y, n_u)` before being applied; a shape mismatch (e.g. after
  changing the output/input columns) is logged and the cache is
  ignored rather than silently misapplied.
- Only the tuned `Q`/`R` arrays are cached for the controller — not
  the live `MPCController` object itself, since it holds a CVXPY
  `Problem`/`Parameter` state that isn't meaningful to serialise. A
  fresh `MPCController` is always constructed from
  (`reduced_model`, `cfg`) and the cached weights are applied via
  `set_weights()`.
- Both cache paths are configurable via `--model_path` /
  `--controller_path`, and `saved/` is git-ignored.

---

## 🎯 Accuracy Gate

Every output must individually reach a minimum worst-case R² (default
`0.95`, `AccuracyConfig.min_r2`) before the pipeline is allowed to
proceed — checked twice:

| Stage | Checked against | On failure |
|-------|------------------|------------|
| Post-identification | Training data | Escalate `n_states` by `n_states_step` (default 5), up to `--max_n_states` (default 60) |
| Post-reduction | Held-out test data | Escalate `n_states_bt` by `n_states_bt_step` (default 2), up to `--max_n_states_bt` (default 50) |

If escalation exhausts its ceiling without meeting the threshold, a
`ModelAccuracyError` halts the pipeline with a diagnostic message
(worst output, achieved R², order reached) instead of silently
shipping an inaccurate model. A cached model (see above) is
re-validated against the same gate before being trusted — a cache
that no longer passes triggers a full rebuild.

Tune or disable via `--min_r2`, `--max_n_states`, `--max_n_states_bt`,
`--no_accuracy_gate`.

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
| Slow runs with weight optimisation | Weight optimisation is off by default. See [Performance Notes](#-performance-notes) for what already speeds `--optimise_weights` up when you do use it; otherwise reduce `--T_sim`, `--Np`, `--Nc` for faster smoke tests. |
| Stale/unexpected results after changing the model or data | The cached model/weights in `saved/` are reused by default — pass `--optimise_model` (and `--optimise_weights` if needed) to force a fresh build, or delete the `saved/` folder. |
| `Files above 50MB cannot be...` type errors when inspecting data | Large CSVs should be read by `pandas`/the pipeline itself, not by generic text-file tools. |

---

## 🤝 Contributing

Contributions are welcome. Please:

1. Follow the existing OOP structure and PEP 8 style.
2. Keep configuration values in `config.py` — avoid hardcoding constants.
3. Add/update tests or a smoke-test run (`python main.py --no_show --T_sim 5 --n_id 8 --n_red 5`) before submitting a PR.
4. Open an issue describing the change before large refactors.

---

## 📄 License

Distributed under the **MIT License**. See [LICENSE](LICENSE) for details.
