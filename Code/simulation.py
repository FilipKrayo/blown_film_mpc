"""
simulation.py
=============
Closed-loop simulation and open-loop model validation.

Classes
-------
SimulationResult
    Immutable container for all closed-loop simulation outputs.
ModelValidator
    Computes validation metrics and generates diagnostic plots
    for the identified/reduced model.
ClosedLoopSimulator
    Simulates the MPC-controlled plant in closed loop.

Author : Blown Film MPC Project
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

from config import SimulationConfig
from estimation import KalmanFilter
from model_reduction import ReducedModel
from mpc_controller import MPCController, MPCResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ValidationMetrics:
    """Per-output validation metrics."""

    output_name: str
    mse: float
    r2: float
    nrmse: float

    def __repr__(self) -> str:
        return (
            f"ValidationMetrics({self.output_name}: "
            f"R²={self.r2:.4f}, NRMSE={self.nrmse:.4f})"
        )


@dataclass
class SimulationResult:
    """
    Container for all outputs of a closed-loop simulation run.

    Attributes
    ----------
    Y_measured  : plant outputs with noise (T, n_y)
    U_applied   : control inputs applied   (T, n_u)
    X_estimated : augmented state estimates (T, n_aug)
    references  : reference trajectories   (T, n_y)
    costs       : QP objective per step    (T,)
    statuses    : solver status per step   (T,)
    Ts          : sampling period
    """

    Y_measured: np.ndarray
    U_applied: np.ndarray
    X_estimated: np.ndarray
    references: np.ndarray
    costs: np.ndarray
    statuses: List[str]
    Ts: float = 3.0

    @property
    def n_steps(self) -> int:
        return len(self.Y_measured)

    @property
    def ise(self) -> float:
        """Integral of squared error."""
        return float(np.sum((self.Y_measured - self.references) ** 2) * self.Ts)

    @property
    def itae(self) -> float:
        """Integral of time-weighted absolute error."""
        t = np.arange(self.n_steps)[:, None]
        return float(np.sum(t * np.abs(self.Y_measured - self.references)) * self.Ts)

    @property
    def mean_mse(self) -> float:
        return float(np.mean((self.Y_measured - self.references) ** 2))


# ---------------------------------------------------------------------------
# Model validator
# ---------------------------------------------------------------------------

class ModelValidator:
    """
    Validates a reduced model on held-out test data.

    Parameters
    ----------
    reduced_model : ReducedModel
    output_names  : list of output signal names (for plots/reports)
    """

    def __init__(
        self,
        reduced_model: ReducedModel,
        output_names: Optional[List[str]] = None,
    ) -> None:
        self._model = reduced_model
        self._names = output_names or [
            f"y_{i}" for i in range(reduced_model.n_outputs)
        ]

    # ------------------------------------------------------------------
    def validate(
        self,
        U_test: np.ndarray,
        Y_test: np.ndarray,
    ) -> List[ValidationMetrics]:
        """
        Simulate the model on test data and compute metrics.

        Parameters
        ----------
        U_test : test inputs  (T, n_u)
        Y_test : test outputs (T, n_y)

        Returns
        -------
        List of ValidationMetrics, one per output channel.
        """
        Y_pred = self._simulate(U_test)
        n_out  = min(Y_test.shape[1], Y_pred.shape[1])
        metrics: List[ValidationMetrics] = []

        for i in range(n_out):
            yt   = Y_test[:, i]
            yp   = Y_pred[:, i]
            mse  = float(mean_squared_error(yt, yp))
            r2   = float(r2_score(yt, yp))
            nrmse = float(np.sqrt(mse) / (np.std(yt) + 1e-9))
            name  = self._names[i] if i < len(self._names) else f"y_{i}"
            metrics.append(ValidationMetrics(name, mse, r2, nrmse))

        avg_r2    = np.mean([m.r2    for m in metrics])
        avg_nrmse = np.mean([m.nrmse for m in metrics])
        logger.info(
            "Validation complete: avg R²=%.4f, avg NRMSE=%.4f",
            avg_r2, avg_nrmse,
        )
        return metrics

    # ------------------------------------------------------------------
    def predict(self, U: np.ndarray) -> np.ndarray:
        """Return predicted outputs for input sequence U."""
        return self._simulate(U)

    # ------------------------------------------------------------------
    def _simulate(self, U: np.ndarray) -> np.ndarray:
        T   = U.shape[0]
        n_y = self._model.n_outputs
        n   = self._model.n_states
        x   = np.zeros(n)
        Y   = np.zeros((T, n_y))
        for t in range(T):
            Y[t] = self._model.C_d @ x + self._model.D_d @ U[t]
            x    = self._model.A_d @ x + self._model.B_d @ U[t]
        return Y


# ---------------------------------------------------------------------------
# Closed-loop simulator
# ---------------------------------------------------------------------------

class ClosedLoopSimulator:
    """
    Simulates the MPC-controlled plant in closed loop.

    The plant is modelled by the reduced discrete-time model
    (A_d, B_d, C_d, D_d) with additive Gaussian noise.

    Parameters
    ----------
    controller    : MPCController
    kalman_filter : KalmanFilter (will be cloned internally)
    reduced_model : ReducedModel (plant model)
    cfg           : SimulationConfig
    """

    def __init__(
        self,
        controller: MPCController,
        kalman_filter: KalmanFilter,
        reduced_model: ReducedModel,
        cfg: SimulationConfig = SimulationConfig(),
    ) -> None:
        self._ctrl  = controller
        self._kf    = kalman_filter
        self._model = reduced_model
        self._cfg   = cfg

    # ------------------------------------------------------------------
    def run(
        self,
        y_ref: np.ndarray,
        x0: Optional[np.ndarray] = None,
    ) -> SimulationResult:
        """
        Execute the closed-loop simulation.

        Parameters
        ----------
        y_ref : reference — (n_y,) constant or (T, n_y) trajectory
        x0    : initial plant state (n_r,) — defaults to zeros

        Returns
        -------
        SimulationResult
        """
        T   = self._cfg.n_steps
        n_r = self._model.n_states
        n_u = self._model.n_inputs
        n_y = self._model.n_outputs

        x_plant = np.zeros(n_r) if x0 is None else x0.copy()
        kf      = self._kf.clone()
        kf.reset()
        u       = np.zeros(n_u)

        Y_meas  = np.zeros((T, n_y))
        U_app   = np.zeros((T, n_u))
        X_est   = np.zeros((T, self._model.n_aug))
        costs   = np.zeros(T)
        statuses: List[str] = []

        refs = self._build_reference(y_ref, T, n_y)
        noise = self._cfg.noise_std

        logger.info("Running closed-loop simulation for %d steps ...", T)
        for t in tqdm(range(T), desc="MPC Simulation", unit="step"):
            # Plant measurement
            rng    = np.random.default_rng(t)
            y_meas = (
                self._model.C_d @ x_plant
                + noise * rng.standard_normal(n_y)
            )
            Y_meas[t] = y_meas

            # State estimation
            x_est    = kf.update(y_meas, u)
            X_est[t] = x_est

            # MPC solve
            result   = self._ctrl.solve(x_est, refs[t], u)
            u        = result.u_opt
            U_app[t] = u
            costs[t] = result.cost
            statuses.append(result.status)

            # Plant update
            x_plant = (
                self._model.A_d @ x_plant
                + self._model.B_d @ u
                + noise * 0.1 * rng.standard_normal(n_r)
            )

        sim_result = SimulationResult(
            Y_measured=Y_meas,
            U_applied=U_app,
            X_estimated=X_est,
            references=refs,
            costs=costs,
            statuses=statuses,
            Ts=self._model.Ts,
        )
        logger.info(
            "Simulation complete: ISE=%.4f, ITAE=%.4f, mean_cost=%.4f",
            sim_result.ise, sim_result.itae, np.nanmean(costs),
        )
        return sim_result

    # ------------------------------------------------------------------
    @staticmethod
    def _build_reference(
        y_ref: np.ndarray,
        T: int,
        n_y: int,
    ) -> np.ndarray:
        """Expand reference to (T, n_y) array."""
        if y_ref.ndim == 1:
            return np.tile(y_ref, (T, 1))
        if y_ref.shape == (T, n_y):
            return y_ref
        raise ValueError(
            f"y_ref shape {y_ref.shape} incompatible with T={T}, n_y={n_y}."
        )