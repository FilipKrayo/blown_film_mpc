"""
system_identification.py
========================
N4SID subspace system identification and optional parameter
refinement via gradient-based optimisation.

Classes
-------
StateSpaceModel
    Lightweight container for (A, B, C, D) matrices with
    simulation and stability utilities.
SubspaceIdentifier
    Implements the N4SID algorithm to identify a discrete-time
    LTI model from input-output data.
ParameterOptimiser
    Refines (A, B, C, D) by minimising one-step-ahead prediction
    error subject to a stability penalty.

Author : Blown Film MPC Project
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.linalg import svd
from scipy.optimize import minimize

from config import IdentificationConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State space model container
# ---------------------------------------------------------------------------

@dataclass
class StateSpaceModel:
    """
    Container for a discrete-time LTI state space model.

        x_{k+1} = A x_k + B u_k
        y_k     = C x_k + D u_k

    Attributes
    ----------
    A, B, C, D : system matrices
    Ts         : sampling period (seconds)
    """

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    Ts: float = 3.0

    # ------------------------------------------------------------------
    @property
    def n_states(self) -> int:
        return self.A.shape[0]

    @property
    def n_inputs(self) -> int:
        return self.B.shape[1]

    @property
    def n_outputs(self) -> int:
        return self.C.shape[0]

    @property
    def spectral_radius(self) -> float:
        return float(np.max(np.abs(np.linalg.eigvals(self.A))))

    @property
    def is_stable(self) -> bool:
        return self.spectral_radius < 1.0

    # ------------------------------------------------------------------
    def simulate(
        self,
        U: np.ndarray,
        x0: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Simulate the model forward in time.

        Parameters
        ----------
        U  : input sequence (T, n_u)
        x0 : initial state  (n,) — defaults to zeros

        Returns
        -------
        Y_hat : (T, n_y)
        """
        T   = U.shape[0]
        x   = np.zeros(self.n_states) if x0 is None else x0.copy()
        Y   = np.zeros((T, self.n_outputs))
        for t in range(T):
            Y[t] = self.C @ x + self.D @ U[t]
            x    = self.A @ x + self.B @ U[t]
        return Y

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"StateSpaceModel("
            f"n={self.n_states}, m={self.n_inputs}, p={self.n_outputs}, "
            f"Ts={self.Ts}s, stable={self.is_stable}, "
            f"rho={self.spectral_radius:.4f})"
        )


# ---------------------------------------------------------------------------
# N4SID subspace identifier
# ---------------------------------------------------------------------------

class SubspaceIdentifier:
    """
    N4SID-style subspace system identification.

    Identifies a discrete-time LTI model from input-output data
    by projecting future outputs onto the past data space and
    performing an SVD-based state-space extraction.

    Parameters
    ----------
    cfg : IdentificationConfig
    Ts  : sampling period in seconds
    """

    def __init__(
        self,
        cfg: IdentificationConfig = IdentificationConfig(),
        Ts: float = 3.0,
    ) -> None:
        self._cfg = cfg
        self._Ts  = Ts
        self._sv: Optional[np.ndarray] = None   # Hankel singular values
        self._model: Optional[StateSpaceModel] = None

    # ------------------------------------------------------------------
    @property
    def model(self) -> StateSpaceModel:
        if self._model is None:
            raise RuntimeError("Call fit() before accessing the model.")
        return self._model

    @property
    def singular_values(self) -> np.ndarray:
        if self._sv is None:
            raise RuntimeError("Call fit() before accessing singular values.")
        return self._sv

    # ------------------------------------------------------------------
    def fit(self, U: np.ndarray, Y: np.ndarray) -> "SubspaceIdentifier":
        """
        Fit the N4SID model.

        Parameters
        ----------
        U : input data  (T, n_u)
        Y : output data (T, n_y)

        Returns
        -------
        self (for method chaining)
        """
        if U.shape[0] != Y.shape[0]:
            raise ValueError("U and Y must have the same number of rows.")

        logger.info("Running N4SID identification (n=%d, i=%d) ...",
                    self._cfg.n_states, self._cfg.n_block_rows)

        T, n_u = U.shape
        n_y    = Y.shape[1]
        i      = min(self._cfg.n_block_rows, T // 20)
        n      = self._cfg.n_states

        U_H = self._build_hankel(U, 2 * i)
        Y_H = self._build_hankel(Y, 2 * i)

        U_p = U_H[:i * n_u, :]
        U_f = U_H[i * n_u:, :]
        Y_p = Y_H[:i * n_y, :]
        Y_f = Y_H[i * n_y:, :]

        # Oblique projection
        W_p = np.vstack([U_p, Y_p])
        LQ  = np.linalg.lstsq(
            np.vstack([W_p, U_f]).T, Y_f.T, rcond=None
        )[0].T
        O_i = LQ[:, : W_p.shape[0]] @ W_p

        # SVD for order selection
        U_svd, S, _ = svd(O_i, full_matrices=False)
        self._sv     = S

        # Truncate to model order
        n   = min(n, len(S))
        U1  = U_svd[:, :n]
        Obs = U1 @ np.diag(np.sqrt(S[:n]))   # observability matrix

        # Extract C and A
        C_hat = Obs[:n_y, :]
        A_hat = np.linalg.lstsq(Obs[:-n_y, :], Obs[n_y:, :], rcond=None)[0]

        # Recover state sequence
        X_seq = np.linalg.lstsq(Obs, Y_f[: i * n_y, :], rcond=None)[0]

        # Solve for B, D
        B_hat, D_hat = self._solve_BD(
            A_hat, C_hat, X_seq, U, Y, i, n, n_u, n_y
        )

        self._model = StateSpaceModel(
            A=A_hat, B=B_hat, C=C_hat, D=D_hat, Ts=self._Ts
        )
        logger.info("N4SID complete: %r", self._model)
        return self

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_hankel(data: np.ndarray, rows: int) -> np.ndarray:
        """Construct a block-Hankel matrix from (T, m) data."""
        T, m = data.shape
        cols = T - rows + 1
        H    = np.zeros((rows * m, cols))
        for r in range(rows):
            H[r * m : (r + 1) * m, :] = data[r : r + cols].T
        return H

    @staticmethod
    def _solve_BD(
        A: np.ndarray,
        C: np.ndarray,
        X_seq: np.ndarray,
        U: np.ndarray,
        Y: np.ndarray,
        i: int,
        n: int,
        n_u: int,
        n_y: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Least-squares solve for B and D matrices."""
        n_steps = min(X_seq.shape[1] - 1, U.shape[0] - 2 * i)
        idx     = np.arange(2 * i, 2 * i + n_steps)

        Y_reg = Y[idx, :]
        U_reg = U[idx, :]
        X_reg = X_seq[:, :n_steps].T
        X_nxt = X_seq[:, 1 : n_steps + 1].T

        # B from: x_{t+1} - A x_t = B u_t
        residual = X_nxt - X_reg @ A.T
        B_hat    = np.linalg.lstsq(U_reg, residual, rcond=None)[0].T

        # D from: y_t - C x_t = D u_t
        residual_y = Y_reg - X_reg @ C.T
        D_hat      = np.linalg.lstsq(U_reg, residual_y, rcond=None)[0].T

        return B_hat, D_hat


# ---------------------------------------------------------------------------
# Parameter optimiser
# ---------------------------------------------------------------------------

class ParameterOptimiser:
    """
    Refines a StateSpaceModel by minimising one-step-ahead
    prediction MSE with a stability penalty via L-BFGS-B.

    Parameters
    ----------
    model : initial StateSpaceModel (starting point)
    cfg   : IdentificationConfig
    """

    _STABILITY_PENALTY: float = 1e4

    def __init__(
        self,
        model: StateSpaceModel,
        cfg: IdentificationConfig = IdentificationConfig(),
    ) -> None:
        self._model = model
        self._cfg   = cfg

    # ------------------------------------------------------------------
    def optimise(
        self,
        U: np.ndarray,
        Y: np.ndarray,
    ) -> StateSpaceModel:
        """
        Run optimisation and return the refined StateSpaceModel.

        Parameters
        ----------
        U : input data  (T, n_u)
        Y : output data (T, n_y)
        """
        n_samples = min(self._cfg.optimisation_samples, len(U))
        U_opt, Y_opt = U[:n_samples], Y[:n_samples]

        theta0 = self._pack(
            self._model.A, self._model.B,
            self._model.C, self._model.D
        )
        # Small perturbation to escape local minima
        rng    = np.random.default_rng(0)
        theta0 = theta0 + 0.01 * rng.standard_normal(theta0.shape)

        logger.info(
            "Optimising parameters via %s (max_iter=%d, samples=%d) ...",
            self._cfg.optimisation_method,
            self._cfg.optimisation_max_iter,
            n_samples,
        )
        t0  = time.perf_counter()
        res = minimize(
            self._cost,
            theta0,
            args=(U_opt, Y_opt),
            method=self._cfg.optimisation_method,
            options={
                "maxiter": self._cfg.optimisation_max_iter,
                "ftol": 1e-9,
                "gtol": 1e-7,
            },
        )
        elapsed = time.perf_counter() - t0
        logger.info(
            "Optimisation finished in %.1fs | cost=%.6f | success=%s",
            elapsed, res.fun, res.success,
        )

        A, B, C, D = self._unpack(res.x)
        return StateSpaceModel(A=A, B=B, C=C, D=D, Ts=self._model.Ts)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _pack(self, A, B, C, D) -> np.ndarray:
        return np.concatenate([A.ravel(), B.ravel(), C.ravel(), D.ravel()])

    def _unpack(
        self, theta: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n, m, p = self._model.n_states, self._model.n_inputs, self._model.n_outputs
        idx = 0
        A = theta[idx : idx + n * n].reshape(n, n); idx += n * n
        B = theta[idx : idx + n * m].reshape(n, m); idx += n * m
        C = theta[idx : idx + p * n].reshape(p, n); idx += p * n
        D = theta[idx : idx + p * m].reshape(p, m)
        return A, B, C, D

    def _cost(self, theta: np.ndarray, U: np.ndarray, Y: np.ndarray) -> float:
        A, B, C, D = self._unpack(theta)
        rho        = float(np.max(np.abs(np.linalg.eigvals(A))))
        stab_pen   = self._STABILITY_PENALTY * max(0.0, rho - 0.99) ** 2

        tmp_model = StateSpaceModel(
            A=A, B=B, C=C, D=D, Ts=self._model.Ts
        )
        Y_hat = tmp_model.simulate(U)
        mse   = float(np.mean((Y - Y_hat) ** 2))
        return mse + stab_pen