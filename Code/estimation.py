"""
estimation.py
=============
Kalman filter for augmented state estimation.

Classes
-------
KalmanFilter
    Discrete-time Kalman filter operating on the augmented
    state z = [x_r; d] produced by ModelReducer.

Author : Blown Film MPC Project
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Optional

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from config import KalmanConfig
from model_reduction import ReducedModel

logger = logging.getLogger(__name__)


class KalmanFilter:
    """
    Steady-state discrete Kalman filter for the augmented model.

    The filter estimates the augmented state z = [x_r; d] where
    d represents slowly varying output disturbances that enable
    offset-free MPC tracking.

    Parameters
    ----------
    reduced_model : ReducedModel from the reduction pipeline
    cfg           : KalmanConfig
    """

    def __init__(
        self,
        reduced_model: ReducedModel,
        cfg: KalmanConfig = KalmanConfig(),
    ) -> None:
        self._A   = reduced_model.A_aug.copy()
        self._B   = reduced_model.B_aug.copy()
        self._C   = reduced_model.C_aug.copy()
        self._cfg = cfg

        n_aug = self._A.shape[0]
        n_y   = self._C.shape[0]

        self._Q: np.ndarray = cfg.process_noise_scale * np.eye(n_aug)
        self._R: np.ndarray = cfg.measurement_noise_scale * np.eye(n_y)

        # State and covariance
        self._x: np.ndarray = np.zeros(n_aug)
        self._P: np.ndarray = np.eye(n_aug)
        self._Kf: Optional[np.ndarray] = None

        self._initialised: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def gain(self) -> np.ndarray:
        if self._Kf is None:
            raise RuntimeError("Call initialise() before accessing gain.")
        return self._Kf

    @property
    def state(self) -> np.ndarray:
        return self._x.copy()

    def initialise(self) -> "KalmanFilter":
        """
        Compute the steady-state Kalman gain by solving the DARE
        via iterated Lyapunov equations.

        Returns
        -------
        self (for method chaining)
        """
        logger.info("Initialising Kalman filter ...")
        self._Kf = self._compute_steady_state_gain()
        self._initialised = True
        logger.info("Kalman gain computed: Kf%s", self._Kf.shape)
        return self

    def reset(self, x0: Optional[np.ndarray] = None) -> None:
        """Reset state estimate to x0 (or zeros)."""
        n = self._A.shape[0]
        self._x = np.zeros(n) if x0 is None else x0.copy()
        self._P = np.eye(n)

    def update(self, y: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        One predict-update cycle.

        Parameters
        ----------
        y : measurement vector (n_y,)
        u : applied input      (n_u,)

        Returns
        -------
        x_est : updated state estimate (n_aug,)
        """
        if not self._initialised:
            raise RuntimeError("Call initialise() before update().")

        # Prediction step
        x_pred = self._A @ self._x + self._B @ u
        P_pred = self._A @ self._P @ self._A.T + self._Q

        # Innovation
        innov = y - self._C @ x_pred
        S     = self._C @ P_pred @ self._C.T + self._R

        # Kalman gain (time-varying for robustness)
        Kf = P_pred @ self._C.T @ np.linalg.inv(S)

        # Update
        self._x = x_pred + Kf @ innov
        I_KC    = np.eye(len(self._x)) - Kf @ self._C
        self._P = I_KC @ P_pred

        return self._x.copy()

    def batch_filter(
        self,
        U: np.ndarray,
        Y: np.ndarray,
    ) -> np.ndarray:
        """
        Run the filter over an entire dataset.

        Parameters
        ----------
        U : inputs  (T, n_u)
        Y : outputs (T, n_y)

        Returns
        -------
        X_est : estimated states (T, n_aug)
        """
        T     = U.shape[0]
        n_aug = self._A.shape[0]
        X_est = np.zeros((T, n_aug))
        self.reset()
        for t in range(T):
            X_est[t] = self.update(Y[t], U[t])
        return X_est

    def clone(self) -> "KalmanFilter":
        """Return a deep copy (useful for closed-loop simulation)."""
        return deepcopy(self)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_steady_state_gain(self) -> np.ndarray:
        """
        Approximate steady-state gain via one Lyapunov iteration.
        Falls back to a scaled identity gain if the solve fails.
        """
        try:
            P = solve_discrete_lyapunov(
                self._A,
                self._Q + self._A @ self._P @ self._A.T,
            )
            S  = self._C @ P @ self._C.T + self._R
            Kf = P @ self._C.T @ np.linalg.inv(S)
            self._P = P
            return Kf
        except Exception as exc:
            logger.warning(
                "Steady-state Kalman gain computation failed (%s). "
                "Using fallback gain.",
                exc,
            )
            n_aug = self._A.shape[0]
            n_y   = self._C.shape[0]
            return 0.1 * np.ones((n_aug, n_y))