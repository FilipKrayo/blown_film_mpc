"""
model_reduction.py
==================
Model order reduction pipeline:
  1. Balanced truncation via discrete Gramians
  2. POD / Galerkin projection on state snapshots
  3. Integrator augmentation for offset-free MPC

Note: the model produced by ``SubspaceIdentifier`` is already
discrete-time (identified directly from sampled data at Ts), so no
further ZOH discretisation is performed here — both reduction stages
operate natively in the discrete-time domain.

Classes
-------
ReducedModel
    Container for the reduced discrete-time model plus the
    augmented matrices required by the MPC.
ModelReducer
    Orchestrates the full reduction pipeline.

Author : Blown Film MPC Project
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.linalg import cholesky, solve_discrete_lyapunov, svd

from config import ReductionConfig
from system_identification import StateSpaceModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reduced model container
# ---------------------------------------------------------------------------

@dataclass
class ReducedModel:
    """
    Holds the reduced discrete-time model and its MPC-ready
    augmented form.

    Attributes
    ----------
    A_d, B_d, C_d, D_d : reduced discrete-time matrices
    A_aug, B_aug, C_aug : augmented matrices (with integrators)
    hsv                 : Hankel singular values from balanced truncation
    Ts                  : sampling period
    """

    A_d: np.ndarray
    B_d: np.ndarray
    C_d: np.ndarray
    D_d: np.ndarray
    A_aug: np.ndarray
    B_aug: np.ndarray
    C_aug: np.ndarray
    hsv: np.ndarray
    Ts: float = 3.0

    @property
    def n_states(self) -> int:
        return self.A_d.shape[0]

    @property
    def n_inputs(self) -> int:
        return self.B_d.shape[1]

    @property
    def n_outputs(self) -> int:
        return self.C_d.shape[0]

    @property
    def n_aug(self) -> int:
        return self.A_aug.shape[0]

    def __repr__(self) -> str:
        return (
            f"ReducedModel("
            f"n_r={self.n_states}, n_aug={self.n_aug}, "
            f"n_u={self.n_inputs}, n_y={self.n_outputs}, "
            f"Ts={self.Ts}s)"
        )


# ---------------------------------------------------------------------------
# Model reducer
# ---------------------------------------------------------------------------

class ModelReducer:
    """
    Applies balanced truncation, POD/Galerkin projection, and
    integrator augmentation to a StateSpaceModel.

    Parameters
    ----------
    cfg : ReductionConfig
    Ts  : sampling period (seconds)
    """

    _GRAMIAN_REGULARISATION: float = 1e-8

    def __init__(
        self,
        cfg: ReductionConfig = ReductionConfig(),
        Ts: float = 3.0,
    ) -> None:
        self._cfg = cfg
        self._Ts  = Ts

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reduce(self, model: StateSpaceModel) -> ReducedModel:
        """
        Full reduction pipeline.

        Parameters
        ----------
        model : identified full-order StateSpaceModel

        Returns
        -------
        ReducedModel
        """
        logger.info("Starting model order reduction pipeline ...")

        A_s = self._stabilise(model.A)

        # Stage 1: balanced truncation
        A_r, B_r, C_r, D_r, hsv = self._balanced_truncation(
            A_s, model.B, model.C, model.D
        )

        # Stage 2: POD / Galerkin on state snapshots
        A_r, B_r, C_r = self._pod_galerkin(A_r, B_r, C_r, model.B.shape[1])

        # The input model is already discrete-time (identified directly
        # from sampled data by N4SID at Ts), and both balanced truncation
        # and POD/Galerkin above operate on the discrete-time matrices.
        # No further (ZOH) discretisation is required or correct here —
        # re-discretising via matrix-exponential would treat the already
        # discrete A_r as a continuous-time generator and destabilise it.
        A_d, B_d = A_r, B_r
        C_d, D_d = C_r.copy(), D_r.copy()

        # Stage 3: integrator augmentation
        A_aug, B_aug, C_aug = self._augment(A_d, B_d, C_d)

        reduced = ReducedModel(
            A_d=A_d, B_d=B_d, C_d=C_d, D_d=D_d,
            A_aug=A_aug, B_aug=B_aug, C_aug=C_aug,
            hsv=hsv, Ts=self._Ts,
        )
        logger.info("Reduction complete: %r", reduced)
        return reduced

    # ------------------------------------------------------------------
    # Private pipeline stages
    # ------------------------------------------------------------------

    def _stabilise(self, A: np.ndarray) -> np.ndarray:
        """Scale A so that spectral radius < 1."""
        rho = float(np.max(np.abs(np.linalg.eigvals(A))))
        if rho >= 1.0:
            logger.warning(
                "Unstable A detected (rho=%.4f) — rescaling.", rho
            )
            A = A / (rho + 0.05)
        return A

    def _balanced_truncation(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        D: np.ndarray,
    ):
        """
        Balanced truncation via discrete controllability and
        observability Gramians.

        Returns
        -------
        A_r, B_r, C_r, D_r, hsv
        """
        logger.info("Computing balanced truncation ...")
        reg = self._GRAMIAN_REGULARISATION * np.eye(A.shape[0])

        try:
            Wc = solve_discrete_lyapunov(A, B @ B.T)
            Wo = solve_discrete_lyapunov(A.T, C.T @ C)
        except Exception as exc:
            logger.warning("Lyapunov solve failed (%s) — using identity.", exc)
            Wc = np.eye(A.shape[0])
            Wo = np.eye(A.shape[0])

        Wc = (Wc + Wc.T) / 2 + reg
        Wo = (Wo + Wo.T) / 2 + reg

        Lc = cholesky(Wc, lower=True)
        Lo = cholesky(Wo, lower=True)

        M        = Lo.T @ Lc
        U_, S, V = svd(M, full_matrices=False)
        hsv      = S

        # Auto-select order
        r = self._cfg.n_states_bt
        r = min(r, A.shape[0] - 1)
        energy = np.cumsum(S) / (np.sum(S) + 1e-12)
        r_auto = int(np.searchsorted(energy, self._cfg.bt_energy_tolerance)) + 1
        r      = max(r, r_auto, 2)
        r      = min(r, A.shape[0] - 1)

        logger.info(
            "BT: retaining r=%d states (energy=%.2f%%)",
            r, energy[r - 1] * 100,
        )

        Sh  = np.diag(np.sqrt(S[:r]))
        Shi = np.diag(1.0 / np.sqrt(S[:r]))
        T_f = Lc @ V[:r, :].T @ Shi       # (n, r)  forward transform
        T_i = Shi @ U_[:, :r].T @ Lo.T    # (r, n)  inverse transform

        A_r = T_i @ A @ T_f
        B_r = T_i @ B
        C_r = C   @ T_f
        D_r = D.copy()

        return A_r, B_r, C_r, D_r, hsv

    def _pod_galerkin(
        self,
        A_r: np.ndarray,
        B_r: np.ndarray,
        C_r: np.ndarray,
        n_u: int,
    ):
        """
        Further reduce via POD on simulated state snapshots.
        """
        logger.info("Running POD/Galerkin projection ...")
        T_snap = 500
        X_snap = np.zeros((T_snap, A_r.shape[0]))
        x      = np.zeros(A_r.shape[0])
        rng    = np.random.default_rng(0)

        for t in range(T_snap):
            X_snap[t] = x
            u          = rng.standard_normal(n_u) * 0.1
            x          = A_r @ x + B_r @ u

        U_pod, S_pod, _ = svd(X_snap.T, full_matrices=False)
        energy = np.cumsum(S_pod ** 2) / (np.sum(S_pod ** 2) + 1e-12)
        r_pod  = int(np.searchsorted(energy, self._cfg.pod_energy_tolerance)) + 1
        r_pod  = max(r_pod, 2)
        r_pod  = min(r_pod, A_r.shape[0])

        logger.info(
            "POD: retaining r=%d modes (energy=%.2f%%)",
            r_pod, energy[r_pod - 1] * 100,
        )

        Phi = U_pod[:, :r_pod]
        return Phi.T @ A_r @ Phi, Phi.T @ B_r, C_r @ Phi

    @staticmethod
    def _augment(
        A_d: np.ndarray,
        B_d: np.ndarray,
        C_d: np.ndarray,
    ):
        """
        Augment with output-disturbance integrators for
        offset-free MPC tracking.

        Augmented state: z = [x_r; d]  where d ∈ R^{n_y}
        """
        n, m, p = A_d.shape[0], B_d.shape[1], C_d.shape[0]

        A_aug = np.block([
            [A_d,              np.zeros((n, p))],
            [np.zeros((p, n)), np.eye(p)],
        ])
        B_aug = np.vstack([B_d, np.zeros((p, m))])
        C_aug = np.hstack([C_d, np.eye(p)])

        logger.info(
            "Augmented model: A_aug%s, B_aug%s, C_aug%s",
            A_aug.shape, B_aug.shape, C_aug.shape,
        )
        return A_aug, B_aug, C_aug