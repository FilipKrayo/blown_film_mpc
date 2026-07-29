"""
mpc_controller.py
=================
MPC controller, prediction matrix builder and weight optimiser.

Classes
-------
PredictionMatrices
    Immutable container for F and G matrices.
MPCResult
    Immutable container for a single QP solve result.
MPCController
    Builds and solves the MPC QP at each time step using CVXPY.
MPCWeightOptimiser
    Tunes Q and R diagonal weights via Nelder-Mead on a
    closed-loop simulation cost.

Author : Blown Film MPC Project
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cvxpy as cp
import numpy as np
from scipy.optimize import minimize

from config import MPCConfig
from estimation import KalmanFilter
from model_reduction import ReducedModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PredictionMatrices:
    """
    Free-response (F) and forced-response (G) matrices for the
    MPC prediction model.

        Y = F * x_aug + G * DeltaU
    """

    F: np.ndarray   # (Np*n_y, n_aug)
    G: np.ndarray   # (Np*n_y, Nc*n_u)
    Np: int
    Nc: int
    n_y: int
    n_u: int


@dataclass(frozen=True)
class MPCResult:
    """Result of a single MPC QP solve."""

    u_opt: np.ndarray       # optimal input to apply (n_u,)
    du_opt: np.ndarray      # optimal input increment (n_u,)
    Y_pred: np.ndarray      # predicted outputs (Np, n_y)
    cost: float
    status: str
    feasible: bool


# ---------------------------------------------------------------------------
# MPC controller
# ---------------------------------------------------------------------------

class MPCController:
    """
    Linear MPC with CVXPY / OSQP backend.

    The controller solves at each step k:

        min   ||Y - Y_ref||_Q^2 + ||DeltaU||_R^2
        s.t.  x_{j+1} = A_aug x_j + B_aug u_j
              u_min  <= u_j  <= u_max
              du_min <= du_j <= du_max
              y_min  <= y_j  <= y_max  (soft, via slack)

    Parameters
    ----------
    reduced_model : ReducedModel containing augmented matrices
    cfg           : MPCConfig
    """

    def __init__(
        self,
        reduced_model: ReducedModel,
        cfg: MPCConfig = MPCConfig(),
    ) -> None:
        self._A   = reduced_model.A_aug
        self._B   = reduced_model.B_aug
        self._C   = reduced_model.C_aug
        self._cfg = cfg

        self._n_aug = self._A.shape[0]
        self._n_u   = self._B.shape[1]
        self._n_y   = self._C.shape[0]

        # Weights (mutable for optimisation)
        self._Q = self._build_Q_matrix()
        self._R = cfg.r_weight * np.eye(self._n_u)

        # Constraints
        self._u_min  = -cfg.u_bound  * np.ones(self._n_u)
        self._u_max  =  cfg.u_bound  * np.ones(self._n_u)
        self._du_min = -cfg.du_bound * np.ones(self._n_u)
        self._du_max =  cfg.du_bound * np.ones(self._n_u)
        self._y_min  = -cfg.y_bound  * np.ones(self._n_y)
        self._y_max  =  cfg.y_bound  * np.ones(self._n_y)

        # Build prediction matrices
        self._pred = self._build_prediction_matrices()

        logger.info(
            "MPCController ready: Np=%d, Nc=%d, n_aug=%d, n_u=%d, n_y=%d",
            cfg.prediction_horizon, cfg.control_horizon,
            self._n_aug, self._n_u, self._n_y,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def Q(self) -> np.ndarray:
        return self._Q

    @Q.setter
    def Q(self, value: np.ndarray) -> None:
        self._Q = value
        self._pred = self._build_prediction_matrices()

    @property
    def R(self) -> np.ndarray:
        return self._R

    @R.setter
    def R(self, value: np.ndarray) -> None:
        self._R = value
        self._pred = self._build_prediction_matrices()

    @property
    def prediction_matrices(self) -> PredictionMatrices:
        return self._pred

    # ------------------------------------------------------------------
    def solve(
        self,
        x_aug: np.ndarray,
        y_ref: np.ndarray,
        u_prev: np.ndarray,
    ) -> MPCResult:
        """
        Solve the MPC QP for the current time step.

        Parameters
        ----------
        x_aug  : augmented state estimate (n_aug,)
        y_ref  : reference — (n_y,) constant or (Np, n_y) trajectory
        u_prev : previous applied input (n_u,)

        Returns
        -------
        MPCResult
        """
        Np, Nc = self._cfg.prediction_horizon, self._cfg.control_horizon
        n_u, n_y = self._n_u, self._n_y

        Y_ref = (
            np.tile(y_ref, Np)
            if y_ref.ndim == 1
            else y_ref.ravel()
        )

        Y_free = self._pred.F @ x_aug

        # Decision variables
        DU    = cp.Variable(Nc * n_u)
        slack = cp.Variable(Np * n_y, nonneg=True)

        Y_pred = Y_free + self._pred.G @ DU

        # Cost
        Q_big  = np.kron(np.eye(Np), self._Q)
        R_big  = np.kron(np.eye(Nc), self._R)
        cost   = (
            cp.quad_form(Y_pred - Y_ref, Q_big)
            + cp.quad_form(DU, R_big)
            + 1e3 * cp.sum(slack)
        )

        # Constraints
        constraints = self._build_constraints(DU, slack, Y_pred, u_prev)

        prob = cp.Problem(cp.Minimize(cost), constraints)
        self._solve_problem(prob)

        feasible = DU.value is not None
        if not feasible:
            logger.warning("MPC QP infeasible — applying zero increment.")
            du_opt = np.zeros(n_u)
        else:
            du_opt = np.asarray(DU.value[:n_u])

        u_opt   = np.clip(u_prev + du_opt, self._u_min, self._u_max)
        du_full = (
            np.asarray(DU.value)
            if feasible
            else np.zeros(Nc * n_u)
        )
        Y_pred_arr = (
            (Y_free + self._pred.G @ du_full).reshape(Np, n_y)
        )

        return MPCResult(
            u_opt=u_opt,
            du_opt=du_opt,
            Y_pred=Y_pred_arr,
            cost=float(prob.value) if prob.value is not None else np.inf,
            status=str(prob.status),
            feasible=feasible,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_Q_matrix(self) -> np.ndarray:
        """Construct diagonal Q with priority weights."""
        cfg    = self._cfg
        q_diag = np.full(self._n_y, cfg.q_default_weight)
        # Layer thickness outputs (first 3)
        q_diag[: min(3, self._n_y)] = cfg.q_thickness_weight
        # Melt temperature outputs (next 3)
        if self._n_y > 5:
            q_diag[3:6] = cfg.q_temperature_weight
        return np.diag(q_diag)

    def _build_prediction_matrices(self) -> PredictionMatrices:
        """Build F and G for the prediction model."""
        A, B, C = self._A, self._B, self._C
        Np, Nc  = self._cfg.prediction_horizon, self._cfg.control_horizon
        n_aug, n_u, n_y = self._n_aug, self._n_u, self._n_y

        F = np.zeros((Np * n_y, n_aug))
        G = np.zeros((Np * n_y, Nc * n_u))

        A_pow = np.eye(n_aug)
        for j in range(Np):
            A_pow = A @ A_pow
            F[j * n_y : (j + 1) * n_y, :] = C @ A_pow

        for j in range(Np):
            for l in range(min(j + 1, Nc)):
                A_jl = np.linalg.matrix_power(A, j - l)
                G[j * n_y : (j + 1) * n_y,
                  l * n_u : (l + 1) * n_u] = C @ A_jl @ B

        return PredictionMatrices(F=F, G=G, Np=Np, Nc=Nc, n_y=n_y, n_u=n_u)

    def _build_constraints(
        self,
        DU: cp.Variable,
        slack: cp.Variable,
        Y_pred: cp.Expression,
        u_prev: np.ndarray,
    ) -> List:
        """Assemble all CVXPY constraint objects."""
        Nc, n_u, n_y = (
            self._cfg.control_horizon, self._n_u, self._n_y
        )
        Np = self._cfg.prediction_horizon
        constraints = []

        # Reconstruct absolute inputs and enforce bounds
        U_seq = cp.Variable(Nc * n_u)
        for l in range(Nc):
            u_l = (
                u_prev + DU[:n_u]
                if l == 0
                else U_seq[(l - 1) * n_u : l * n_u] + DU[l * n_u : (l + 1) * n_u]
            )
            constraints += [
                U_seq[l * n_u : (l + 1) * n_u] == u_l,
                U_seq[l * n_u : (l + 1) * n_u] >= self._u_min,
                U_seq[l * n_u : (l + 1) * n_u] <= self._u_max,
                DU[l * n_u : (l + 1) * n_u]    >= self._du_min,
                DU[l * n_u : (l + 1) * n_u]    <= self._du_max,
            ]

        # Soft output constraints
        for j in range(Np):
            sl = slack[j * n_y : (j + 1) * n_y]
            yj = Y_pred[j * n_y : (j + 1) * n_y]
            constraints += [
                yj >= self._y_min - sl,
                yj <= self._y_max + sl,
            ]

        return constraints

    @staticmethod
    def _solve_problem(prob: cp.Problem) -> None:
        """Try OSQP first, fall back to ECOS."""
        try:
            prob.solve(
                solver=cp.OSQP,
                warm_start=True,
                verbose=False,
                max_iter=10_000,
                eps_abs=1e-5,
                eps_rel=1e-5,
            )
        except Exception as exc:
            logger.warning("OSQP failed (%s) — retrying with ECOS.", exc)
            prob.solve(solver=cp.ECOS, verbose=False)


# ---------------------------------------------------------------------------
# Weight optimiser
# ---------------------------------------------------------------------------

class MPCWeightOptimiser:
    """
    Optimises MPC Q and R diagonal weights using Nelder-Mead
    minimisation of a closed-loop simulation cost.

    The cost is:
        J = MSE(Y_cl, Y_ref) + lambda * mean(u^2)

    Weights are parameterised in log-space to ensure positivity.

    Parameters
    ----------
    controller    : MPCController instance to tune
    kalman_filter : KalmanFilter for state estimation
    U_val, Y_val  : validation data for closed-loop simulation
    cfg           : MPCConfig
    """

    def __init__(
        self,
        controller: MPCController,
        kalman_filter: KalmanFilter,
        U_val: np.ndarray,
        Y_val: np.ndarray,
        cfg: MPCConfig = MPCConfig(),
    ) -> None:
        self._ctrl = controller
        self._kf   = kalman_filter
        self._U    = U_val
        self._Y    = Y_val
        self._cfg  = cfg
        self._history: List[float] = []

    # ------------------------------------------------------------------
    @property
    def cost_history(self) -> List[float]:
        return list(self._history)

    # ------------------------------------------------------------------
    def optimise(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Nelder-Mead optimisation.

        Returns
        -------
        Q_opt : optimised Q matrix (n_y × n_y diagonal)
        R_opt : optimised R matrix (n_u × n_u diagonal)
        """
        n_y = self._ctrl._n_y
        n_u = self._ctrl._n_u
        x0  = np.zeros(n_y + n_u)   # log(1) = 0

        logger.info(
            "Optimising MPC weights (Nelder-Mead, %d iterations) ...",
            self._cfg.weight_opt_iterations,
        )
        res = minimize(
            self._closed_loop_cost,
            x0,
            method="Nelder-Mead",
            options={
                "maxiter": self._cfg.weight_opt_iterations,
                "xatol": 1e-3,
                "fatol": 1e-4,
                "disp": False,
            },
        )

        params = np.exp(res.x)
        Q_opt  = np.diag(params[:n_y])
        R_opt  = np.diag(params[n_y:])

        logger.info(
            "Weight optimisation complete: cost=%.6f | success=%s",
            res.fun, res.success,
        )
        return Q_opt, R_opt

    # ------------------------------------------------------------------
    def _closed_loop_cost(self, log_params: np.ndarray) -> float:
        """Evaluate closed-loop cost for given log-scale parameters."""
        params = np.exp(log_params)
        n_y    = self._ctrl._n_y
        n_u    = self._ctrl._n_u

        # Temporarily update weights
        self._ctrl.Q = np.diag(params[:n_y])
        self._ctrl.R = np.diag(params[n_y:])

        T   = min(200, len(self._U))
        kf  = self._kf.clone()
        kf.reset()
        u   = np.zeros(n_u)
        Y_cl = np.zeros((T, n_y))
        y_ref = np.zeros(n_y)

        for t in range(T):
            x   = kf.update(self._Y[t], u)
            res = self._ctrl.solve(x, y_ref, u)
            u   = res.u_opt
            Y_cl[t] = self._Y[t]

        tracking = float(np.mean((Y_cl - y_ref) ** 2))
        effort   = self._cfg.weight_opt_lambda * float(np.mean(u ** 2))
        cost     = tracking + effort
        self._history.append(cost)
        return cost