"""
grey_box.py
===========
Grey-box system identification: instead of a black-box N4SID fit,
optimise a small set of PHYSICAL parameter group scale factors of a
``physical_model.FirstPrinciplesModel`` so that its linearised,
discretised model reproduces measured input-output data. This makes
the identified model's structure (sparsity, subsystem coupling)
inherited directly from the first-principles physics rather than an
arbitrary state-space realisation.

Mirrors the existing ``MPCWeightOptimiser`` pattern already used in
this codebase (mpc_controller.py): optimise a handful of group-level
scale factors rather than every individual constant, keeping the
search space small and tractable.

Classes
-------
GreyBoxIdentifier
    Fits ``PARAM_GROUPS`` scale factors via bounded L-BFGS-B, then
    returns a ``system_identification.StateSpaceModel``.

Author : Blown Film MPC Project
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from config import INPUT_COLS, PhysicalModelConfig
from physical_model import (
    FirstPrinciplesModel,
    PhysicalParameters,
    eliminate_fast_states,
    stabilise_discrete_matrix,
)
from system_identification import StateSpaceModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Real-data input mapping
# ---------------------------------------------------------------------------

# Index of each real INPUT_COLS entry that has a direct FirstPrinciplesModel
# input counterpart. Entries with no physical-model equivalent (thickness
# setpoints — the model has no direct SDickeSoll input) are left unmapped;
# those model inputs are held at their nominal value instead.
_REAL_ZONE_SETPOINT_IDX = list(range(0, 12))     # 3 extruders x 4 zones
_REAL_IBC_SETPOINT_IDX = list(range(15, 18))
_REAL_COOLING_SETPOINT_IDX = list(range(18, 21))
_REAL_HAULOFF_SETPOINT_IDX = [21]
_REAL_WINDER_SPEED_IDX = [22, 24]                # ST113/ST114 VARWdSpSpeed
_REAL_WINDER_TENS_IDX = [23, 25]                 # ST113/ST114 VARWdSpTens


def map_real_inputs_to_model(U_real: np.ndarray, model: FirstPrinciplesModel) -> np.ndarray:
    """
    Map real SCADA input columns (``config.INPUT_COLS`` order) onto the
    ``FirstPrinciplesModel`` input vector. Real signals without a direct
    physical-model counterpart (dosing feed/proportion setpoints, die
    zone setpoints, layer-thickness setpoints) are held at the model's
    nominal value for the whole horizon — a documented simplification,
    not a full bidirectional mapping.
    """
    assert U_real.shape[1] == len(INPUT_COLS), (
        f"expected {len(INPUT_COLS)} real input columns, got {U_real.shape[1]}"
    )
    T = U_real.shape[0]
    U_model = np.tile(model.nominal_inputs(), (T, 1))

    U_model[:, model.sl_u_T_sp] = U_real[:, _REAL_ZONE_SETPOINT_IDX]
    U_model[:, model.sl_u_N_ibc_set] = U_real[:, _REAL_IBC_SETPOINT_IDX]
    U_model[:, model.sl_u_T_sp_cool] = U_real[:, _REAL_COOLING_SETPOINT_IDX]
    U_model[:, model.sl_u_v_haul_set] = U_real[:, _REAL_HAULOFF_SETPOINT_IDX]
    U_model[:, model.sl_u_T_drive_set] = U_real[:, _REAL_WINDER_SPEED_IDX]
    U_model[:, model.sl_u_sigma_web_set] = U_real[:, _REAL_WINDER_TENS_IDX]
    return U_model


# ---------------------------------------------------------------------------
# Grey-box identifier
# ---------------------------------------------------------------------------

@dataclass
class GreyBoxResult:
    """Diagnostics from a ``GreyBoxIdentifier.fit()`` call."""

    scales: np.ndarray
    param_groups: List[str]
    cost: float
    success: bool
    n_iterations: int


class GreyBoxIdentifier:
    """
    Optimises a small vector of multiplicative scale factors — one per
    named physical-parameter group — so that the resulting linearised
    discrete-time ``FirstPrinciplesModel`` best reproduces measured
    (U, Y) data.

    Parameters
    ----------
    model : FirstPrinciplesModel
        Defines subsystem multiplicities and nominal parameters (the
        latter are copied, never mutated in place).
    Ts    : sampling period used for ZOH discretisation.
    cfg   : PhysicalModelConfig (optimiser iteration/bounds settings,
            plus singular-perturbation threshold/gap settings).
    horizon_seconds : MPC prediction horizon in seconds (Np * Ts),
            used to validate the singular-perturbation step-response
            match, if enabled. ``None`` skips that validation.
    """

    #: Physical-parameter groups optimised as single scale factors.
    #: Covers the main thermal/flow/control-gain behaviour without
    #: exploding the search space to every individual array entry
    #: (e.g. per-zone heater gains share one scale, not 12 separate ones).
    PARAM_GROUPS: Tuple[str, ...] = (
        "m0_visc", "Ea_visc", "alpha_drag", "beta_pressure",
        "h_zone", "Kp_zone", "dos_gain", "Kc_dos",
        "die_heater_gain", "Kp_die", "UA_ibc", "UA_cool",
        "J_haul", "Kv_haul", "J_winder", "B_winder", "tau_drive", "E_film",
    )

    _STABILITY_PENALTY: float = 1.0e4

    def __init__(
        self,
        model: FirstPrinciplesModel,
        Ts: float = 3.0,
        cfg: PhysicalModelConfig = PhysicalModelConfig(),
        horizon_seconds: Optional[float] = None,
    ) -> None:
        self.model = model
        self.Ts = Ts
        self.cfg = cfg
        self.horizon_seconds = horizon_seconds
        self._tau_threshold = (
            cfg.fast_time_constant_threshold_factor * Ts if cfg.enable_singular_perturbation else None
        )
        self._base_params = copy.deepcopy(model.params)
        self._x0_warm: Optional[np.ndarray] = None
        self._n_iter = 0
        self.last_diagnostics: Optional[dict] = None

    # ------------------------------------------------------------------
    def fit(self, U: np.ndarray, Y: np.ndarray) -> Tuple[StateSpaceModel, GreyBoxResult, np.ndarray]:
        """
        Optimise the parameter-group scale factors against (U, Y) and
        return the resulting ``StateSpaceModel``, fit diagnostics, and
        the model-space input array actually used (identical to ``U``
        unless real SCADA columns needed mapping onto the model's own
        input layout — see ``map_real_inputs_to_model``).
        """
        if U.shape[1] != self.model.n_inputs:
            logger.info(
                "Input dimension (%d) does not match the physical model "
                "(%d) — mapping real SCADA columns onto the model's "
                "inputs (unmapped ones held at nominal value).",
                U.shape[1], self.model.n_inputs,
            )
            U = map_real_inputs_to_model(U, self.model)

        n_groups = len(self.PARAM_GROUPS)
        x0 = np.ones(n_groups)
        bounds = [(0.2, 5.0)] * n_groups
        self._x0_warm = None
        self._n_iter = 0

        logger.info(
            "Grey-box identification: optimising %d parameter-group "
            "scales (max_iter=%d) ...", n_groups, self.cfg.grey_box_max_iter,
        )
        res = minimize(
            self._cost, x0, args=(U, Y),
            method=self.cfg.grey_box_optimisation_method,
            bounds=bounds,
            options={"maxiter": self.cfg.grey_box_max_iter},
        )
        logger.info(
            "Grey-box identification finished after %d evaluations | "
            "cost=%.6g | success=%s", self._n_iter, res.fun, res.success,
        )

        model = self._build_model(res.x)
        u0 = model.nominal_inputs()
        x0_guess = self._x0_warm if self._x0_warm is not None else model.nominal_state_guess()
        ss, _, _, sp_diagnostics = model.to_state_space_model(
            Ts=self.Ts, u0=u0, x0_guess_override=x0_guess,
            tau_threshold=self._tau_threshold, horizon_seconds=self.horizon_seconds,
            min_spectral_gap=self.cfg.singular_perturbation_min_gap,
        )

        if not (np.all(np.isfinite(ss.A)) and np.all(np.isfinite(ss.B))):
            # The operating-point solve can land on a poorly-converged
            # equilibrium (see FirstPrinciplesModel.solve_operating_point's
            # own convergence warning) whose huge state magnitudes can
            # blow up the singular-perturbation elimination numerically.
            # Falling back to the full-order (non-eliminated) model is
            # always well-defined since it only relies on the Jacobian
            # itself, not on any state-scale-dependent transform.
            logger.warning(
                "Singular-perturbation-reduced model contains non-finite "
                "values (likely from a poorly-converged operating point) "
                "— falling back to the full-order model for this fit."
            )
            ss, _, _, sp_diagnostics = model.to_state_space_model(
                Ts=self.Ts, u0=u0, x0_guess_override=x0_guess,
            )

        if not ss.is_stable:
            logger.warning(
                "Grey-box fit produced an open-loop unstable model "
                "(rho=%.4f) — the real line is never operated without "
                "closed-loop control, so clipping eigenvalues back inside "
                "the unit disk for downstream balanced-truncation/Kalman "
                "compatibility.", ss.spectral_radius,
            )
            ss = StateSpaceModel(
                A=stabilise_discrete_matrix(ss.A, threshold=0.85), B=ss.B, C=ss.C, D=ss.D, Ts=ss.Ts
            )

        result = GreyBoxResult(
            scales=res.x, param_groups=list(self.PARAM_GROUPS),
            cost=float(res.fun), success=bool(res.success), n_iterations=self._n_iter,
        )
        self.last_diagnostics = sp_diagnostics
        return ss, result, U

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_scales(self, scales: np.ndarray) -> PhysicalParameters:
        params = copy.deepcopy(self._base_params)
        for name, scale in zip(self.PARAM_GROUPS, scales):
            base_value = getattr(self._base_params, name)
            setattr(params, name, base_value * scale)
        return params

    def _build_model(self, scales: np.ndarray) -> FirstPrinciplesModel:
        params = self._apply_scales(scales)
        return FirstPrinciplesModel(
            params=params,
            n_extruders=self.model.n_ext, n_zones=self.model.n_zone,
            n_components=self.model.n_comp, n_die_zones=self.model.n_die,
            n_ibc=self.model.n_ibc, n_winders=self.model.n_wind,
        )

    def _cost(self, scales: np.ndarray, U: np.ndarray, Y: np.ndarray) -> float:
        self._n_iter += 1
        model = self._build_model(scales)
        u0 = model.nominal_inputs()
        x0_guess = self._x0_warm if self._x0_warm is not None else model.nominal_state_guess()

        try:
            x0 = model.solve_operating_point(u0, x0_guess)
            A_c, B_c, C_c, D_c = model.linearise(x0, u0)
            if self._tau_threshold is not None:
                A_c, B_c, C_c, D_c, _ = eliminate_fast_states(
                    A_c, B_c, C_c, D_c, self._tau_threshold,
                    min_spectral_gap=self.cfg.singular_perturbation_min_gap,
                )
            A_d, B_d, C_d, D_d = model.discretise(A_c, B_c, C_c, D_c, self.Ts)
        except Exception:
            logger.debug("Grey-box candidate raised during linearisation; penalising.")
            return 1.0e8

        if not (np.all(np.isfinite(A_d)) and np.all(np.isfinite(B_d))):
            return 1.0e8
        self._x0_warm = x0

        rho = float(np.max(np.abs(np.linalg.eigvals(A_d))))
        stab_pen = self._STABILITY_PENALTY * max(0.0, rho - 0.999) ** 2

        ss = StateSpaceModel(A=A_d, B=B_d, C=C_d, D=D_d, Ts=self.Ts)
        Y_hat = ss.simulate(U)
        if not np.all(np.isfinite(Y_hat)):
            return 1.0e8
        mse = float(np.mean((Y - Y_hat) ** 2))
        if not np.isfinite(mse) or mse > 1.0e8:
            return 1.0e8
        return mse + stab_pen
