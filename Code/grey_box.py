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

from config import INPUT_COLS, OUTPUT_COLS, PhysicalModelConfig
from physical_model import (
    FirstPrinciplesModel,
    PhysicalParameters,
    eliminate_fast_states,
    stabilise_discrete_matrix,
)
from system_identification import StateSpaceModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Real-data <-> model unit conversions
# ---------------------------------------------------------------------------
# Per extrusion_data_legend.xlsx: temperature setpoints/actuals are °C (model
# uses Kelvin), extruder pressure is bar (model uses Pa), IBC/blower signals
# and one winder-drive signal are a 0-100% of full scale (model uses physical
# rpm/torque-like units), haul-off speed is m/min (model uses m/s), and
# winder tension is N (model's "sigma_web" is a stress in Pa, related to
# tension by force = stress * film_thickness * film_width — the same relation
# already used internally, see dynamics()'s `sigma_web * h_film * film_width`
# term). Comparing raw values without these conversions was the dominant
# source of the grey-box fit's catastrophic misfit on real data.
_DEG_C_TO_K = 273.15
_BAR_TO_PA = 1.0e5
_IBC_PCT_FULL_SCALE = 1200.0   # rpm treated as the blower's "100%" (matches nominal_inputs' N_ibc_set)


def _tension_force_per_stress(model: FirstPrinciplesModel) -> float:
    """film_thickness(nominal) * film_width: converts a web stress [Pa] to a tension force [N]."""
    h_film_ref = float(model.nominal_state_guess()[model.sl_h_film][0])
    return h_film_ref * model.params.film_width


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
    ``FirstPrinciplesModel`` input vector, converting engineering units
    (per ``extrusion_data_legend.xlsx``) to the model's SI/native units.
    Real signals without a direct physical-model counterpart (dosing
    feed/proportion setpoints, die zone setpoints, layer-thickness
    setpoints) are held at the model's nominal value for the whole
    horizon — a documented simplification, not a full bidirectional
    mapping.
    """
    assert U_real.shape[1] == len(INPUT_COLS), (
        f"expected {len(INPUT_COLS)} real input columns, got {U_real.shape[1]}"
    )
    T = U_real.shape[0]
    U_model = np.tile(model.nominal_inputs(), (T, 1))

    U_model[:, model.sl_u_T_sp] = U_real[:, _REAL_ZONE_SETPOINT_IDX] + _DEG_C_TO_K
    U_model[:, model.sl_u_N_ibc_set] = U_real[:, _REAL_IBC_SETPOINT_IDX] / 100.0 * _IBC_PCT_FULL_SCALE
    U_model[:, model.sl_u_T_sp_cool] = U_real[:, _REAL_COOLING_SETPOINT_IDX] + _DEG_C_TO_K
    U_model[:, model.sl_u_v_haul_set] = U_real[:, _REAL_HAULOFF_SETPOINT_IDX] / 60.0

    # T_drive_set is a torque-like relaxation target, not a speed — the real
    # signal (0-100% winder drive command) has no exact physical-model
    # counterpart, so it's scaled proportionally against the nominal torque.
    T_drive_nom = model.nominal_inputs()[model.sl_u_T_drive_set]
    U_model[:, model.sl_u_T_drive_set] = U_real[:, _REAL_WINDER_SPEED_IDX] / 100.0 * T_drive_nom

    U_model[:, model.sl_u_sigma_web_set] = U_real[:, _REAL_WINDER_TENS_IDX] / _tension_force_per_stress(model)
    return U_model


# ---------------------------------------------------------------------------
# Real-data output mapping
# ---------------------------------------------------------------------------

def map_model_outputs_to_real(Y_model: np.ndarray, model: FirstPrinciplesModel) -> np.ndarray:
    """
    Convert the physical model's native output vector (``outputs()``'s own
    T_melt/P/delta/N_ibc/T_cool/v_haul/R_roll/L_rem/sigma_web layout and SI
    units) into ``config.OUTPUT_COLS``' real-SCADA column order and
    engineering units, so it can be compared directly against real (U, Y)
    data. Mirrors ``map_real_inputs_to_model`` for the output side.

    The model's own output order does NOT match ``OUTPUT_COLS`` (e.g. the
    model emits T_melt first, but ``OUTPUT_COLS`` lists layer-thickness
    first) — comparing them position-for-position without this remap
    compares physically unrelated quantities and was the dominant source
    of the grey-box fit's catastrophic misfit on real data.
    """
    n_e, n_i, n_w = model.n_ext, model.n_ibc, model.n_wind
    assert (n_e, n_i, n_w) == (3, 3, 2), (
        "output remap assumes the real dataset's 3-extruder/3-IBC/2-winder layout"
    )
    force_per_stress = _tension_force_per_stress(model)

    T = Y_model.shape[0]
    Y_real = np.zeros((T, len(OUTPUT_COLS)))

    # Layer-thickness fraction -> absolute thickness [um], approximated
    # against the nominal total film thickness (no better reference exists
    # for a lumped 0-D bubble/film model).
    h_film_ref = float(model.nominal_state_guess()[model.sl_h_film][0])
    Y_real[:, 0:3] = Y_model[:, model.sl_y_delta] * (h_film_ref * 1.0e6)
    Y_real[:, 3:6] = Y_model[:, model.sl_y_T_melt] - _DEG_C_TO_K
    Y_real[:, 6:9] = Y_model[:, model.sl_y_P] / _BAR_TO_PA
    Y_real[:, 9:12] = Y_model[:, model.sl_y_N_ibc] / _IBC_PCT_FULL_SCALE * 100.0
    Y_real[:, 12:15] = Y_model[:, model.sl_y_T_cool] - _DEG_C_TO_K
    Y_real[:, 15] = Y_model[:, model.sl_y_v_haul][:, 0] * 60.0

    R_roll = Y_model[:, model.sl_y_R_roll]
    L_rem = Y_model[:, model.sl_y_L_rem]
    sigma_web = Y_model[:, model.sl_y_sigma_web]
    for w in range(n_w):
        base = 16 + 3 * w
        Y_real[:, base] = R_roll[:, w] * 1000.0            # m -> mm
        Y_real[:, base + 1] = L_rem[:, w]                  # already meters
        Y_real[:, base + 2] = sigma_web[:, w] * force_per_stress  # Pa -> N
    return Y_real


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
        #: Whether the last ``fit()`` call mapped real SCADA (U, Y) columns
        #: onto/from the model's own layout (vs. already model-native data,
        #: e.g. synthetic data generated directly from this physical model).
        self.last_real_data_mode: bool = False
        self._real_data_mode = False
        self._y_scale: Optional[np.ndarray] = None
        #: The optimised FirstPrinciplesModel from the last ``fit()`` call
        #: — needed by callers to convert ``ss.simulate(U)`` back into
        #: real-SCADA order/units via ``map_model_outputs_to_real``.
        self.last_physical_model: Optional[FirstPrinciplesModel] = None

    # ------------------------------------------------------------------
    def fit(self, U: np.ndarray, Y: np.ndarray) -> Tuple[StateSpaceModel, GreyBoxResult, np.ndarray]:
        """
        Optimise the parameter-group scale factors against (U, Y) and
        return the resulting ``StateSpaceModel``, fit diagnostics, and
        the model-space input array actually used (identical to ``U``
        unless real SCADA columns needed mapping onto the model's own
        input layout — see ``map_real_inputs_to_model``).
        """
        self._real_data_mode = U.shape[1] != self.model.n_inputs
        if self._real_data_mode:
            logger.info(
                "Input dimension (%d) does not match the physical model "
                "(%d) — mapping real SCADA columns onto the model's "
                "inputs (unmapped ones held at nominal value).",
                U.shape[1], self.model.n_inputs,
            )
            U = map_real_inputs_to_model(U, self.model)

        # Real-world channels span wildly different units/scales (um vs bar
        # vs % vs m) -- normalise each output's squared error by its own
        # variance so no single large-magnitude channel dominates the cost.
        self._y_scale = np.maximum(np.std(Y, axis=0), 1.0e-6) if self._real_data_mode else np.ones(Y.shape[1])
        self.last_real_data_mode = self._real_data_mode

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
        self.last_physical_model = model
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
        if self._real_data_mode:
            Y_hat = map_model_outputs_to_real(Y_hat, model)
        if not np.all(np.isfinite(Y_hat)):
            return 1.0e8
        mse = float(np.mean(((Y - Y_hat) / self._y_scale) ** 2))
        if not np.isfinite(mse) or mse > 1.0e8:
            return 1.0e8
        return mse + stab_pen
