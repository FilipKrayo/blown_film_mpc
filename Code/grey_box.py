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

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.linalg import cholesky, solve_discrete_lyapunov, svd
from scipy.optimize import minimize
from tqdm import tqdm

from config import INPUT_COLS, OUTPUT_COLS, PhysicalModelConfig
from physical_model import FirstPrinciplesModel, stabilise_discrete_matrix
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

    cost: float
    success: bool
    n_iterations: int
    reduced_order: int
    n_live_inputs: int
    hankel_singular_values: np.ndarray


class GreyBoxIdentifier:
    """
    Grey-box identification via a single linearisation, not a physical-
    parameter-scale search: the ``FirstPrinciplesModel`` is linearised
    ONCE, at the operating point given by the MEAN of the real training
    data (rather than the model's own, potentially unrepresentative,
    nominal design point), discretised, then reduced to a small state
    order via HSV/balanced truncation. The reduced ``A``, ``B`` (only
    the "live", i.e. real-data-mapped, input columns) and ``C`` matrices
    are then optimised directly against measured (U, Y) data via bounded
    L-BFGS-B — ``D`` is left fixed at its (state-order-independent)
    linearised value. This keeps the initial guess physically motivated
    (it comes directly from ``physical_model.py``'s own parameters) while
    avoiding the earlier 18-parameter-group nonlinear rescaling search,
    whose candidates could land arbitrarily far from the real data's
    actual operating region.

    Parameters
    ----------
    model : FirstPrinciplesModel
        Defines subsystem multiplicities and nominal parameters (used
        only to obtain the initial linearisation — never scaled).
    Ts    : sampling period used for ZOH discretisation.
    cfg   : PhysicalModelConfig (optimiser iteration/reduced-order settings).
    horizon_seconds : MPC prediction horizon in seconds (Np * Ts), used
            as the windowed-simulation reset interval (see
            ``simulate_windowed``). ``None`` simulates one continuous
            rollout.
    """

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
        self.reduced_order = cfg.grey_box_reduced_order
        self._n_iter = 0
        self.last_diagnostics: Optional[dict] = None
        #: Whether the last ``fit()`` call mapped real SCADA (U, Y) columns
        #: onto/from the model's own layout (vs. already model-native data,
        #: e.g. synthetic data generated directly from this physical model).
        self.last_real_data_mode: bool = False
        self._real_data_mode = False
        self._y_scale: Optional[np.ndarray] = None
        #: Boolean mask over output columns excluding zero-variance real
        #: columns (no fittable signal) from the cost/accuracy comparison.
        self._active_outputs: Optional[np.ndarray] = None
        self.last_active_outputs: Optional[np.ndarray] = None
        self._window_steps: int = 1
        #: The optimised FirstPrinciplesModel from the last ``fit()`` call
        #: — needed by callers to convert ``ss.simulate(U)`` back into
        #: real-SCADA order/units via ``map_model_outputs_to_real``.
        self.last_physical_model: Optional[FirstPrinciplesModel] = None
        #: Operating point (x0, u0) the last fit()'s returned StateSpaceModel
        #: was linearised around, plus y0=outputs(x0,u0) — needed because the
        #: model's A/B/C/D describe DEVIATION dynamics, not absolute values.
        self.last_x0: Optional[np.ndarray] = None
        self.last_u0: Optional[np.ndarray] = None
        self.last_y0: Optional[np.ndarray] = None
        #: Reduced-order, discrete-time offset carrying forward any leftover
        #: (x0, u0) equilibrium residual — see simulate_windowed()'s docstring.
        self.last_offset: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def fit(self, U: np.ndarray, Y: np.ndarray) -> Tuple[StateSpaceModel, GreyBoxResult, np.ndarray]:
        """
        Linearise the physical model once (at the mean of ``U``), reduce
        it to ``cfg.grey_box_reduced_order`` states via balanced
        truncation, then optimise the reduced ``A``/``B`` (live inputs
        only)/``C`` matrices directly against ``(U, Y)``. Returns the
        resulting ``StateSpaceModel``, fit diagnostics, and the model-
        space input array actually used (identical to ``U`` unless real
        SCADA columns needed mapping onto the model's own input layout —
        see ``map_real_inputs_to_model``).
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
        # Columns with (near-)zero variance (e.g. an unpopulated/disconnected
        # SCADA tag reading a constant 0 for the whole dataset) carry no fit-
        # able signal at all -- excluded entirely rather than floor-dividing
        # by a near-zero std, which would otherwise turn a tiny prediction
        # into a spurious, enormous cost contribution unrelated to fit quality.
        y_std = np.std(Y, axis=0)
        self._active_outputs = y_std > 1.0e-9 if self._real_data_mode else np.ones(Y.shape[1], dtype=bool)
        if self._real_data_mode and not np.all(self._active_outputs):
            logger.warning(
                "Excluding %d zero-variance real output column(s) from the "
                "grey-box cost/accuracy comparison (no fittable signal): "
                "indices %s",
                int(np.sum(~self._active_outputs)),
                list(np.nonzero(~self._active_outputs)[0]),
            )
        self._y_scale = np.maximum(y_std, 1.0e-6) if self._real_data_mode else np.ones(Y.shape[1])
        self.last_real_data_mode = self._real_data_mode
        self.last_active_outputs = self._active_outputs

        # Simulate in short, non-overlapping windows (reset to the
        # equilibrium/deviation-zero state at the start of each window)
        # instead of one continuous open-loop rollout across the whole
        # ~157-day training set. Some physical states are structurally
        # slow integrators (e.g. winder roll build-up/remaining length)
        # whose real-world counterparts get periodically reset (roll
        # changes) that this model has no mechanism for -- an unbounded
        # single continuous simulation diverges regardless of parameter
        # scales. Bounding the rollout to the same horizon the model is
        # actually used for downstream (MPC prediction, horizon_seconds)
        # avoids this while still scoring genuine multi-step dynamic fit.
        self._window_steps = (
            max(1, int(round(self.horizon_seconds / self.Ts))) if self.horizon_seconds else U.shape[0]
        )

        # Linearise ONCE, at the operating point given by the real data's
        # own mean (rather than the model's own, possibly unrepresentative,
        # nominal design point) — this directly targets the root cause
        # behind the winder/tension-channel divergence found previously
        # (the model's nominal web-tension input was 500-3000x smaller
        # than real-data deviations, putting the linearisation nowhere
        # near the region it needed to be valid in).
        #
        # The coordinatewise mean itself isn't necessarily a physically
        # realisable input combination (e.g. it can blend distinct
        # operating regimes that were never simultaneously true, or
        # average a bimodal setpoint into a value never commanded) — use
        # the medoid instead: the actual observed row of U closest to
        # that mean (normalised per-column, since inputs span very
        # different units/scales), guaranteeing u0 is something the real
        # process genuinely did.
        model = self.model
        self.last_physical_model = model
        u_mean = U.mean(axis=0)
        u_std = np.maximum(np.std(U, axis=0), 1.0e-9)
        medoid_idx = int(np.argmin(np.linalg.norm((U - u_mean) / u_std, axis=1)))
        u0 = U[medoid_idx]
        x0_guess = model.nominal_state_guess()
        x0 = model.solve_operating_point(u0, x0_guess)
        A_c, B_c, C_c, D_c = model.linearise(x0, u0)
        A_d, B_d, C_d, D_d = model.discretise(A_c, B_c, C_c, D_c, self.Ts)
        y0 = model.outputs(x0, u0)

        # x0 may not be an exact equilibrium (solve_operating_point logs a
        # warning when it isn't) — rather than silently assuming
        # dynamics(x0, u0) == 0 (which turns any leftover residual into an
        # unmodelled constant drift in every simulated window below), carry
        # it forward as an explicit discrete-time offset term.
        f0_scaled = model.dynamics(x0, u0) / np.maximum(np.abs(x0), 1.0)
        offset_d = model.discretise_offset(A_c, f0_scaled, self.Ts)

        # Inputs the real dataset never actually varies (unmapped columns
        # held at a constant nominal value by map_real_inputs_to_model, see
        # module docstring) have deviation U-u0 == 0 for every sample, so
        # their B-matrix columns contribute exactly zero to any prediction
        # — excluding them from the optimisation avoids wasting parameters
        # (and gradient evaluations) on directions with no signal at all.
        live_mask = np.var(U, axis=0) > 1.0e-12
        live_idx = np.nonzero(live_mask)[0]
        logger.info(
            "Grey-box: %d/%d model inputs vary in the training data "
            "(optimising only their B-matrix columns).",
            live_idx.size, U.shape[1],
        )

        r = min(self.reduced_order, A_d.shape[0] - 1)
        A_r, B_r, C_r, D_r, hsv, T_i = self._balanced_truncate(A_d, B_d, C_d, D_d, r)
        offset_r = T_i @ offset_d
        logger.info(
            "Grey-box: reduced full order %d -> %d states via balanced "
            "truncation (retained %.2f%% HSV energy).",
            A_d.shape[0], r, 100.0 * np.sum(hsv[:r]) / (np.sum(hsv) + 1e-12),
        )

        theta0 = self._pack(A_r, B_r[:, live_idx], C_r)
        self._n_iter = 0

        logger.info(
            "Grey-box identification: optimising reduced A(%dx%d)/"
            "B(%dx%d, live only)/C(%dx%d) directly (%d params, "
            "max_iter=%d) ...",
            r, r, r, live_idx.size, C_r.shape[0], r, theta0.size, self.cfg.grey_box_max_iter,
        )

        # Progress bar wrapper for objective function evaluations (evaluation
        # count, not iteration count, since gradient-based methods like
        # L-BFGS-B call the cost several times per iteration for line search).
        pbar = tqdm(
            total=self.cfg.grey_box_max_iter,
            desc="Grey-box identification",
            unit="eval",
            ncols=80,
        )

        def _cost_with_progress(theta, *args):
            cost = self._cost_linear(theta, *args)
            pbar.update(1)
            return cost

        res = minimize(
            _cost_with_progress, theta0,
            args=(U, Y, r, live_idx, B_r, C_r.shape[0], D_r, u0, y0, model, offset_r),
            method=self.cfg.grey_box_optimisation_method,
            options={"maxiter": self.cfg.grey_box_max_iter},
        )
        pbar.close()
        logger.info(
            "Grey-box identification finished after %d evaluations | "
            "cost=%.6g | success=%s", self._n_iter, res.fun, res.success,
        )

        A_opt, B_opt, C_opt = self._unpack(res.x, r, live_idx.size, C_r.shape[0])
        B_full = B_r.copy()
        B_full[:, live_idx] = B_opt

        rho_opt = float(np.max(np.abs(np.linalg.eigvals(A_opt))))
        if rho_opt >= 1.0:
            logger.warning(
                "Grey-box fit produced an open-loop unstable reduced "
                "model (rho=%.4f) — the real line is never operated "
                "without closed-loop control, so clipping eigenvalues "
                "back inside the unit disk for downstream balanced-"
                "truncation/Kalman compatibility.", rho_opt,
            )
            A_opt = stabilise_discrete_matrix(A_opt, threshold=0.85)

        ss = StateSpaceModel(A=A_opt, B=B_full, C=C_opt, D=D_r, Ts=self.Ts)
        result = GreyBoxResult(
            cost=float(res.fun), success=bool(res.success), n_iterations=self._n_iter,
            reduced_order=r, n_live_inputs=int(live_idx.size), hankel_singular_values=hsv,
        )
        self.last_diagnostics = None
        # ss's A/B/C/D describe DEVIATION dynamics around (x0, u0) — expose
        # the operating point so callers can simulate in deviation
        # coordinates (U - last_u0) and add last_y0 back to recover
        # absolute-unit predictions (see grey_box._cost_linear's own handling).
        self.last_x0 = x0
        self.last_u0 = u0
        self.last_y0 = y0
        self.last_offset = offset_r
        return ss, result, U

    # ------------------------------------------------------------------
    @staticmethod
    def simulate_windowed(
        ss: StateSpaceModel, U: np.ndarray, u0: np.ndarray, y0: np.ndarray, window_steps: int,
        offset: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Simulate a deviation-form ``StateSpaceModel`` (linearised around
        ``(x0, u0)`` with ``y0 = outputs(x0, u0)``) in short, non-
        overlapping windows, resetting the state to the deviation-zero
        equilibrium at the start of each window, instead of one
        continuous open-loop rollout across the whole input sequence.
        See ``_cost()``'s comment for why: some states are structural
        slow integrators (e.g. winder roll build-up) whose real-world
        counterparts get periodically reset in a way this model has no
        mechanism for, so an unbounded continuous rollout diverges
        regardless of parameter scales.

        ``offset`` (reduced-order, discrete-time) carries forward any
        leftover ``x0``/``u0`` equilibrium residual — see ``fit()`` —
        as a constant forcing term, via an extra input column driven by
        a constant unit input, instead of silently assuming it's zero.
        """
        U_dev = U - u0[None, :]
        T = U_dev.shape[0]
        if offset is not None:
            ss = StateSpaceModel(
                A=ss.A, B=np.hstack([ss.B, offset[:, None]]), C=ss.C,
                D=np.hstack([ss.D, np.zeros((ss.C.shape[0], 1))]), Ts=ss.Ts,
            )
            U_dev = np.hstack([U_dev, np.ones((T, 1))])
        Y_hat = np.empty((T, y0.shape[0]))
        for start in range(0, T, window_steps):
            end = min(start + window_steps, T)
            Y_hat[start:end] = ss.simulate(U_dev[start:end]) + y0[None, :]
        return Y_hat

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    _GRAMIAN_REGULARISATION: float = 1.0e-5

    @classmethod
    def _balanced_truncate(
        cls, A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, r: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Reduce ``(A, B, C, D)`` to ``r`` states via balanced truncation
        (mirrors ``model_reduction.py``'s ``_balanced_truncation``). The
        discrete Lyapunov equations behind the controllability/
        observability Gramians are only well-posed for a STABLE ``A`` —
        this model's raw linearisation is genuinely open-loop unstable
        (the bubble/haul-off subsystem, only ever run under closed-loop
        control), so the Gramians are computed from a stabilised copy of
        ``A`` purely to obtain a well-conditioned reduction transform;
        that transform is then applied to the ACTUAL (unstabilised)
        matrices so the reduced model still carries the real dynamics.
        """
        n = A.shape[0]
        rho = float(np.max(np.abs(np.linalg.eigvals(A))))
        A_gram = stabilise_discrete_matrix(A, threshold=0.98) if rho > 0.98 else A

        Wc = solve_discrete_lyapunov(A_gram, B @ B.T)
        Wo = solve_discrete_lyapunov(A_gram.T, C.T @ C)
        Wc = (Wc + Wc.T) / 2
        Wo = (Wo + Wo.T) / 2
        reg_c = cls._GRAMIAN_REGULARISATION * max(1.0, float(np.max(np.abs(Wc)))) * np.eye(n)
        reg_o = cls._GRAMIAN_REGULARISATION * max(1.0, float(np.max(np.abs(Wo)))) * np.eye(n)
        Wc = Wc + reg_c
        Wo = Wo + reg_o

        Lc = cholesky(Wc, lower=True)
        Lo = cholesky(Wo, lower=True)
        M = Lo.T @ Lc
        U_, S, V = svd(M, full_matrices=False)

        r = max(2, min(r, n - 1))
        Sh_i = np.diag(1.0 / np.sqrt(S[:r]))
        T_f = Lc @ V[:r, :].T @ Sh_i     # (n, r) forward transform
        T_i = Sh_i @ U_[:, :r].T @ Lo.T  # (r, n) inverse transform

        A_r = T_i @ A @ T_f
        B_r = T_i @ B
        C_r = C @ T_f
        D_r = D.copy()
        return A_r, B_r, C_r, D_r, S, T_i

    @staticmethod
    def _pack(A_r: np.ndarray, B_live: np.ndarray, C_r: np.ndarray) -> np.ndarray:
        return np.concatenate([A_r.ravel(), B_live.ravel(), C_r.ravel()])

    @staticmethod
    def _unpack(
        theta: np.ndarray, r: int, n_live: int, p: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        i = r * r
        A_r = theta[:i].reshape(r, r)
        j = i + r * n_live
        B_live = theta[i:j].reshape(r, n_live)
        C_r = theta[j:j + p * r].reshape(p, r)
        return A_r, B_live, C_r

    def _cost_linear(
        self,
        theta: np.ndarray,
        U: np.ndarray,
        Y: np.ndarray,
        r: int,
        live_idx: np.ndarray,
        B_template: np.ndarray,
        p: int,
        D_r: np.ndarray,
        u0: np.ndarray,
        y0: np.ndarray,
        model: FirstPrinciplesModel,
        offset: np.ndarray,
    ) -> float:
        self._n_iter += 1
        A_r, B_live, C_r = self._unpack(theta, r, live_idx.size, p)
        if not (np.all(np.isfinite(A_r)) and np.all(np.isfinite(B_live)) and np.all(np.isfinite(C_r))):
            return 1.0e8
        B_r = B_template.copy()
        B_r[:, live_idx] = B_live

        rho = float(np.max(np.abs(np.linalg.eigvals(A_r))))
        stab_pen = self._STABILITY_PENALTY * max(0.0, rho - 0.999) ** 2
        # See _balanced_truncate's docstring / SUB-BUG 3 in the repo notes:
        # the reduced A can still be open-loop unstable — stabilise only
        # the copy used for simulation so the optimiser still gets a real
        # MSE gradient signal instead of a flat overflow-penalty plateau.
        A_sim = stabilise_discrete_matrix(A_r, threshold=0.98) if rho > 0.98 else A_r

        ss = StateSpaceModel(A=A_sim, B=B_r, C=C_r, D=D_r, Ts=self.Ts)
        Y_hat = self.simulate_windowed(ss, U, u0, y0, self._window_steps, offset)
        if self._real_data_mode:
            Y_hat = map_model_outputs_to_real(Y_hat, model)
        if not np.all(np.isfinite(Y_hat)):
            return 1.0e8
        active = self._active_outputs
        mse = float(np.mean(((Y[:, active] - Y_hat[:, active]) / self._y_scale[active]) ** 2))
        if not np.isfinite(mse) or mse > 1.0e8:
            return 1.0e8
        return mse + stab_pen
