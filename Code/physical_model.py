"""
physical_model.py
==================
Executable first-principles nonlinear model of the co-extrusion blown
film line, implementing the physics documented in README.md ("Original
PDE/ODE System Model" and "State Space Linearisation" sections) as
real code rather than as documentation only.

Pipeline of operations provided by ``FirstPrinciplesModel``
-------------------------------------------------------------
1. ``dynamics(x, u)``            -> xdot   nonlinear ODE right-hand side f(x, u)
2. ``outputs(x, u)``             -> y      measurement map g(x, u)
3. ``solve_operating_point(u0)`` -> x0     Newton-Raphson steady state f(x0,u0)=0
                                            (README §2)
4. ``linearise(x0, u0)``         -> (A_c, B_c, C_c, D_c)  numerical Jacobians
                                            (README §3)
5. ``discretise(...)``           -> (A_d, B_d, C_d, D_d)  Van Loan / matrix
                                            exponential ZOH (README §5.1)
6. ``to_state_space_model(...)`` -> StateSpaceModel        drop-in replacement
                                            for the identified model — consumed
                                            unchanged by model_reduction.py,
                                            estimation.py, mpc_controller.py.
7. ``simulate_nonlinear(U, x0)`` -> Y       RK4 forward simulation, used to
                                            generate physically-structured
                                            synthetic data.

State / input / output layout
------------------------------
Multiplicities (n_extruders, n_zones_per_extruder, n_components,
n_die_zones, n_ibc, n_winders) are configurable. At the README's
illustrative configuration (4, 8, 5, 7, 3, 2) the state count is
exactly 146, matching README §10 "Full-Order State Vector":

    Extruder   (T_k,j, P_k, mdot_out_k, T_melt_k)       : 44
    Dosing     (m_i,k, N_dos_i,k, phi_i,k)               : 60
    Die head   (T_die_j, u_die_j)                        : 14
    Bubble     (R, h_film, P_bub, v_z, T_f, sigma_zz)    : 6
    Cooling    (T_IBC_l, N_IBC_l, T_cool_l)               : 9
    Haul-off   (v_haul, sigma_haul, L_job)                : 3
    Winders    (omega_drum_w, R_roll_w, L_w,
                sigma_web_w, T_drive_w)                   : 10
    ------------------------------------------------------------
    Total                                                 : 146

The README documents illustrative I/O totals (u in R^96, y in R^58)
for the full observability/controllability discussion, but does not
enumerate an exhaustive input/output list. The vectors implemented
here are derived directly from the manipulated variables and measured
quantities that actually appear in the per-subsystem equations; their
counts (90 inputs, 22 outputs at the default multiplicities) are not
forced to match 96/58 exactly.

Deliberate simplifications relative to the README's narrative (this
repository explicitly treats full PDE solving — moving mesh, FEM,
implicit time-stepping — as out of scope; see README §12 note):
  * Extruder zone heaters are proportional-only (no separate actuator
    state), keeping exactly 11 states/extruder. Die zones DO get a
    separate first-order actuator state, matching the README §10
    table's "2 states per die zone".
  * The dosing PI law is simplified to a proportional + relaxation
    form (the literal nested integral in README §2.3 would need a 4th
    per-component state not present in the §10 table).
  * The bubble is lumped (0-D): no ``-v_z ∂R/∂z`` convection term.
    Strain rates are algebraic proxies of (v_z, P_bub) relative to the
    nominal operating point rather than re-derived from a moving mesh.
  * ``v_z`` and ``sigma_zz`` are modelled as first-order relaxations
    toward their quasi-steady algebraic values (the README only gives
    quasi-steady relations for them, e.g. §4.3 dF_z/dz = 0).
  * Both winders currently draw from the same haul-off velocity/film
    thickness (the real line splits/alternates between two winders).

Author : Blown Film MPC Project
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.linalg import expm, schur
from scipy.optimize import root

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------

@dataclass
class PhysicalParameters:
    """
    Every physical constant used by :class:`FirstPrinciplesModel`.

    Array-valued fields are indexed per unit (extruder, zone,
    component, IBC unit, winder). Numeric defaults (see
    :func:`default_physical_parameters`) are illustrative/literature-
    informed nominal values for a polymer melt blown-film line — they
    are starting points to be refined by grey-box identification
    against real or synthetic data, not authoritative constants.
    """

    # Extruder (per extruder k)
    rho_melt: np.ndarray
    cp_melt: np.ndarray
    lambda_melt: np.ndarray
    alpha_drag: np.ndarray
    beta_pressure: np.ndarray
    bulk_modulus: np.ndarray
    m0_visc: np.ndarray
    n_powerlaw: np.ndarray
    Ea_visc: np.ndarray
    flow_area: np.ndarray
    barrel_length: np.ndarray
    shear_rate_gain: np.ndarray
    outlet_relax_tau: np.ndarray
    melt_temp_relax_tau: np.ndarray
    screw_speed_nom: np.ndarray
    R_gas: float = 8.314

    # Zone heater (per extruder k, per zone j) — shape (n_ext, n_zone)
    h_zone: np.ndarray = field(repr=False, default=None)
    A_zone: np.ndarray = field(repr=False, default=None)
    thermal_mass_zone: np.ndarray = field(repr=False, default=None)
    Kp_zone: np.ndarray = field(repr=False, default=None)
    P_max_zone: np.ndarray = field(repr=False, default=None)

    # Dosing (per extruder k, per component i) — shape (n_ext, n_comp)
    rho_bulk: np.ndarray = field(repr=False, default=None)
    dos_gain: np.ndarray = field(repr=False, default=None)
    Kc_dos: np.ndarray = field(repr=False, default=None)
    tau_N_dos: np.ndarray = field(repr=False, default=None)
    tau_phi_dos: np.ndarray = field(repr=False, default=None)
    N_dos_nom: np.ndarray = field(repr=False, default=None)

    # Die head (per zone) — shape (n_die,)
    die_thermal_mass: np.ndarray = field(repr=False, default=None)
    die_heater_gain: np.ndarray = field(repr=False, default=None)
    die_UA: np.ndarray = field(repr=False, default=None)
    die_coupling: float = 5.0
    Kp_die: np.ndarray = field(repr=False, default=None)
    tau_die_actuator: np.ndarray = field(repr=False, default=None)

    # Bubble / film
    gamma_air: float = 1.4
    M_air: float = 0.029
    T_air: float = 293.15
    rho_film: float = 920.0
    cp_film: float = 2300.0
    h_ext_conv: float = 40.0
    h_int_conv: float = 30.0
    tau_relax_stress: float = 2.0
    E_film: float = 3.0e8
    film_width: float = 1.2
    bubble_volume_nom: float = 0.5
    ibc_gain: float = 5.0e-6
    ibc_outflow_gain: float = 2.0e-6
    L_ref_axial: float = 1.0
    tau_vz: float = 5.0

    # Cooling (per IBC unit l) — shape (n_ibc,)
    UA_ibc: np.ndarray = field(repr=False, default=None)
    tau_ibc_speed: np.ndarray = field(repr=False, default=None)
    ibc_air_thermal_mass: np.ndarray = field(repr=False, default=None)
    ibc_mdot_gain: np.ndarray = field(repr=False, default=None)
    T_in_ibc: float = 293.15
    rho_air: float = 1.2
    cp_air: float = 1005.0
    M_cool: np.ndarray = field(repr=False, default=None)
    cp_cool: float = 4180.0
    UA_cool: np.ndarray = field(repr=False, default=None)
    Qdot_cool_gain: np.ndarray = field(repr=False, default=None)
    T_amb: float = 293.15

    # Haul-off
    J_haul: float = 50.0
    Kv_haul: float = 200.0
    F_fric: float = 5.0

    # Winder (per winder w) — shape (n_wind,)
    J_winder: np.ndarray = field(repr=False, default=None)
    B_winder: np.ndarray = field(repr=False, default=None)
    tau_drive: np.ndarray = field(repr=False, default=None)
    L_span: np.ndarray = field(repr=False, default=None)
    eta_pack: np.ndarray = field(repr=False, default=None)
    L_target: np.ndarray = field(repr=False, default=None)
    #: Tension-control proportional gain (dimensionless): real winders hold
    #: web tension at a setpoint via a torque feedback loop, not an open
    #: speed/torque profile. Without this, ``sigma_web_set`` (u) had no
    #: path into the dynamics at all (a dead input).
    Kp_tension: float = 1.0


def stabilise_discrete_matrix(A: np.ndarray, threshold: float = 0.98) -> np.ndarray:
    """
    Uniformly scale a discrete-time A matrix down so its spectral
    radius is at most ``threshold``, so simulation/Lyapunov-based
    analysis stays well-posed. The real line is never operated open-
    loop (IBC/haul-off/winder feedback keeps it stable in practice) —
    this substitutes for that closed-loop stabilisation for consumers
    (e.g. synthetic data generation, or as a safety net on a grey-box
    fit) that need a numerically stable discrete-time matrix, without
    representing a modelling error. A uniform scalar rescale (rather
    than per-eigenvalue clipping via eigendecomposition) is used
    deliberately: this system's A can be poorly conditioned/near-
    defective (very different subsystem time constants), so inverting
    its eigenvector matrix to reconstruct a "clipped" A is numerically
    unreliable — a scalar rescale needs no matrix inversion and always
    produces a well-defined result. Not applied inside identification/
    linearisation itself.
    """
    rho = float(np.max(np.abs(np.linalg.eigvals(A))))
    if rho <= threshold:
        return A
    return A * (threshold / rho)


def eliminate_fast_states(
    A_c: np.ndarray,
    B_c: np.ndarray,
    C_c: np.ndarray,
    D_c: np.ndarray,
    tau_threshold: float,
    horizon_seconds: Optional[float] = None,
    min_spectral_gap: float = 3.0,
    u0: Optional[np.ndarray] = None,
    y_scale: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """
    Eliminate states faster than ``tau_threshold`` via singular
    perturbation (README's "Time-scale separation" step, README §12
    pipeline diagram — currently skipped by the full-fidelity model).

    A state with time constant tau << the MPC sample rate completes
    its whole transient between samples, so a sampled-data controller
    can never observe it mid-step regardless of whether it's in the
    model — eliminating it costs nothing from the controller's point
    of view. ``tau_threshold`` should therefore be set as a small
    multiple of Ts (see ``config.PhysicalModelConfig``), not a fixed
    absolute value.

    Uses an ORDERED REAL SCHUR decomposition rather than eigendecom-
    position: this system's continuous-time Jacobian can be poorly
    conditioned (time constants spanning ~0.1s to ~1800s per the
    README), so inverting an eigenvector matrix is numerically
    unreliable (this was already learned the hard way on an earlier
    stabilisation attempt). The Schur transform is orthogonal (no
    inversion needed) and yields an upper-block-triangular form where
    the "fast" (trailing) block's dynamics don't depend on the "slow"
    (leading) block at all — exactly the structure singular
    perturbation needs, and only the small fast x fast block ever
    needs inverting.

    Returns
    -------
    A_slow, B_slow, C_slow, D_slow : reduced continuous-time matrices
    diagnostics : dict with keys
        n_slow, n_fast, tau_fastest_retained, tau_slowest_eliminated,
        spectral_gap (tau_fastest_retained / tau_slowest_eliminated,
        larger is safer), dc_gain_rel_error, horizon_rel_error (None
        if ``horizon_seconds`` wasn't given), warnings (list[str]),
        applied (bool — False if elimination was skipped, e.g.
        because no states qualified as fast, or T22 was singular).
    """
    n = A_c.shape[0]
    warnings_list: list = []
    diagnostics: Dict[str, object] = {
        "n_slow": n, "n_fast": 0,
        "tau_fastest_retained": None, "tau_slowest_eliminated": None,
        "spectral_gap": None, "dc_gain_rel_error": None,
        "horizon_rel_error": None, "horizon_worst_channel_error": None,
        "warnings": warnings_list, "applied": False,
    }

    if tau_threshold <= 0:
        warnings_list.append("tau_threshold <= 0 — singular perturbation disabled.")
        return A_c, B_c, C_c, D_c, diagnostics

    decay_cutoff = 1.0 / tau_threshold

    def is_slow(eigval: complex) -> bool:
        return eigval.real > -decay_cutoff

    try:
        T, Z, sdim = schur(A_c, output="real", sort=is_slow)
    except Exception as exc:
        warnings_list.append(f"Schur decomposition failed ({exc}) — elimination skipped.")
        return A_c, B_c, C_c, D_c, diagnostics

    n_slow, n_fast = sdim, n - sdim
    if n_fast == 0:
        warnings_list.append("No states faster than tau_threshold — nothing eliminated.")
        return A_c, B_c, C_c, D_c, diagnostics
    if n_slow == 0:
        warnings_list.append("Every state is faster than tau_threshold — skipping elimination.")
        return A_c, B_c, C_c, D_c, diagnostics

    T11, T12, T22 = T[:n_slow, :n_slow], T[:n_slow, n_slow:], T[n_slow:, n_slow:]
    B_t, C_t = Z.T @ B_c, C_c @ Z
    B1, B2 = B_t[:n_slow, :], B_t[n_slow:, :]
    C1, C2 = C_t[:, :n_slow], C_t[:, n_slow:]

    eig_slow = np.linalg.eigvals(T11)
    eig_fast = np.linalg.eigvals(T22)
    slow_decaying = eig_slow[eig_slow.real < 0]
    tau_fastest_retained = float(1.0 / (-slow_decaying.real).min()) if len(slow_decaying) else np.inf
    tau_slowest_eliminated = float(1.0 / (-eig_fast.real).max())
    spectral_gap = tau_fastest_retained / tau_slowest_eliminated if tau_slowest_eliminated > 0 else np.inf

    diagnostics.update(
        n_slow=n_slow, n_fast=n_fast,
        tau_fastest_retained=tau_fastest_retained,
        tau_slowest_eliminated=tau_slowest_eliminated,
        spectral_gap=spectral_gap,
    )
    if spectral_gap < min_spectral_gap:
        warnings_list.append(
            f"Spectral gap ({spectral_gap:.2f}x) below the required "
            f"{min_spectral_gap:.2f}x margin — retained and eliminated "
            "timescales are too close for a reliable approximation."
        )

    try:
        T22_inv = np.linalg.inv(T22)
    except np.linalg.LinAlgError:
        warnings_list.append("Fast block T22 is singular — elimination skipped.")
        return A_c, B_c, C_c, D_c, diagnostics

    A_slow = T11
    B_slow = B1 - T12 @ T22_inv @ B2
    C_slow = C1
    D_slow = D_c - C2 @ T22_inv @ B2

    if not all(np.all(np.isfinite(M)) for M in (A_slow, B_slow, C_slow, D_slow)):
        # Can happen if x0 (the operating point the linearisation was
        # taken at) itself has a non-physical magnitude — e.g. from a
        # poorly-converged Newton-Raphson solve — which the state-scale
        # normalisation in linearise() then amplifies. The full-order
        # matrices are always well-defined regardless, so fall back to them.
        warnings_list.append(
            "Reduced matrices contain non-finite values (likely a poorly "
            "converged operating point) — elimination skipped."
        )
        for msg in warnings_list:
            logger.warning("eliminate_fast_states: %s", msg)
        return A_c, B_c, C_c, D_c, diagnostics

    diagnostics["applied"] = True

    # --- Validation 1: steady-state (DC) gain must be preserved -----------
    try:
        dc_full = -C_c @ np.linalg.pinv(A_c) @ B_c + D_c
        dc_slow = -C_slow @ np.linalg.pinv(A_slow) @ B_slow + D_slow
        denom = max(float(np.linalg.norm(dc_full)), 1e-12)
        dc_error = float(np.linalg.norm(dc_full - dc_slow) / denom)
        diagnostics["dc_gain_rel_error"] = dc_error
        if dc_error > 0.05:
            warnings_list.append(f"DC-gain mismatch after elimination: {dc_error:.1%}.")
    except Exception as exc:
        warnings_list.append(f"DC-gain check failed ({exc}).")

    # --- Validation 2: step response over the MPC horizon -----------------
    # Both discretised systems are stabilised (eigenvalue-clipped) purely
    # for this comparison — if the retained slow dynamics include an
    # open-loop unstable mode (e.g. the bubble subsystem, a known property
    # of this model unrelated to the elimination itself), comparing raw
    # trajectories would just measure exponential blow-up divergence
    # rather than whether the reduction preserved the dynamic SHAPE.
    if horizon_seconds is not None and horizon_seconds > 0:
        try:
            n_steps = 100
            dt = horizon_seconds / n_steps
            A_d_full, B_d_full = FirstPrinciplesModel.discretise(A_c, B_c, C_c, D_c, dt)[:2]
            A_d_slow, B_d_slow = FirstPrinciplesModel.discretise(A_slow, B_slow, C_slow, D_slow, dt)[:2]
            A_d_full = stabilise_discrete_matrix(A_d_full)
            A_d_slow = stabilise_discrete_matrix(A_d_slow)

            m = B_c.shape[1]
            # Scale the validation step relative to each input's own
            # nominal magnitude (a flat unit step is wildly disproportionate
            # across channels spanning e.g. ~500 degC vs ~0.003 kg/s feed
            # rates, and would swamp the comparison with an unrealistic
            # excitation rather than measuring the reduction's accuracy).
            u_step = 0.02 * np.maximum(np.abs(u0), 1.0) if u0 is not None else np.ones(m)
            x_full, x_slow = np.zeros(n), np.zeros(n_slow)
            Y_full = np.zeros((n_steps, C_c.shape[0]))
            Y_slow = np.zeros((n_steps, C_slow.shape[0]))
            for t in range(n_steps):
                Y_full[t] = C_c @ x_full + D_c @ u_step
                Y_slow[t] = C_slow @ x_slow + D_slow @ u_step
                x_full = A_d_full @ x_full + B_d_full @ u_step
                x_slow = A_d_slow @ x_slow + B_d_slow @ u_step

            # Normalise per output channel before comparing: raw physical
            # units span huge scale differences (e.g. pressure ~1e7 Pa vs.
            # velocity ~0.6 m/s), so an unnormalised norm would be
            # dominated entirely by whichever output happens to have the
            # largest absolute magnitude, regardless of how well the other
            # (possibly much more important) outputs are reproduced.
            p = C_c.shape[0]
            if y_scale is not None:
                out_scale = np.maximum(np.abs(y_scale), 1e-6)
            else:
                out_scale = np.maximum(np.max(np.abs(Y_full), axis=0), 1e-6)
            Y_full_n = Y_full / out_scale
            Y_slow_n = Y_slow / out_scale

            # Per-channel relative error, not one aggregate norm: a single
            # channel with an outsized raw magnitude (e.g. pressure) would
            # otherwise dominate a combined norm even when every other
            # (possibly more important) output matches well. Report the
            # typical (median) and worst-case channel so both are visible.
            per_channel_num = np.linalg.norm(Y_full_n - Y_slow_n, axis=0)
            per_channel_den = np.maximum(np.linalg.norm(Y_full_n, axis=0), 1e-9)
            per_channel_rel = per_channel_num / per_channel_den
            horizon_error = float(np.median(per_channel_rel))
            horizon_worst = float(np.max(per_channel_rel))
            diagnostics["horizon_rel_error"] = horizon_error
            diagnostics["horizon_worst_channel_error"] = horizon_worst
            if horizon_error > 0.05:
                warnings_list.append(
                    f"Step-response mismatch over the MPC horizon: median "
                    f"{horizon_error:.1%} across outputs (worst channel "
                    f"{horizon_worst:.1%})."
                )
            elif horizon_worst > 0.20:
                warnings_list.append(
                    f"One or more output channels mismatch over the MPC "
                    f"horizon (worst channel {horizon_worst:.1%}) even "
                    f"though the typical/median channel is fine "
                    f"({horizon_error:.1%}) — inspect per-output before "
                    "trusting the reduced model for that specific signal."
                )
        except Exception as exc:
            warnings_list.append(f"Horizon validation failed ({exc}).")

    for msg in warnings_list:
        logger.warning("eliminate_fast_states: %s", msg)

    return A_slow, B_slow, C_slow, D_slow, diagnostics



def default_physical_parameters(
    n_extruders: int,
    n_zones: int,
    n_components: int,
    n_die_zones: int,
    n_ibc: int,
    n_winders: int,
) -> PhysicalParameters:
    """
    Build a :class:`PhysicalParameters` instance with illustrative
    defaults. Replicated units (extruders, zones, components, IBC
    units, winders) get a small (+/-3%) deterministic per-index
    variation rather than perfectly identical values — real hardware
    is never exactly identical across units, and identical parameters
    (combined with identical nominal inputs) make otherwise-symmetric
    subsystems exactly rank-deficient in the controllability/
    observability Gramian used by balanced truncation downstream.
    """
    rng = np.random.default_rng(0)

    def vary(value: float, *shape: int) -> np.ndarray:
        return value * (1.0 + 0.03 * rng.standard_normal(shape))

    return PhysicalParameters(
        rho_melt=vary(900.0, n_extruders),
        cp_melt=vary(2200.0, n_extruders),
        lambda_melt=vary(0.25, n_extruders),
        alpha_drag=vary(1.5e-7, n_extruders),
        beta_pressure=vary(5.0e-11, n_extruders),
        bulk_modulus=vary(1.5e9, n_extruders),
        m0_visc=vary(0.03, n_extruders),
        n_powerlaw=vary(0.4, n_extruders),
        Ea_visc=vary(4.5e4, n_extruders),
        flow_area=vary(8.0e-4, n_extruders),
        barrel_length=vary(1.2, n_extruders),
        shear_rate_gain=vary(2.0, n_extruders),
        outlet_relax_tau=vary(2.0, n_extruders),
        melt_temp_relax_tau=vary(8.0, n_extruders),
        screw_speed_nom=vary(100.0, n_extruders),
        h_zone=vary(25.0, n_extruders, n_zones),
        A_zone=vary(0.05, n_extruders, n_zones),
        thermal_mass_zone=vary(5.0e4, n_extruders, n_zones),
        Kp_zone=vary(50.0, n_extruders, n_zones),
        P_max_zone=vary(8000.0, n_extruders, n_zones),
        rho_bulk=vary(550.0, n_extruders, n_components),
        dos_gain=vary(2.0e-7, n_extruders, n_components),
        Kc_dos=vary(0.5, n_extruders, n_components),
        tau_N_dos=vary(3.0, n_extruders, n_components),
        tau_phi_dos=vary(4.0, n_extruders, n_components),
        N_dos_nom=vary(30.0, n_extruders, n_components),
        die_thermal_mass=vary(3.0e4, n_die_zones),
        die_heater_gain=vary(40.0, n_die_zones),
        die_UA=vary(15.0, n_die_zones),
        Kp_die=vary(30.0, n_die_zones),
        tau_die_actuator=vary(6.0, n_die_zones),
        UA_ibc=vary(200.0, n_ibc),
        tau_ibc_speed=vary(2.0, n_ibc),
        ibc_air_thermal_mass=vary(1500.0, n_ibc),
        ibc_mdot_gain=vary(0.02, n_ibc),
        M_cool=vary(50.0, n_ibc),
        UA_cool=vary(100.0, n_ibc),
        Qdot_cool_gain=vary(500.0, n_ibc),
        J_winder=vary(8.0, n_winders),
        B_winder=vary(2.0, n_winders),
        tau_drive=vary(0.5, n_winders),
        L_span=vary(1.5, n_winders),
        eta_pack=vary(0.9, n_winders),
        L_target=vary(5000.0, n_winders),
    )


# ---------------------------------------------------------------------------
# First-principles model
# ---------------------------------------------------------------------------

class FirstPrinciplesModel:
    """
    Executable nonlinear first-principles model of the blown film
    line — see module docstring for the state/input/output layout
    and the list of deliberate simplifications.

    Parameters
    ----------
    params        : PhysicalParameters
    n_extruders, n_zones, n_components, n_die_zones, n_ibc, n_winders
                  : subsystem multiplicities. Defaults (3, 4, 5, 7, 3, 2)
                    align extruder/IBC/winder counts with the real
                    dataset (config.PhysicalModelConfig) rather than
                    the README's illustrative 146-state configuration
                    (4, 8, 5, 7, 3, 2) — see module docstring.
    """

    def __init__(
        self,
        params: Optional[PhysicalParameters] = None,
        n_extruders: int = 3,
        n_zones: int = 4,
        n_components: int = 5,
        n_die_zones: int = 7,
        n_ibc: int = 3,
        n_winders: int = 2,
    ) -> None:
        self.n_ext = n_extruders
        self.n_zone = n_zones
        self.n_comp = n_components
        self.n_die = n_die_zones
        self.n_ibc = n_ibc
        self.n_wind = n_winders
        self.params = params or default_physical_parameters(
            n_extruders, n_zones, n_components, n_die_zones, n_ibc, n_winders
        )
        self._build_layout()

    # ------------------------------------------------------------------
    # Layout bookkeeping
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        """Compute slice objects for every state/input/output sub-block."""
        idx = 0

        def take(size: int) -> slice:
            nonlocal idx
            s = slice(idx, idx + size)
            idx += size
            return s

        n_e, n_z, n_c = self.n_ext, self.n_zone, self.n_comp
        n_d, n_i, n_w = self.n_die, self.n_ibc, self.n_wind

        self.sl_T = take(n_e * n_z)
        self.sl_P = take(n_e)
        self.sl_mdot_out = take(n_e)
        self.sl_T_melt = take(n_e)
        self.sl_m_dos = take(n_e * n_c)
        self.sl_N_dos = take(n_e * n_c)
        self.sl_phi = take(n_e * n_c)
        self.sl_T_die = take(n_d)
        self.sl_u_die = take(n_d)
        self.sl_R = take(1)
        self.sl_h_film = take(1)
        self.sl_P_bub = take(1)
        self.sl_v_z = take(1)
        self.sl_T_f = take(1)
        self.sl_sigma_zz = take(1)
        self.sl_T_ibc = take(n_i)
        self.sl_N_ibc = take(n_i)
        self.sl_T_cool = take(n_i)
        self.sl_v_haul = take(1)
        self.sl_sigma_haul = take(1)
        self.sl_L_job = take(1)
        self.sl_omega_drum = take(n_w)
        self.sl_R_roll = take(n_w)
        self.sl_L_w = take(n_w)
        self.sl_sigma_web = take(n_w)
        self.sl_T_drive = take(n_w)
        self.n_states = idx

        # Inputs
        idx = 0
        self.sl_u_T_sp = take(n_e * n_z)
        self.sl_u_mdot_feed = take(n_e * n_c)
        self.sl_u_phi_set = take(n_e * n_c)
        self.sl_u_T_sp_die = take(n_d)
        self.sl_u_N_ibc_set = take(n_i)
        self.sl_u_T_sp_cool = take(n_i)
        self.sl_u_v_haul_set = take(1)
        self.sl_u_T_drive_set = take(n_w)
        self.sl_u_sigma_web_set = take(n_w)
        self.n_inputs = idx

        # Outputs
        idx = 0
        self.sl_y_T_melt = take(n_e)
        self.sl_y_P = take(n_e)
        self.sl_y_delta = take(n_e)   # per-extruder layer-thickness fraction
        self.sl_y_N_ibc = take(n_i)
        self.sl_y_T_cool = take(n_i)
        self.sl_y_v_haul = take(1)
        self.sl_y_R_roll = take(n_w)
        self.sl_y_L_rem = take(n_w)
        self.sl_y_sigma_web = take(n_w)
        self.n_outputs = idx

    # ------------------------------------------------------------------
    # Nominal operating point
    # ------------------------------------------------------------------

    def _design_point(self) -> dict:
        """
        Derive a mutually mass/force-balanced nominal design point so
        the Newton-Raphson solve (§2) starts close to equilibrium
        instead of from an arbitrary, badly-imbalanced guess.
        """
        p = self.params
        T_nom = 493.15
        gamma_dot = p.shear_rate_gain * p.screw_speed_nom / 60.0
        mu_nom = (p.m0_visc * np.exp(p.Ea_visc / (p.R_gas * T_nom))
                  * np.maximum(gamma_dot, 1e-6) ** (p.n_powerlaw - 1.0))

        # Dosing: feed rate balanced against outflow at N_dos_nom.
        mdot_out_dos_nom = p.rho_bulk * p.dos_gain * p.N_dos_nom  # (n_ext, n_comp)
        phi_nom = mdot_out_dos_nom / np.sum(mdot_out_dos_nom, axis=1, keepdims=True)
        rho_mix_nom = np.sum(phi_nom * p.rho_bulk, axis=1)  # (n_ext,)
        mdot_extruder_nom = np.sum(mdot_out_dos_nom, axis=1)  # (n_ext,)

        # Extruder pressure that makes Q_k match the dosing-fed throughput.
        Q_nom = mdot_extruder_nom / np.where(rho_mix_nom > 0, rho_mix_nom, 900.0)
        P_nom = (p.alpha_drag * p.screw_speed_nom - Q_nom) * mu_nom / p.beta_pressure
        P_nom = np.clip(P_nom, 1.0e5, 5.0e7)

        v_z_nom = float(np.mean(Q_nom / p.flow_area))
        mdot_total_nom = float(np.sum(mdot_extruder_nom))

        return dict(
            T_nom=T_nom, mdot_out_dos_nom=mdot_out_dos_nom, phi_nom=phi_nom,
            rho_mix_nom=rho_mix_nom, mdot_extruder_nom=mdot_extruder_nom,
            P_nom=P_nom, v_z_nom=v_z_nom, mdot_total_nom=mdot_total_nom,
        )

    def nominal_inputs(self) -> np.ndarray:
        """A physically reasonable, mass-balanced nominal input vector u0 (README §2)."""
        p = self.params
        dp = self._design_point()
        u = np.zeros(self.n_inputs)
        u[self.sl_u_T_sp] = dp["T_nom"]
        u[self.sl_u_mdot_feed] = dp["mdot_out_dos_nom"].ravel()
        u[self.sl_u_phi_set] = dp["phi_nom"].ravel()
        # u_die's dynamics are a proportional (offset-prone) actuator law:
        # u_die_dot = (Kp_die*(T_sp_die - T_die) - u_die) / tau. Steady state
        # requires u_die = die_UA*(T_die-T_amb)/die_heater_gain (to balance
        # heat loss) AND u_die = Kp_die*(T_sp_die - T_die) simultaneously, so
        # T_sp_die must be offset above the die temperature target by the
        # actuator's proportional droop (u_die_eq/Kp_die) rather than set
        # equal to it — otherwise no consistent equilibrium exists and the
        # operating-point Newton solve diverges trying to satisfy both.
        u_die_eq = p.die_UA * (dp["T_nom"] - p.T_amb) / p.die_heater_gain
        u[self.sl_u_T_sp_die] = dp["T_nom"] + u_die_eq / p.Kp_die
        u[self.sl_u_N_ibc_set] = 1200.0
        u[self.sl_u_T_sp_cool] = 288.15
        u[self.sl_u_v_haul_set] = 0.6   # m/s (~36 m/min)
        u[self.sl_u_T_drive_set] = 6.0
        u[self.sl_u_sigma_web_set] = 500.0
        return u

    def nominal_state_guess(self) -> np.ndarray:
        """A mass/force-balanced initial guess for the Newton-Raphson solve."""
        p = self.params
        dp = self._design_point()

        # Bubble/film: fixed realistic magnitudes (R ~ 0.2 m, h ~ 50 um);
        # deriving R from mdot_total/v_z is circular since v_z_nom is
        # itself derived from the same mdot_total (see _design_point),
        # so it carries no independent information here.
        R_nom = 0.2
        h_film_nom = 5.0e-5
        # P_bub's IBC mass-balance equilibrium (mdot_in_ibc == mdot_out_ibc,
        # see dynamics()'s P_bub_dot) is ibc_gain*N_ibc / (2*ibc_outflow_gain)
        # for the nominal N_ibc_set=1200 — NOT the previously-hardcoded 300.0,
        # which left a mass-balance residual amplified ~2.4e5x by the ideal-
        # gas conversion factor (gamma_air/bubble_volume_nom * R_gas*T_air/M_air),
        # large enough on its own to make the Newton solve diverge.
        P_bub_nom = p.ibc_gain * 1200.0 / max(2.0 * p.ibc_outflow_gain, 1e-12)
        v_haul_nom = 0.6
        # sigma_haul_dot = E_film*h_film*width/L_span * (v_haul - v_z) has a
        # huge stiffness gain (E_film ~1e8) — deriving v_z_nom independently
        # from a mass-balance formula (mdot_total/(rho*2*pi*R*h)) leaves a
        # tiny (v_haul - v_z) mismatch that gets amplified into a residual
        # of O(1e3), which was enough to make the Newton solve diverge to
        # non-physical magnitudes. Set v_z_nom = v_haul_nom directly instead
        # (their equality IS the equilibrium condition for that state), so
        # the initial guess starts near that equation's root.
        v_z_nom = v_haul_nom
        sigma_zz_nom = P_bub_nom * R_nom / max(2 * h_film_nom, 1e-9)

        # Winder: R_roll*omega_drum = v_haul at equilibrium; back out the
        # drive torque that balances omega_drum_dot at that point.
        R_roll_nom = 0.2
        omega_drum_nom = v_haul_nom / R_roll_nom
        sigma_web_nom = 500.0
        T_drive_nom = (sigma_web_nom * h_film_nom * p.film_width * R_roll_nom
                       + p.B_winder * omega_drum_nom)

        x = np.zeros(self.n_states)
        x[self.sl_T] = dp["T_nom"]
        x[self.sl_P] = np.repeat(dp["P_nom"], 1)
        x[self.sl_mdot_out] = dp["mdot_extruder_nom"]
        x[self.sl_T_melt] = dp["T_nom"]
        x[self.sl_m_dos] = 1.0
        x[self.sl_N_dos] = self.params.N_dos_nom.ravel()
        x[self.sl_phi] = dp["phi_nom"].ravel()
        x[self.sl_T_die] = dp["T_nom"]
        # Matches the T_sp_die offset applied in nominal_inputs() so u_die
        # starts at its actual steady-state value instead of an arbitrary
        # placeholder (previously 0.5, vs. a true equilibrium of O(10-100)).
        x[self.sl_u_die] = p.die_UA * (dp["T_nom"] - p.T_amb) / p.die_heater_gain
        x[self.sl_R] = R_nom
        x[self.sl_h_film] = h_film_nom
        x[self.sl_P_bub] = P_bub_nom
        x[self.sl_v_z] = v_z_nom
        # T_f_dot=0 requires T_f = weighted average of T_amb and T_ibc (its
        # only two coupling terms) — the previously-hardcoded 313.15 ignored
        # that balance (true equilibrium ~291K), leaving a residual of ~-15.
        T_f_nom = (p.h_ext_conv * p.T_amb + p.h_int_conv * 288.15) / (p.h_ext_conv + p.h_int_conv)
        x[self.sl_T_f] = T_f_nom
        x[self.sl_sigma_zz] = sigma_zz_nom
        x[self.sl_T_ibc] = 288.15
        x[self.sl_N_ibc] = 1200.0
        x[self.sl_T_cool] = 288.15
        x[self.sl_v_haul] = v_haul_nom
        x[self.sl_sigma_haul] = sigma_web_nom
        x[self.sl_L_job] = 0.0
        x[self.sl_omega_drum] = omega_drum_nom
        x[self.sl_R_roll] = R_roll_nom
        x[self.sl_L_w] = 0.0
        x[self.sl_sigma_web] = sigma_web_nom
        x[self.sl_T_drive] = T_drive_nom
        return x

    # ------------------------------------------------------------------
    # Dynamics f(x, u)
    # ------------------------------------------------------------------

    def _extruder_flow(self, T: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Per-extruder volumetric throughput Q_k (README §1.1/§1.2), shared
        by ``dynamics()`` and ``outputs()`` so the layer-thickness output
        uses the same flow calculation as the mass-balance dynamics."""
        p = self.params
        gamma_dot = p.shear_rate_gain * p.screw_speed_nom / 60.0
        mu_outlet = (p.m0_visc * np.exp(p.Ea_visc / (p.R_gas * T[:, -1]))
                     * np.maximum(gamma_dot, 1e-6) ** (p.n_powerlaw - 1.0))
        return p.alpha_drag * p.screw_speed_nom - p.beta_pressure * P / mu_outlet

    def dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Nonlinear ODE right-hand side xdot = f(x, u)."""
        p = self.params
        n_e, n_z, n_c = self.n_ext, self.n_zone, self.n_comp
        xdot = np.zeros_like(x)

        T = x[self.sl_T].reshape(n_e, n_z)
        P = x[self.sl_P]
        mdot_out = x[self.sl_mdot_out]
        T_melt = x[self.sl_T_melt]
        m_dos = x[self.sl_m_dos].reshape(n_e, n_c)
        N_dos = x[self.sl_N_dos].reshape(n_e, n_c)
        phi = x[self.sl_phi].reshape(n_e, n_c)
        T_die = x[self.sl_T_die]
        u_die = x[self.sl_u_die]
        R = x[self.sl_R][0]
        h_film = x[self.sl_h_film][0]
        P_bub = x[self.sl_P_bub][0]
        v_z = x[self.sl_v_z][0]
        T_f = x[self.sl_T_f][0]
        sigma_zz = x[self.sl_sigma_zz][0]
        T_ibc = x[self.sl_T_ibc]
        N_ibc = x[self.sl_N_ibc]
        T_cool = x[self.sl_T_cool]
        v_haul = x[self.sl_v_haul][0]
        sigma_haul = x[self.sl_sigma_haul][0]
        omega_drum = x[self.sl_omega_drum]
        R_roll = x[self.sl_R_roll]
        sigma_web = x[self.sl_sigma_web]
        T_drive = x[self.sl_T_drive]

        T_sp = u[self.sl_u_T_sp].reshape(n_e, n_z)
        mdot_feed = u[self.sl_u_mdot_feed].reshape(n_e, n_c)
        phi_set = u[self.sl_u_phi_set].reshape(n_e, n_c)
        T_sp_die = u[self.sl_u_T_sp_die]
        N_ibc_set = u[self.sl_u_N_ibc_set]
        T_sp_cool = u[self.sl_u_T_sp_cool]
        v_haul_set = u[self.sl_u_v_haul_set][0]
        T_drive_set = u[self.sl_u_T_drive_set]
        sigma_web_set = u[self.sl_u_sigma_web_set]

        # --- Dosing (§2) --------------------------------------------------
        Q_dos = p.dos_gain * N_dos
        mdot_out_dos = p.rho_bulk * Q_dos
        m_dos_dot = mdot_feed - mdot_out_dos
        N_dos_dot = (p.Kc_dos * (phi_set - phi)
                     - (N_dos - p.N_dos_nom) / p.tau_N_dos)
        mdot_out_dos_sum = np.sum(mdot_out_dos, axis=1, keepdims=True)
        mdot_out_dos_sum = np.where(mdot_out_dos_sum <= 0, 1e-9, mdot_out_dos_sum)
        phi_alg = mdot_out_dos / mdot_out_dos_sum
        phi_dot = (phi_alg - phi) / p.tau_phi_dos
        rho_mix = np.sum(phi * p.rho_bulk, axis=1)  # per extruder

        # --- Extruder (§1) -------------------------------------------------
        gamma_dot = p.shear_rate_gain * p.screw_speed_nom / 60.0
        mu = (p.m0_visc[:, None] * np.exp(p.Ea_visc[:, None] / (p.R_gas * T))
              * np.maximum(gamma_dot[:, None], 1e-6) ** (p.n_powerlaw[:, None] - 1.0))
        eta_diss = mu * gamma_dot[:, None] ** 2

        Q_k = self._extruder_flow(T, P)
        v_zk = Q_k / p.flow_area
        dz = p.barrel_length / n_z

        qdot_heater = np.zeros((n_e, n_z))
        for j in range(n_z):
            power_cmd = p.Kp_zone[:, j] * (T_sp[:, j] - T[:, j])
            power_cmd = np.clip(power_cmd, 0.0, p.P_max_zone[:, j])
            qdot_heater[:, j] = p.h_zone[:, j] * p.A_zone[:, j] * power_cmd

        T_dot = np.zeros((n_e, n_z))
        for j in range(n_z):
            T_left = T[:, j - 1] if j > 0 else T[:, j]
            T_right = T[:, j + 1] if j < n_z - 1 else T[:, j]
            diffusion = p.lambda_melt / dz ** 2 * (T_right - 2 * T[:, j] + T_left)
            convection = (p.rho_melt * v_zk / dz) * (T[:, j] - T_left)
            T_dot[:, j] = (diffusion - convection + eta_diss[:, j]
                           + qdot_heater[:, j]) / (p.rho_melt * p.cp_melt)

        mdot_in_extruder = np.sum(mdot_out_dos, axis=1)
        P_dot = p.bulk_modulus / p.rho_melt * (mdot_in_extruder - mdot_out)
        mdot_out_alg = Q_k * np.where(rho_mix > 0, rho_mix, p.rho_melt)
        mdot_out_dot = (mdot_out_alg - mdot_out) / p.outlet_relax_tau
        T_melt_dot = (T[:, -1] - T_melt) / p.melt_temp_relax_tau

        # --- Die head (§3.1/3.2) -------------------------------------------
        T_die_dot = np.zeros(self.n_die)
        for j in range(self.n_die):
            left = T_die[j - 1] if j > 0 else T_die[j]
            right = T_die[j + 1] if j < self.n_die - 1 else T_die[j]
            coupling = p.die_coupling * (left - 2 * T_die[j] + right)
            T_die_dot[j] = (p.die_heater_gain[j] * u_die[j]
                            - p.die_UA[j] * (T_die[j] - p.T_amb)
                            + coupling) / p.die_thermal_mass[j]
        u_die_dot = (p.Kp_die * (T_sp_die - T_die) - u_die) / p.tau_die_actuator

        # --- Bubble / film (§4/§5.1) ----------------------------------------
        mdot_total = float(np.sum(mdot_out))
        v_z_alg = mdot_total / max(p.rho_film * 2 * np.pi * R * h_film, 1e-9)
        eps_dot_z = (v_z - v_z_alg) / p.L_ref_axial
        eps_dot_theta = (P_bub - 300.0) / max(p.E_film * h_film, 1e-9)

        R_dot = 0.5 * R * (eps_dot_theta - eps_dot_z)
        h_film_dot = -h_film * (eps_dot_z + eps_dot_theta)

        mdot_in_ibc = p.ibc_gain * float(np.mean(N_ibc)) - p.ibc_outflow_gain * P_bub
        mdot_out_ibc = p.ibc_outflow_gain * P_bub
        P_bub_dot = (p.gamma_air / p.bubble_volume_nom * (mdot_in_ibc - mdot_out_ibc)
                     * (p.R_gas * p.T_air / p.M_air))
        v_z_dot = (v_z_alg - v_z) / p.tau_vz
        T_f_dot = (-p.h_ext_conv * (T_f - p.T_amb)
                   - p.h_int_conv * (T_f - float(np.mean(T_ibc)))) / (
                       p.rho_film * p.cp_film * max(h_film, 1e-9))
        sigma_zz_alg = P_bub * R / max(2 * h_film, 1e-9)
        sigma_zz_dot = (sigma_zz_alg - sigma_zz) / p.tau_relax_stress

        # --- Cooling (§5.2-5.4) ----------------------------------------------
        mdot_ibc_air = p.ibc_mdot_gain * N_ibc
        T_ibc_dot = (mdot_ibc_air * p.cp_air * (p.T_in_ibc - T_ibc)
                     - p.UA_ibc * (T_ibc - T_f)) / (p.ibc_air_thermal_mass * p.cp_air)
        N_ibc_dot = (N_ibc_set - N_ibc) / p.tau_ibc_speed
        Qdot_cool = p.Qdot_cool_gain * (T_sp_cool - T_cool)
        T_cool_dot = (Qdot_cool - p.UA_cool * (T_cool - p.T_amb)) / (p.M_cool * p.cp_cool)

        # --- Haul-off (§6) -----------------------------------------------------
        v_haul_dot = (p.Kv_haul * (v_haul_set - v_haul)
                      - sigma_haul * h_film * p.film_width - p.F_fric) / p.J_haul
        sigma_haul_dot = (p.E_film * h_film * p.film_width / p.L_span[0]
                          * (v_haul - v_z))
        L_job_dot = v_haul

        # --- Winders (§7) --------------------------------------------------------
        omega_drum_dot = (T_drive - sigma_web * h_film * p.film_width * R_roll
                          - p.B_winder * omega_drum) / p.J_winder
        R_roll_dot = (h_film * v_haul) / np.maximum(2 * np.pi * R_roll * p.eta_pack, 1e-9)
        L_w_dot = np.full(self.n_wind, v_haul)
        sigma_web_dot = (p.E_film * h_film * p.film_width / p.L_span
                         * (omega_drum * R_roll - v_haul))
        # Tension feedback: real winders adjust drive torque to hold web
        # tension at sigma_web_set (a closed-loop torque/tension control
        # loop), not just follow an open speed/torque profile. The
        # correction is scaled by h_film*film_width*R_roll to match the
        # torque units of the sigma_web term already in omega_drum_dot's
        # balance (see T_drive_nom's equilibrium formula, unaffected since
        # sigma_web_set == sigma_web_nom at the nominal operating point).
        T_drive_dot = (T_drive_set
                       + p.Kp_tension * h_film * p.film_width * R_roll * (sigma_web_set - sigma_web)
                       - T_drive) / p.tau_drive

        xdot[self.sl_T] = T_dot.ravel()
        xdot[self.sl_P] = P_dot
        xdot[self.sl_mdot_out] = mdot_out_dot
        xdot[self.sl_T_melt] = T_melt_dot
        xdot[self.sl_m_dos] = m_dos_dot.ravel()
        xdot[self.sl_N_dos] = N_dos_dot.ravel()
        xdot[self.sl_phi] = phi_dot.ravel()
        xdot[self.sl_T_die] = T_die_dot
        xdot[self.sl_u_die] = u_die_dot
        xdot[self.sl_R] = R_dot
        xdot[self.sl_h_film] = h_film_dot
        xdot[self.sl_P_bub] = P_bub_dot
        xdot[self.sl_v_z] = v_z_dot
        xdot[self.sl_T_f] = T_f_dot
        xdot[self.sl_sigma_zz] = sigma_zz_dot
        xdot[self.sl_T_ibc] = T_ibc_dot
        xdot[self.sl_N_ibc] = N_ibc_dot
        xdot[self.sl_T_cool] = T_cool_dot
        xdot[self.sl_v_haul] = v_haul_dot
        xdot[self.sl_sigma_haul] = sigma_haul_dot
        xdot[self.sl_L_job] = L_job_dot
        xdot[self.sl_omega_drum] = omega_drum_dot
        xdot[self.sl_R_roll] = R_roll_dot
        xdot[self.sl_L_w] = L_w_dot
        xdot[self.sl_sigma_web] = sigma_web_dot
        xdot[self.sl_T_drive] = T_drive_dot
        return xdot

    # ------------------------------------------------------------------
    # Output map g(x, u)
    # ------------------------------------------------------------------

    def outputs(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Measurement map y = g(x, u) (README §3.7-style selection)."""
        p = self.params
        T = x[self.sl_T].reshape(self.n_ext, self.n_zone)
        P = x[self.sl_P]
        Q_k = self._extruder_flow(T, P)
        Q_sum = np.sum(Q_k)
        delta_k = Q_k / Q_sum if Q_sum > 0 else np.full(self.n_ext, 1.0 / self.n_ext)

        y = np.zeros(self.n_outputs)
        y[self.sl_y_T_melt] = x[self.sl_T_melt]
        y[self.sl_y_P] = x[self.sl_P]
        y[self.sl_y_delta] = delta_k
        y[self.sl_y_N_ibc] = x[self.sl_N_ibc]
        y[self.sl_y_T_cool] = x[self.sl_T_cool]
        y[self.sl_y_v_haul] = x[self.sl_v_haul]
        y[self.sl_y_R_roll] = x[self.sl_R_roll]
        y[self.sl_y_L_rem] = p.L_target - x[self.sl_L_w]
        y[self.sl_y_sigma_web] = x[self.sl_sigma_web]
        return y

    # ------------------------------------------------------------------
    # Operating point (README §2)
    # ------------------------------------------------------------------

    def _solve_operating_point_step(
        self, u: np.ndarray, x_guess: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, str]:
        """
        Single Newton-Raphson solve of f(x, u) = 0 for one fixed input
        ``u``, warm-started from ``x_guess``. Factored out of
        ``solve_operating_point`` so it can be re-used, warm-started,
        across continuation steps.

        State components span many orders of magnitude (pressures
        ~1e7, radii ~0.1, lengths ~0/1e3, ...), so the residual is
        solved in variables scaled by the guess's own magnitude —
        unscaled root-finding over such a wide dynamic range converges
        poorly / diverges.

        An unconstrained solve can wander to wildly non-physical
        magnitudes (observed: >1e14) when it can't find an exact root,
        which then corrupts anything derived from the operating
        point's own scale (state normalisation in ``linearise()``,
        singular perturbation in ``eliminate_fast_states``). Rather
        than a bounded solver (much slower per call), the result is
        clamped to a generous multiple of the guess magnitude after
        the fact: cheap, and just as effective at preventing runaway
        divergence.
        """
        scale = np.maximum(np.abs(x_guess), 1.0)

        # L_job/L_w (job-length and per-winder roll-length counters) and
        # m_dos (accumulated dosed mass) are pure integrators: their xdot
        # doesn't depend on their own state value at all (only on other
        # states/inputs), so under normal operation they have no root and
        # their row/column of the Jacobian is structurally zero — making
        # the full-state Jacobian singular. That singularity was observed
        # to make the solver diverge wildly (>1e10x) even from an otherwise
        # -good guess. They're excluded from the root-solve here and held
        # at their initial-guess value instead; that's still a valid
        # linearisation point for them since their own dynamics don't
        # depend on their value.
        excluded = np.zeros(self.n_states, dtype=bool)
        excluded[self.sl_L_job] = True
        excluded[self.sl_L_w] = True
        excluded[self.sl_m_dos] = True
        free = ~excluded

        def scaled_residual(z_free: np.ndarray) -> np.ndarray:
            z = x_guess / scale
            z[free] = z_free
            return (self.dynamics(z * scale, u) * scale)[free]

        z0 = x_guess / scale
        result = root(
            scaled_residual, z0[free], method="hybr",
            options={"maxfev": 20000},
        )
        z_bound = 50.0
        z_free_clamped = np.clip(result.x, -z_bound, z_bound)
        if np.any(np.abs(result.x) > z_bound):
            logger.warning(
                "Operating-point solve diverged to a non-physical magnitude "
                "(max %.3gx the nominal guess) — clamping to +/-%.0fx.",
                float(np.max(np.abs(result.x))), z_bound,
            )
        z = x_guess / scale
        z[free] = z_free_clamped
        x = z * scale
        residual_norm = float(np.linalg.norm(self.dynamics(x, u)[free]))
        return x, residual_norm, bool(result.success), str(result.message)

    def solve_operating_point(
        self,
        u0: Optional[np.ndarray] = None,
        x0_guess: Optional[np.ndarray] = None,
        n_continuation_steps: int = 8,
    ) -> np.ndarray:
        """
        Newton-Raphson solve of f(x0, u0) = 0 for the steady-state x0
        (README §2).

        ``x0_guess`` (default ``nominal_state_guess()``) is only a
        mass/force-balanced guess relative to ``nominal_inputs()`` —
        when ``u0`` differs substantially from that (e.g. a real
        dataset's operating input), a single cold Newton solve from
        that guess can be far from ``u0``'s own equilibrium and fail
        to converge. Instead, ``u`` is stepped in
        ``n_continuation_steps`` increments from ``nominal_inputs()``
        to ``u0``, re-solving at each step warm-started from the
        previous step's (already-converged) solution — every
        individual solve then starts close to its own root.
        """
        u0 = self.nominal_inputs() if u0 is None else u0
        x0_guess = self.nominal_state_guess() if x0_guess is None else x0_guess

        u_start = self.nominal_inputs()
        if np.allclose(u0, u_start):
            x0, residual_norm, success, message = self._solve_operating_point_step(u0, x0_guess)
        else:
            x = x0_guess
            for alpha in np.linspace(0.0, 1.0, n_continuation_steps + 1)[1:]:
                u_step = u_start + alpha * (u0 - u_start)
                x, residual_norm, success, message = self._solve_operating_point_step(u_step, x)
            x0 = x

        if not success or residual_norm > 1.0:
            logger.warning(
                "Operating-point solve did not fully converge "
                "(unscaled residual norm=%.3e): %s", residual_norm, message,
            )
        return x0

    # ------------------------------------------------------------------
    # Linearisation (README §3) and discretisation (README §5.1)
    # ------------------------------------------------------------------

    def linearise(
        self, x0: np.ndarray, u0: np.ndarray, eps: float = 1e-6
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Numerical (finite-difference) Jacobian linearisation at (x0, u0).

        The state vector mixes raw physical units spanning many orders
        of magnitude (e.g. pressure ~1e7 Pa vs. bubble radius ~0.2 m),
        which makes the raw realisation extremely poorly conditioned
        for Gramian-based analysis (balanced truncation) downstream.
        The returned matrices therefore represent a similarity-
        transformed state x_scaled = x_tilde / scale (scale = |x0|,
        floored at 1) — a diagonal change of state basis that leaves
        the input-output behaviour identical but keeps every state
        component O(1) in magnitude.
        """
        n, m = self.n_states, self.n_inputs
        p = self.n_outputs
        f0 = self.dynamics(x0, u0)
        g0 = self.outputs(x0, u0)

        A_c = np.zeros((n, n))
        for j in range(n):
            dx = np.zeros(n)
            step = eps * max(1.0, abs(x0[j]))
            dx[j] = step
            A_c[:, j] = (self.dynamics(x0 + dx, u0) - f0) / step

        B_c = np.zeros((n, m))
        for j in range(m):
            du = np.zeros(m)
            step = eps * max(1.0, abs(u0[j]))
            du[j] = step
            B_c[:, j] = (self.dynamics(x0, u0 + du) - f0) / step

        C_c = np.zeros((p, n))
        for j in range(n):
            dx = np.zeros(n)
            step = eps * max(1.0, abs(x0[j]))
            dx[j] = step
            C_c[:, j] = (self.outputs(x0 + dx, u0) - g0) / step

        D_c = np.zeros((p, m))
        for j in range(m):
            du = np.zeros(m)
            step = eps * max(1.0, abs(u0[j]))
            du[j] = step
            D_c[:, j] = (self.outputs(x0, u0 + du) - g0) / step

        # Similarity transform to a well-conditioned, O(1)-scaled state basis.
        scale = np.maximum(np.abs(x0), 1.0)
        A_c = A_c * scale[None, :] / scale[:, None]
        B_c = B_c / scale[:, None]
        C_c = C_c * scale[None, :]

        return A_c, B_c, C_c, D_c

    @staticmethod
    def discretise(
        A_c: np.ndarray, B_c: np.ndarray, C_c: np.ndarray, D_c: np.ndarray, Ts: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Van Loan / matrix-exponential ZOH discretisation (README §5.1)."""
        n, m = A_c.shape[0], B_c.shape[1]
        M = np.zeros((n + m, n + m))
        M[:n, :n] = A_c
        M[:n, n:] = B_c
        expM = expm(M * Ts)
        A_d = expM[:n, :n]
        B_d = expM[:n, n:]
        return A_d, B_d, C_c, D_c

    @staticmethod
    def discretise_offset(A_c: np.ndarray, f0_scaled: np.ndarray, Ts: float) -> np.ndarray:
        """
        Discrete-time equivalent of a constant continuous forcing term
        ``f0_scaled`` (the residual ``dynamics(x0, u0)`` left over when
        ``x0`` isn't an exact equilibrium, in the same scaled state
        basis as ``linearise()``), via the same van Loan/matrix-
        exponential trick ``discretise()`` uses for ``B_c`` — treat it
        as an extra input column driven by a constant unit input.
        """
        n = A_c.shape[0]
        M = np.zeros((n + 1, n + 1))
        M[:n, :n] = A_c
        M[:n, n] = f0_scaled
        expM = expm(M * Ts)
        return expM[:n, n]

    # ------------------------------------------------------------------
    # Convenience: full pipeline to a StateSpaceModel
    # ------------------------------------------------------------------

    def to_state_space_model(
        self,
        Ts: float = 3.0,
        u0: Optional[np.ndarray] = None,
        eps: float = 1e-6,
        x0_guess_override: Optional[np.ndarray] = None,
        tau_threshold: Optional[float] = None,
        horizon_seconds: Optional[float] = None,
        min_spectral_gap: float = 3.0,
    ):
        """
        Solve for the operating point, linearise and discretise, and
        return a ``system_identification.StateSpaceModel`` — a drop-in
        replacement for the identified model consumed unchanged by
        model_reduction.py / estimation.py / mpc_controller.py.

        ``x0_guess_override`` lets callers warm-start the Newton-
        Raphson solve (e.g. from a previous grey-box iteration's
        equilibrium) instead of the default nominal guess.

        ``tau_threshold`` (if given) eliminates states faster than it
        via singular perturbation (``eliminate_fast_states``, README's
        "Time-scale separation" pipeline step) before discretisation —
        pass ``None`` (default) to keep every state at full fidelity.
        ``horizon_seconds`` (typically Np * Ts) enables the step-
        response validation check when elimination is applied.

        Returns
        -------
        (StateSpaceModel, x0, u0, diagnostics) — ``diagnostics`` is
        ``None`` unless ``tau_threshold`` was given.
        """
        from system_identification import StateSpaceModel  # local import avoids a cycle

        u0 = self.nominal_inputs() if u0 is None else u0
        x0 = self.solve_operating_point(u0, x0_guess_override)
        A_c, B_c, C_c, D_c = self.linearise(x0, u0, eps=eps)

        diagnostics = None
        if tau_threshold is not None:
            y_scale = np.maximum(np.abs(self.outputs(x0, u0)), 1.0)
            A_c, B_c, C_c, D_c, diagnostics = eliminate_fast_states(
                A_c, B_c, C_c, D_c, tau_threshold,
                horizon_seconds=horizon_seconds, min_spectral_gap=min_spectral_gap,
                u0=u0, y_scale=y_scale,
            )

        A_d, B_d, C_d, D_d = self.discretise(A_c, B_c, C_c, D_c, Ts)
        return StateSpaceModel(A=A_d, B=B_d, C=C_d, D=D_d, Ts=Ts), x0, u0, diagnostics

    # ------------------------------------------------------------------
    # Nonlinear simulation (for physics-based synthetic data)
    # ------------------------------------------------------------------
    def simulate_nonlinear(
        self, U: np.ndarray, x0: np.ndarray, dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """RK4 forward simulation of the nonlinear model over input sequence U."""
        T = U.shape[0]
        X = np.zeros((T, self.n_states))
        Y = np.zeros((T, self.n_outputs))
        x = x0.copy()
        for t in range(T):
            u = U[t]
            Y[t] = self.outputs(x, u)
            X[t] = x
            k1 = self.dynamics(x, u)
            k2 = self.dynamics(x + 0.5 * dt * k1, u)
            k3 = self.dynamics(x + 0.5 * dt * k2, u)
            k4 = self.dynamics(x + dt * k3, u)
            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return X, Y
