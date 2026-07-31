"""
main.py
=======
Pipeline orchestrator for the Co-Extrusion Blown Film Line
System Identification and MPC project.

Execution order
---------------
1.  Data loading / synthetic generation
2.  Model (N4SID identification + reduction) — loaded from cache if
    available, otherwise built fresh and cached for next time
3.  Kalman filter design
4.  Model validation on test set
5.  MPC controller construction — tuned weights loaded from cache if
    available; MPC weight optimisation is off by default (opt in with
    --optimise_weights)
6.  Closed-loop simulation
7.  Report generation

Caching
-------
Two pipeline artefacts are cached to disk (see ``persistence.py``) so
repeated runs don't have to redo expensive work:

  * the identified + reduced plant model (``--model_path``)
  * the tuned MPC weights (``--controller_path``)

Both default to "try load from disk; if missing, build fresh and save
for next time". Use ``--optimise_model`` / ``--optimise_weights`` to
force a fresh build even when a cache is present.

Accuracy gate
-------------
Every output must individually reach ``AccuracyConfig.min_r2``
(worst-case R², default 0.95) — once right after N4SID identification
(on training data) and again after the full reduction pipeline (on
held-out test data). A cached model is re-checked against the same
gate before it's trusted. On failure the corresponding model order is
escalated and retried, up to ``max_n_states`` / ``max_n_states_bt``;
if that ceiling is exhausted a ``ModelAccuracyError`` halts the
pipeline. Disable via --no_accuracy_gate; tune via --min_r2,
--max_n_states, --max_n_states_bt.

Sampling time
-------------
For real data, ``Ts`` defaults to the median interval between
consecutive timestamps in the data file, not a fixed constant — this
avoids a mismatch between the identified model's discrete-time step
and the physical clock (which would silently distort the MPC horizon
duration, ITAE scaling, and report timings). Pass --Ts to override;
a mismatch against the detected value then only produces a warning.

Usage
-----
    # With real data:
    python main.py --data path/to/data.csv

    # With synthetic data (default — omit --data):
    python main.py

    # Force a fresh model (ignore cached one):
    python main.py --optimise_model

    # Force MPC weight (re-)optimisation (ignore cached weights):
    python main.py --optimise_weights

Author : Blown Film MPC Project
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import logging
import os
import sys
import warnings

# ── Console encoding ─────────────────────────────────────────────────────────
# The report and log messages use Unicode box-drawing characters (─, █) and
# symbols (², ₀). On Windows, the default console code page (cp1252) cannot
# encode these and raises UnicodeEncodeError when printing. Force UTF-8 output
# streams so the pipeline runs identically on every platform.
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name)
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


import numpy as np

# ── Project modules ──────────────────────────────────────────────────────────
from accuracy import AccuracyResult, ModelAccuracyError, evaluate_accuracy
from config import (
    AccuracyConfig,
    DataConfig,
    IdentificationConfig,
    KalmanConfig,
    MPCConfig,
    ProjectConfig,
    ReductionConfig,
    SimulationConfig,
)
from data_manager import DataManager, IODataset
from estimation import KalmanFilter
from grey_box import GreyBoxIdentifier
from model_reduction import ModelReducer, ReducedModel
from mpc_controller import MPCController, MPCWeightOptimiser
from persistence import ControllerWeightsStore, ModelStore
from physical_model import FirstPrinciplesModel
from simulation import ClosedLoopSimulator, ModelValidator, SimulationResult
from system_identification import (
    ParameterOptimiser,
    StateSpaceModel,
    SubspaceIdentifier,
)
from utils import Plotter, ReportWriter

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


# =============================================================================
# Pipeline
# =============================================================================

class BlownFilmPipeline:
    """
    Orchestrates the full system identification and MPC pipeline.

    Parameters
    ----------
    cfg             : ProjectConfig (all sub-configs bundled)
    data_path       : path to real data file (None → use synthetic)
    output_dir      : directory for figures and report
    show_plots      : whether to display figures interactively
    model_path      : path to load/save the cached reduced model
    controller_path : path to load/save the cached MPC weights
    force_new_model : ignore any cached model and identify/reduce a
                      fresh one (see --optimise_model)
    ts_explicit     : whether the sampling time was explicitly passed
                      via --Ts (disables auto-detection from real-data
                      timestamps, other than a mismatch warning)
    """

    def __init__(
        self,
        cfg: ProjectConfig,
        data_path: str | None = None,
        output_dir: str = "outputs",
        show_plots: bool = True,
        model_path: str = os.path.join("saved", "reduced_model.pkl"),
        controller_path: str = os.path.join("saved", "mpc_weights.pkl"),
        force_new_model: bool = False,
        ts_explicit: bool = False,
    ) -> None:
        self._cfg        = cfg
        self._data_path  = data_path
        self._output_dir = output_dir

        self._model_path      = model_path
        self._controller_path = controller_path
        self._force_new_model = force_new_model
        self._ts_explicit     = ts_explicit

        os.makedirs(output_dir, exist_ok=True)

        self._plotter = Plotter(output_dir=output_dir, show=show_plots)
        self._report  = ReportWriter(output_dir=output_dir)

        # Pipeline artefacts (populated during run)
        self._dataset:  IODataset | None       = None
        self._ss_model: StateSpaceModel | None = None
        self._reduced:  ReducedModel | None    = None
        self._kf:       KalmanFilter | None    = None
        self._mpc:      MPCController | None   = None
        self._sim_result: SimulationResult | None = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute all pipeline stages in order."""
        np.random.seed(self._cfg.data.random_seed)

        self._stage_data()
        self._stage_model()
        self._stage_kalman()
        self._stage_validation()
        self._stage_mpc()
        self._stage_simulation()
        self._stage_report()

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def _stage_data(self) -> None:
        logger.info("━━━ STAGE 1: Data ━━━")
        dm = DataManager(cfg=self._cfg.data)

        if self._data_path is not None:
            self._dataset = dm.load_and_prepare(self._data_path)
        else:
            logger.info("No data file provided — using synthetic data.")
            self._dataset = dm.prepare_synthetic()

        self._reconcile_sampling_time()

        ds = self._dataset
        self._report.add_section(
            "1. Data Summary",
            f"Source        : {'synthetic' if self._data_path is None else self._data_path}\n"
            f"Train samples : {ds.n_train}\n"
            f"Test  samples : {ds.n_test}\n"
            f"Inputs  n_u   : {ds.n_inputs}\n"
            f"Outputs n_y   : {ds.n_outputs}\n"
            f"Sampling time : {self._cfg.data.sampling_time} s",
        )

    def _reconcile_sampling_time(self) -> None:
        """
        Reconcile the configured sampling time against the interval
        detected from real-data timestamps (``IODataset.detected_sampling_time``).

        If ``--Ts`` was not explicitly passed, the detected value is
        adopted automatically. If it was explicitly passed, a mismatch
        only triggers a warning — the user's choice is respected.
        """
        detected = self._dataset.detected_sampling_time
        if detected is None:
            return

        configured = self._cfg.data.sampling_time
        rel_diff = abs(detected - configured) / configured

        if not self._ts_explicit:
            if rel_diff > 0.01:
                logger.info(
                    "Auto-detected sampling time %.3gs from data timestamps "
                    "(overriding default %.3gs). Pass --Ts to override.",
                    detected, configured,
                )
                self._cfg = replace(
                    self._cfg, data=replace(self._cfg.data, sampling_time=detected)
                )
        elif rel_diff > 0.05:
            logger.warning(
                "Configured --Ts=%.3gs differs from the %.3gs interval "
                "detected in the data timestamps; using the configured "
                "value as requested.",
                configured, detected,
            )

    # ------------------------------------------------------------------
    def _stage_model(self) -> None:
        """
        Produce ``self._reduced``, either by loading a cached
        ``ReducedModel`` from disk or by running identification +
        reduction fresh (and caching the result for next time).

        A fresh build is forced when ``self._force_new_model`` is set
        (--optimise_model), when no valid cache is found, or when a
        cached model fails the accuracy gate (see ``AccuracyConfig``).
        """
        logger.info("━━━ STAGE 2-3: Model (Identification + Reduction) ━━━")

        cached = None
        if not self._force_new_model:
            cached = ModelStore.load(self._model_path)
        elif os.path.isfile(self._model_path):
            logger.info(
                "--optimise_model set — ignoring cached model at %s.",
                self._model_path,
            )

        if cached is not None:
            check = self._check_reduction_accuracy(cached)
            logger.info(check.summary())
            if check.passed or not self._cfg.accuracy.enabled:
                self._reduced = cached
                self._report.add_section(
                    "2-3. Model (Identification + Reduction)",
                    f"Loaded cached model from : {self._model_path}\n"
                    f"{self._reduced!r}\n"
                    f"Accuracy gate            : {check.summary()}",
                )
                return
            logger.warning(
                "Cached model at %s fails the accuracy gate — "
                "rebuilding from scratch (%s).",
                self._model_path, check.summary(),
            )

        self._stage_identification()
        self._stage_reduction()
        ModelStore.save(self._reduced, self._model_path)

    # ------------------------------------------------------------------
    def _check_reduction_accuracy(self, reduced: ReducedModel) -> AccuracyResult:
        """Evaluate a (candidate) reduced model against held-out test data."""
        ds        = self._dataset
        validator = ModelValidator(reduced_model=reduced, output_names=ds.output_cols)
        Y_pred    = validator.predict(ds.U_test)
        return evaluate_accuracy(
            ds.Y_test, Y_pred, ds.output_cols,
            stage="post-reduction",
            threshold=self._cfg.accuracy.min_r2,
            n_states=reduced.n_states,
        )

    # ------------------------------------------------------------------
    def _stage_identification(self) -> None:
        """
        Dispatch to the configured identification method
        (``IdentificationConfig.method``): ``"physical"`` (default) —
        grey-box parameter optimisation of a FirstPrinciplesModel — or
        ``"n4sid"`` — the original black-box subspace identification.
        """
        logger.info("━━━ STAGE 2: System Identification ━━━")
        if self._cfg.identification.method == "physical":
            self._stage_identification_physical()
        else:
            self._stage_identification_n4sid()

    def _stage_identification_physical(self) -> None:
        """
        Grey-box identification: optimise FirstPrinciplesModel physical
        parameter-group scales against the training data (grey_box.py).
        The model's state count is fixed by ``PhysicalModelConfig``
        multiplicities (no "order" to escalate), so the accuracy gate
        is checked once — a failure raises ``ModelAccuracyError``
        directly instead of retrying at a different order.
        """
        ds       = self._dataset
        acc_cfg  = self._cfg.accuracy
        phys_cfg = self._cfg.physical_model

        phys_model = FirstPrinciplesModel(
            n_extruders=phys_cfg.n_extruders, n_zones=phys_cfg.n_zones,
            n_components=phys_cfg.n_components, n_die_zones=phys_cfg.n_die_zones,
            n_ibc=phys_cfg.n_ibc, n_winders=phys_cfg.n_winders,
        )
        horizon_seconds = self._cfg.mpc.prediction_horizon * self._cfg.data.sampling_time
        identifier = GreyBoxIdentifier(
            model=phys_model, Ts=self._cfg.data.sampling_time, cfg=phys_cfg,
            horizon_seconds=horizon_seconds,
        )
        model, gb_result, U_model = identifier.fit(ds.U_train, ds.Y_train)

        check = evaluate_accuracy(
            ds.Y_train, model.simulate(U_model), ds.output_cols,
            stage="post-identification",
            threshold=acc_cfg.min_r2,
            n_states=model.n_states,
        )
        logger.info(check.summary())
        if acc_cfg.enabled and not check.passed:
            raise ModelAccuracyError(
                "Grey-box physical identification could not reach the "
                f"required accuracy (worst-case R²={check.worst_r2:.4f} on "
                f"output {check.worst_output!r}, need >={acc_cfg.min_r2:.4f}) "
                f"at n_states={model.n_states}. This path has no order to "
                "escalate — consider --identification_method n4sid, richer "
                "excitation data, or raising PhysicalModelConfig.grey_box_max_iter."
            )

        self._ss_model = model
        avg_r2 = float(np.mean(check.per_output_r2))
        logger.info("Identified model: %r", model)

        sp = identifier.last_diagnostics
        sp_summary = "disabled"
        if sp is not None:
            if sp["applied"]:
                sp_summary = (
                    f"{sp['n_fast']} fast state(s) eliminated, {sp['n_slow']} retained "
                    f"(spectral gap {sp['spectral_gap']:.2f}x, DC-gain error "
                    f"{sp['dc_gain_rel_error']:.1%}"
                    + (f", horizon error {sp['horizon_rel_error']:.1%}" if sp["horizon_rel_error"] is not None else "")
                    + ")"
                )
            else:
                sp_summary = "not applied (" + "; ".join(sp["warnings"]) + ")" if sp["warnings"] else "not applied"

        self._report.add_section(
            "2. System Identification (Grey-Box / Physical)",
            f"Model order       : {model.n_states}\n"
            f"Stable            : {model.is_stable}\n"
            f"Spectral rho      : {model.spectral_radius:.4f}\n"
            f"Training R² (avg) : {avg_r2:.4f}\n"
            f"Grey-box cost     : {gb_result.cost:.6g}\n"
            f"Grey-box success  : {gb_result.success} "
            f"({gb_result.n_iterations} evaluations)\n"
            f"Singular pert.    : {sp_summary}\n"
            f"Accuracy gate     : {check.summary()}",
        )

    def _stage_identification_n4sid(self) -> None:
        """
        Run N4SID (+ optional parameter refinement). If the accuracy
        gate is enabled, escalate ``n_states`` and retry until every
        output's training R² clears ``AccuracyConfig.min_r2``, up to
        ``max_n_states``.
        """
        ds      = self._dataset
        id_cfg  = self._cfg.identification
        acc_cfg = self._cfg.accuracy

        n          = id_cfg.n_states
        prev_n: int | None = None
        identifier: SubspaceIdentifier | None = None
        model: StateSpaceModel | None = None
        check: AccuracyResult | None = None

        while True:
            attempt_cfg = replace(id_cfg, n_states=n)
            identifier  = SubspaceIdentifier(
                cfg=attempt_cfg, Ts=self._cfg.data.sampling_time
            )
            identifier.fit(ds.U_train, ds.Y_train)
            model = identifier.model

            if attempt_cfg.optimise_params:
                model = ParameterOptimiser(
                    model=model, cfg=attempt_cfg
                ).optimise(ds.U_train, ds.Y_train)

            check = evaluate_accuracy(
                ds.Y_train, model.simulate(ds.U_train), ds.output_cols,
                stage="post-identification",
                threshold=acc_cfg.min_r2,
                n_states=model.n_states,
            )
            logger.info(check.summary())

            if not acc_cfg.enabled or check.passed:
                break
            if model.n_states >= acc_cfg.max_n_states or model.n_states == prev_n:
                raise ModelAccuracyError(
                    "System identification could not reach the required "
                    f"accuracy (worst-case R²={check.worst_r2:.4f} on output "
                    f"{check.worst_output!r}, need >={acc_cfg.min_r2:.4f}) "
                    f"even at n_states={model.n_states} "
                    f"(cap max_n_states={acc_cfg.max_n_states}). Consider "
                    "richer/faster excitation data, raising --max_n_states, "
                    "or increasing n_block_rows in IdentificationConfig."
                )
            prev_n = model.n_states
            n      = min(model.n_states + acc_cfg.n_states_step, acc_cfg.max_n_states)
            logger.warning(
                "Accuracy gate failed post-identification — escalating "
                "n_states to %d and retrying.", n,
            )

        self._ss_model = model

        self._plotter.plot_singular_values(
            sv=identifier.singular_values,
            selected_order=model.n_states,
        )

        avg_r2 = float(np.mean(check.per_output_r2))
        logger.info("Identified model: %r", model)
        self._report.add_section(
            "2. System Identification (N4SID)",
            f"Model order       : {model.n_states}\n"
            f"Stable            : {model.is_stable}\n"
            f"Spectral rho      : {model.spectral_radius:.4f}\n"
            f"Training R² (avg) : {avg_r2:.4f}\n"
            f"Param opt         : {id_cfg.optimise_params}\n"
            f"Accuracy gate     : {check.summary()}",
        )


    # ------------------------------------------------------------------
    def _stage_reduction(self) -> None:
        """
        Run balanced truncation + POD/Galerkin reduction. If the
        accuracy gate is enabled, escalate ``n_states_bt`` and retry
        until every output's test R² clears ``AccuracyConfig.min_r2``,
        up to ``max_n_states_bt`` (and the full model order).
        """
        logger.info("━━━ STAGE 3: Model Order Reduction ━━━")
        red_cfg = self._cfg.reduction
        acc_cfg = self._cfg.accuracy

        n_bt = red_cfg.n_states_bt
        prev_n: int | None = None
        reduced: ReducedModel | None = None
        check: AccuracyResult | None = None

        while True:
            attempt_cfg = replace(red_cfg, n_states_bt=n_bt)
            reducer     = ModelReducer(
                cfg=attempt_cfg, Ts=self._cfg.data.sampling_time
            )
            reduced = reducer.reduce(self._ss_model)

            check = self._check_reduction_accuracy(reduced)
            logger.info(check.summary())

            if not acc_cfg.enabled or check.passed:
                break
            if (
                reduced.n_states >= acc_cfg.max_n_states_bt
                or reduced.n_states >= self._ss_model.n_states - 1
                or reduced.n_states == prev_n
            ):
                raise ModelAccuracyError(
                    "Model order reduction could not retain the required "
                    f"accuracy (worst-case test R²={check.worst_r2:.4f} on "
                    f"output {check.worst_output!r}, need "
                    f">={acc_cfg.min_r2:.4f}) even at reduced order="
                    f"{reduced.n_states} (full order="
                    f"{self._ss_model.n_states}, cap max_n_states_bt="
                    f"{acc_cfg.max_n_states_bt}). The full-order model "
                    "already passed this threshold on training data, so "
                    "consider: raising --n_id / --max_n_states further, "
                    "checking for train/test overfitting, or raising "
                    "pod_energy_tolerance."
                )
            prev_n = reduced.n_states
            n_bt   = min(
                reduced.n_states + acc_cfg.n_states_bt_step,
                acc_cfg.max_n_states_bt,
            )
            logger.warning(
                "Accuracy gate failed post-reduction — escalating "
                "n_states_bt to %d and retrying.", n_bt,
            )

        self._reduced = reduced

        self._plotter.plot_hsv(
            hsv=reduced.hsv,
            truncation_order=reduced.n_states,
        )

        self._report.add_section(
            "3. Model Order Reduction",
            f"Full order    : {self._ss_model.n_states}\n"
            f"Reduced order : {reduced.n_states}\n"
            f"Augmented dim : {reduced.n_aug}\n"
            f"Reduction     : "
            f"{(1 - reduced.n_states / self._ss_model.n_states) * 100:.1f}%\n"
            f"Ts            : {reduced.Ts} s\n"
            f"Accuracy gate : {check.summary()}",
        )

    # ------------------------------------------------------------------
    def _stage_kalman(self) -> None:
        logger.info("━━━ STAGE 4: Kalman Filter ━━━")
        self._kf = KalmanFilter(
            reduced_model=self._reduced,
            cfg=self._cfg.kalman,
        ).initialise()

        self._report.add_section(
            "4. Kalman Filter",
            f"Augmented state dim : {self._reduced.n_aug}\n"
            f"Process noise Q     : {self._cfg.kalman.process_noise_scale:.2e} × I\n"
            f"Measurement noise R : {self._cfg.kalman.measurement_noise_scale:.2e} × I\n"
            f"Gain shape          : {self._kf.gain.shape}",
        )

    # ------------------------------------------------------------------
    def _stage_validation(self) -> None:
        logger.info("━━━ STAGE 5: Model Validation ━━━")
        ds        = self._dataset
        validator = ModelValidator(
            reduced_model=self._reduced,
            output_names=ds.output_cols,
        )
        metrics   = validator.validate(ds.U_test, ds.Y_test)
        Y_pred    = validator.predict(ds.U_test)

        self._plotter.plot_validation(
            Y_true=ds.Y_test,
            Y_pred=Y_pred,
            output_names=ds.output_cols,
        )
        self._plotter.plot_residuals(
            Y_true=ds.Y_test,
            Y_pred=Y_pred,
            output_names=ds.output_cols,
        )
        self._plotter.plot_metrics_heatmap(metrics)

        avg_r2    = np.mean([m.r2    for m in metrics])
        avg_nrmse = np.mean([m.nrmse for m in metrics])

        self._report.add_section(
            "5. Model Validation (Test Set)",
            f"Avg R²    : {avg_r2:.4f}\n"
            f"Avg NRMSE : {avg_nrmse:.4f}\n\n"
            + ReportWriter.format_metrics_table(metrics),
        )

    # ------------------------------------------------------------------
    def _stage_mpc(self) -> None:
        """
        Build the MPC controller with fixed config weights, then
        either apply cached tuned weights, run the weight optimiser
        fresh (--optimise_weights), or fall back to the fixed config
        weights untouched — whichever applies.
        """
        logger.info("━━━ STAGE 6: MPC Design ━━━")
        cfg = self._cfg.mpc
        ds  = self._dataset

        self._mpc = MPCController(
            reduced_model=self._reduced,
            cfg=cfg,
        )

        if cfg.optimise_weights:
            weight_opt = MPCWeightOptimiser(
                controller=self._mpc,
                kalman_filter=self._kf,
                reduced_model=self._reduced,
                U_val=ds.U_test,
                Y_val=ds.Y_test,
                cfg=cfg,
            )
            Q_opt, R_opt = weight_opt.optimise()
            self._mpc.set_weights(Q_opt, R_opt)
            self._plotter.plot_weight_optimisation(weight_opt.cost_history)

            ControllerWeightsStore.save(Q_opt, R_opt, self._controller_path)
            weights_source = "optimised now (Nelder-Mead) and cached"
        else:
            cached = ControllerWeightsStore.load(
                self._controller_path, n_y=ds.n_outputs, n_u=ds.n_inputs,
            )
            if cached is not None:
                self._mpc.set_weights(cached.Q, cached.R)
                weights_source = f"loaded from cache ({self._controller_path})"
                if self._force_new_model:
                    logger.warning(
                        "Using cached MPC weights tuned against a previous "
                        "model (--optimise_model was set) — consider also "
                        "passing --optimise_weights to re-tune them."
                    )
            else:
                weights_source = "fixed config defaults (not optimised)"

        self._report.add_section(
            "6. MPC Configuration",
            f"Prediction horizon Np : {cfg.prediction_horizon}\n"
            f"Control horizon Nc    : {cfg.control_horizon}\n"
            f"Sampling time Ts      : {self._cfg.data.sampling_time} s\n"
            f"Prediction window     : "
            f"{cfg.prediction_horizon * self._cfg.data.sampling_time:.0f} s\n"
            f"QP size               : "
            f"{cfg.control_horizon * ds.n_inputs} variables\n"
            f"Solver                : OSQP (warm-start)\n"
            f"MPC weights           : {weights_source}",
        )

    # ------------------------------------------------------------------
    def _stage_simulation(self) -> None:
        logger.info("━━━ STAGE 7: Closed-Loop Simulation ━━━")
        cfg = self._cfg.simulation
        ds  = self._dataset
        n_y = self._reduced.n_outputs

        # Build step-change reference trajectory
        y_ref = self._build_reference(cfg, n_y)

        simulator = ClosedLoopSimulator(
            controller=self._mpc,
            kalman_filter=self._kf,
            reduced_model=self._reduced,
            cfg=cfg,
        )
        self._sim_result = simulator.run(y_ref=y_ref)

        self._plotter.plot_closed_loop(
            result=self._sim_result,
            output_names=ds.output_cols,
            input_names=ds.input_cols,
        )

        r = self._sim_result
        self._report.add_section(
            "7. Closed-Loop Simulation",
            f"Steps         : {r.n_steps}\n"
            f"Duration      : {r.n_steps * self._cfg.data.sampling_time:.0f} s\n"
            f"ISE           : {r.ise:.4f}\n"
            f"ITAE          : {r.itae:.4f}\n"
            f"Mean MSE      : {r.mean_mse:.6f}\n"
            f"Mean QP cost  : {np.nanmean(r.costs):.4f}",
        )

    # ------------------------------------------------------------------
    def _stage_report(self) -> None:
        logger.info("━━━ STAGE 8: Report ━━━")
        self._report.print_report()
        self._report.save("blown_film_sysid_mpc_report.txt")
        logger.info("Pipeline complete. Outputs in: %s", self._output_dir)

    # ------------------------------------------------------------------
    @staticmethod
    def _build_reference(
        cfg: SimulationConfig,
        n_y: int,
    ) -> np.ndarray:
        """Construct a step-change reference trajectory."""
        T     = cfg.n_steps
        y_ref = np.zeros((T, n_y))
        t1, t2, t3 = cfg.ref_step_time_1, cfg.ref_step_time_2, cfg.ref_step_time_3
        a1, a2     = cfg.ref_amplitude_1,  cfg.ref_amplitude_2

        y_ref[t1:t2, : min(3, n_y)] = a1
        y_ref[t2:t3, : min(3, n_y)] = a2
        return y_ref


# =============================================================================
# CLI entry point
# =============================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Blown Film Line: System Identification + MPC",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to CSV/Excel data file (omit for synthetic data)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs",
        help="Directory for figures and report",
    )
    parser.add_argument(
        "--n_id", type=int, default=145,
        help="N4SID model order (only used when --identification_method n4sid)",
    )
    parser.add_argument(
        "--identification_method", type=str, default="physical",
        choices=["physical", "n4sid"],
        help=(
            "'physical': grey-box — optimise FirstPrinciplesModel physical "
            "parameter scales against the data (see grey_box.py). "
            "'n4sid': black-box N4SID subspace identification (+ optional "
            "L-BFGS-B refinement), the previous default."
        ),
    )
    parser.add_argument(
        "--n_red", type=int, default=22,
        help="Target order after balanced truncation",
    )
    parser.add_argument(
        "--min_r2", type=float, default=0.95,
        help="Required worst-case per-output R² (accuracy gate)",
    )
    parser.add_argument(
        "--max_n_states", type=int, default=100,
        help="Ceiling for automatic N4SID order escalation on gate failure",
    )
    parser.add_argument(
        "--max_n_states_bt", type=int, default=50,
        help="Ceiling for automatic reduction-order escalation on gate failure",
    )
    parser.add_argument(
        "--no_accuracy_gate", action="store_true",
        help="Disable the minimum-accuracy gate entirely (debugging only)",
    )
    parser.add_argument(
        "--Ts", type=float, default=None,
        help=(
            "Sampling time in seconds. Default: auto-detected from the "
            "data timestamps for real data (falls back to 3.0s if that "
            "fails), or 3.0s for synthetic data. Passing this explicitly "
            "overrides auto-detection."
        ),
    )
    parser.add_argument(
        "--Np", type=int, default=20,
        help="MPC prediction horizon",
    )
    parser.add_argument(
        "--Nc", type=int, default=8,
        help="MPC control horizon",
    )
    parser.add_argument(
        "--T_sim", type=int, default=300,
        help="Closed-loop simulation steps",
    )
    parser.add_argument(
        "--no_param_opt", action="store_true",
        help="Skip parameter optimisation after N4SID",
    )
    parser.add_argument(
        "--optimise_weights", action="store_true",
        help=(
            "Run MPC weight optimisation (Nelder-Mead), ignoring any "
            "cached weights, and cache the result. Off by default: "
            "loads cached weights if available, otherwise uses fixed "
            "config weights."
        ),
    )
    parser.add_argument(
        "--optimise_model", action="store_true",
        help=(
            "Force fresh identification + reduction, ignoring any "
            "cached model, and cache the result. Off by default: "
            "loads the cached model if available."
        ),
    )
    parser.add_argument(
        "--model_path", type=str,
        default=os.path.join("saved", "reduced_model.pkl"),
        help="Path to load/save the cached reduced model",
    )
    parser.add_argument(
        "--controller_path", type=str,
        default=os.path.join("saved", "mpc_weights.pkl"),
        help="Path to load/save the cached MPC weights",
    )
    parser.add_argument(
        "--no_show", action="store_true",
        help="Do not display figures interactively",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    cfg = ProjectConfig(
        data=DataConfig(sampling_time=args.Ts if args.Ts is not None else 3.0),
        identification=IdentificationConfig(
            n_states=args.n_id,
            optimise_params=not args.no_param_opt,
            method=args.identification_method,
        ),
        reduction=ReductionConfig(n_states_bt=args.n_red),
        accuracy=AccuracyConfig(
            min_r2=args.min_r2,
            enabled=not args.no_accuracy_gate,
            max_n_states=args.max_n_states,
            max_n_states_bt=args.max_n_states_bt,
        ),
        kalman=KalmanConfig(),
        mpc=MPCConfig(
            prediction_horizon=args.Np,
            control_horizon=args.Nc,
            optimise_weights=args.optimise_weights,
        ),
        simulation=SimulationConfig(n_steps=args.T_sim),
        output_dir=args.output_dir,
    )

    pipeline = BlownFilmPipeline(
        cfg=cfg,
        data_path=args.data,
        output_dir=args.output_dir,
        show_plots=not args.no_show,
        model_path=args.model_path,
        controller_path=args.controller_path,
        force_new_model=args.optimise_model,
        ts_explicit=args.Ts is not None,
    )

    try:
        pipeline.run()
    except ModelAccuracyError as exc:
        logger.error("Accuracy gate failed — pipeline halted.\n%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()