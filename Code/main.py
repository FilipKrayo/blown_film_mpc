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
from config import (
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
from model_reduction import ModelReducer, ReducedModel
from mpc_controller import MPCController, MPCWeightOptimiser
from persistence import ControllerWeightsStore, ModelStore
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
    ) -> None:
        self._cfg        = cfg
        self._data_path  = data_path
        self._output_dir = output_dir

        self._model_path      = model_path
        self._controller_path = controller_path
        self._force_new_model = force_new_model

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

    # ------------------------------------------------------------------
    def _stage_model(self) -> None:
        """
        Produce ``self._reduced``, either by loading a cached
        ``ReducedModel`` from disk or by running identification +
        reduction fresh (and caching the result for next time).

        A fresh build is forced when ``self._force_new_model`` is set
        (--optimise_model) or when no valid cache is found.
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
            self._reduced = cached
            self._report.add_section(
                "2-3. Model (Identification + Reduction)",
                f"Loaded cached model from : {self._model_path}\n"
                f"{self._reduced!r}",
            )
            return

        self._stage_identification()
        self._stage_reduction()
        ModelStore.save(self._reduced, self._model_path)

    # ------------------------------------------------------------------
    def _stage_identification(self) -> None:
        logger.info("━━━ STAGE 2: System Identification ━━━")
        ds  = self._dataset
        cfg = self._cfg.identification

        identifier = SubspaceIdentifier(cfg=cfg, Ts=self._cfg.data.sampling_time)
        identifier.fit(ds.U_train, ds.Y_train)

        self._plotter.plot_singular_values(
            sv=identifier.singular_values,
            selected_order=cfg.n_states,
        )

        model = identifier.model

        if cfg.optimise_params:
            optimiser = ParameterOptimiser(model=model, cfg=cfg)
            model     = optimiser.optimise(ds.U_train, ds.Y_train)

        self._ss_model = model

        y_hat = model.simulate(ds.U_train)
        from sklearn.metrics import r2_score
        r2_tr = float(r2_score(ds.Y_train, y_hat))

        logger.info("Identified model: %r", model)
        self._report.add_section(
            "2. System Identification (N4SID)",
            f"Model order   : {model.n_states}\n"
            f"Stable        : {model.is_stable}\n"
            f"Spectral rho  : {model.spectral_radius:.4f}\n"
            f"Training R²   : {r2_tr:.4f}\n"
            f"Param opt     : {cfg.optimise_params}",
        )

    # ------------------------------------------------------------------
    def _stage_reduction(self) -> None:
        logger.info("━━━ STAGE 3: Model Order Reduction ━━━")
        reducer = ModelReducer(
            cfg=self._cfg.reduction,
            Ts=self._cfg.data.sampling_time,
        )
        self._reduced = reducer.reduce(self._ss_model)

        self._plotter.plot_hsv(
            hsv=self._reduced.hsv,
            truncation_order=self._reduced.n_states,
        )

        self._report.add_section(
            "3. Model Order Reduction",
            f"Full order    : {self._ss_model.n_states}\n"
            f"Reduced order : {self._reduced.n_states}\n"
            f"Augmented dim : {self._reduced.n_aug}\n"
            f"Reduction     : "
            f"{(1 - self._reduced.n_states / self._ss_model.n_states) * 100:.1f}%\n"
            f"Ts            : {self._reduced.Ts} s",
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
        "--n_id", type=int, default=20,
        help="N4SID model order",
    )
    parser.add_argument(
        "--n_red", type=int, default=12,
        help="Target order after balanced truncation",
    )
    parser.add_argument(
        "--Ts", type=float, default=3.0,
        help="Sampling time in seconds",
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
        data=DataConfig(sampling_time=args.Ts),
        identification=IdentificationConfig(
            n_states=args.n_id,
            optimise_params=not args.no_param_opt,
        ),
        reduction=ReductionConfig(n_states_bt=args.n_red),
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
    )
    pipeline.run()


if __name__ == "__main__":
    main()