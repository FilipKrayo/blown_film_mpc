"""
utils.py
========
Plotting, console reporting and miscellaneous helper utilities.

Classes
-------
Plotter
    Generates all diagnostic and results figures.
ReportWriter
    Builds and saves a structured plain-text summary report.

Author : Blown Film MPC Project
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns

from simulation import SimulationResult, ValidationMetrics

logger = logging.getLogger(__name__)

# Global plot style
plt.rcParams.update({
    "figure.dpi":      150,
    "axes.grid":       True,
    "grid.alpha":      0.3,
    "axes.titlesize":  10,
    "axes.labelsize":  9,
    "legend.fontsize": 8,
    "font.family":     "sans-serif",
})


# ---------------------------------------------------------------------------
# Plotter
# ---------------------------------------------------------------------------

class Plotter:
    """
    Centralised figure generation for the blown film MPC project.

    All figures are saved to ``output_dir`` and optionally displayed.

    Parameters
    ----------
    output_dir : directory for saved figures
    show       : whether to call plt.show() after each figure
    """

    def __init__(
        self,
        output_dir: str = "outputs",
        show: bool = True,
    ) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self._dir  = output_dir
        self._show = show

    # ------------------------------------------------------------------
    def plot_singular_values(
        self,
        sv: np.ndarray,
        selected_order: int,
        title: str = "Hankel Singular Values — Model Order Selection",
        filename: str = "singular_values.png",
    ) -> None:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.semilogy(sv[:60], "bo-", markersize=5, linewidth=1)
        ax.axvline(
            selected_order, color="r", linestyle="--",
            label=f"Selected order n={selected_order}",
        )
        ax.set_xlabel("Index")
        ax.set_ylabel("Singular Value (log scale)")
        ax.set_title(title, fontweight="bold")
        ax.legend()
        self._save_and_show(fig, filename)

    # ------------------------------------------------------------------
    def plot_hsv(
        self,
        hsv: np.ndarray,
        truncation_order: int,
        filename: str = "hsv_balanced_truncation.png",
    ) -> None:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.semilogy(hsv, "gs-", markersize=5, linewidth=1)
        ax.axvline(
            truncation_order, color="r", linestyle="--",
            label=f"Truncation at r={truncation_order}",
        )
        ax.set_xlabel("State Index")
        ax.set_ylabel("Hankel Singular Value")
        ax.set_title("Balanced Truncation — Hankel Singular Values",
                     fontweight="bold")
        ax.legend()
        self._save_and_show(fig, filename)

    # ------------------------------------------------------------------
    def plot_validation(
        self,
        Y_true: np.ndarray,
        Y_pred: np.ndarray,
        output_names: List[str],
        n_plot: int = 6,
        filename: str = "validation_predictions.png",
    ) -> None:
        n_out = min(n_plot, Y_true.shape[1])
        fig, axes = plt.subplots(n_out, 1,
                                 figsize=(14, 3 * n_out),
                                 sharex=True)
        axes = np.atleast_1d(axes)
        t    = np.arange(Y_true.shape[0])

        for i, ax in enumerate(axes):
            ax.plot(t, Y_true[:, i], "b-",  lw=1.2,
                    label="Measured", alpha=0.85)
            ax.plot(t, Y_pred[:, i], "r--", lw=1.2,
                    label="Predicted", alpha=0.9)
            name = output_names[i] if i < len(output_names) else f"y_{i}"
            ax.set_ylabel(name[-30:])
            ax.legend(loc="upper right")

        axes[-1].set_xlabel("Time Step (k)")
        fig.suptitle("Model Validation — Test Set",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        self._save_and_show(fig, filename)

    # ------------------------------------------------------------------
    def plot_residuals(
        self,
        Y_true: np.ndarray,
        Y_pred: np.ndarray,
        output_names: List[str],
        n_plot: int = 4,
        filename: str = "residuals.png",
    ) -> None:
        n_out = min(n_plot, Y_true.shape[1])
        fig, axes = plt.subplots(1, n_out, figsize=(14, 4))
        axes = np.atleast_1d(axes)

        for i, ax in enumerate(axes):
            res = Y_true[:, i] - Y_pred[:, i]
            ax.hist(res, bins=40, color="steelblue",
                    edgecolor="white", alpha=0.85)
            ax.axvline(0, color="r", linestyle="--")
            name = output_names[i] if i < len(output_names) else f"y_{i}"
            ax.set_title(f"Residuals: {name[-20:]}")
            ax.set_xlabel("Error")

        fig.suptitle("Residual Distributions", fontweight="bold")
        plt.tight_layout()
        self._save_and_show(fig, filename)

    # ------------------------------------------------------------------
    def plot_metrics_heatmap(
        self,
        metrics: List[ValidationMetrics],
        filename: str = "metrics_heatmap.png",
    ) -> None:
        data = {
            "R²":    [m.r2    for m in metrics],
            "NRMSE": [m.nrmse for m in metrics],
            "MSE":   [m.mse   for m in metrics],
        }
        names = [m.output_name[-30:] for m in metrics]
        df    = __import__("pandas").DataFrame(data, index=names)

        fig, axes = plt.subplots(1, 3, figsize=(15, max(4, len(df) // 3)))
        cmaps = {"R²": "RdYlGn", "NRMSE": "RdYlGn_r", "MSE": "RdYlGn_r"}
        for ax, col in zip(axes, ["R²", "NRMSE", "MSE"]):
            sns.heatmap(
                df[[col]], ax=ax, annot=True, fmt=".3f",
                cmap=cmaps[col], linewidths=0.5,
            )
            ax.set_title(col, fontweight="bold")

        fig.suptitle("Validation Metrics", fontsize=13, fontweight="bold")
        plt.tight_layout()
        self._save_and_show(fig, filename)

    # ------------------------------------------------------------------
    def plot_closed_loop(
        self,
        result: SimulationResult,
        output_names: List[str],
        input_names: List[str],
        n_out_plot: int = 6,
        n_in_plot: int = 4,
    ) -> None:
        self._plot_output_tracking(result, output_names, n_out_plot)
        self._plot_control_inputs(result, input_names, n_in_plot)
        self._plot_mpc_cost(result)

    # ------------------------------------------------------------------
    def plot_weight_optimisation(
        self,
        history: List[float],
        filename: str = "mpc_weight_optimisation.png",
    ) -> None:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(history, "b-o", markersize=4, linewidth=1)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Closed-Loop Cost")
        ax.set_title("MPC Weight Optimisation History", fontweight="bold")
        self._save_and_show(fig, filename)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _plot_output_tracking(
        self,
        result: SimulationResult,
        names: List[str],
        n_plot: int,
    ) -> None:
        n_out = min(n_plot, result.Y_measured.shape[1])
        t     = np.arange(result.n_steps)
        fig, axes = plt.subplots(n_out, 1,
                                 figsize=(14, 3 * n_out),
                                 sharex=True)
        axes = np.atleast_1d(axes)
        for i, ax in enumerate(axes):
            ax.plot(t, result.Y_measured[:, i], "b-",
                    lw=1.2, label="Measured", alpha=0.85)
            ax.plot(t, result.references[:, i], "g--",
                    lw=1.5, label="Reference")
            name = names[i] if i < len(names) else f"y_{i}"
            ax.set_ylabel(name[-25:])
            ax.legend(loc="upper right")
        axes[-1].set_xlabel("Time Step (k)")
        fig.suptitle("MPC Closed-Loop: Output Tracking",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        self._save_and_show(fig, "mpc_output_tracking.png")

    def _plot_control_inputs(
        self,
        result: SimulationResult,
        names: List[str],
        n_plot: int,
    ) -> None:
        n_in = min(n_plot, result.U_applied.shape[1])
        t    = np.arange(result.n_steps)
        fig, axes = plt.subplots(n_in, 1,
                                 figsize=(14, 2.5 * n_in),
                                 sharex=True)
        axes = np.atleast_1d(axes)
        for i, ax in enumerate(axes):
            ax.step(t, result.U_applied[:, i], "r-",
                    lw=1.2, where="post")
            name = names[i] if i < len(names) else f"u_{i}"
            ax.set_ylabel(name[-25:])
        axes[-1].set_xlabel("Time Step (k)")
        fig.suptitle("MPC Control Inputs",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        self._save_and_show(fig, "mpc_inputs.png")

    def _plot_mpc_cost(self, result: SimulationResult) -> None:
        t   = np.arange(result.n_steps)
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(t, result.costs, "k-", lw=1.0, alpha=0.8)
        ax.set_xlabel("Time Step (k)")
        ax.set_ylabel("QP Objective Value")
        ax.set_title("MPC Objective per Step", fontweight="bold")
        self._save_and_show(fig, "mpc_cost.png")

    def _save_and_show(self, fig: plt.Figure, filename: str) -> None:
        path = os.path.join(self._dir, filename)
        fig.savefig(path, bbox_inches="tight")
        logger.info("Figure saved: %s", path)
        if self._show:
            plt.show()
        plt.close(fig)


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

class ReportWriter:
    """
    Builds and saves a structured plain-text summary report.

    Parameters
    ----------
    output_dir : directory for the report file
    """

    _SEPARATOR = "=" * 72

    def __init__(self, output_dir: str = "outputs") -> None:
        os.makedirs(output_dir, exist_ok=True)
        self._dir      = output_dir
        self._sections: List[tuple] = []

    # ------------------------------------------------------------------
    def add_section(self, title: str, content: str) -> None:
        """Append a titled section to the report."""
        self._sections.append((title, content))

    # ------------------------------------------------------------------
    def print_report(self) -> None:
        """Print the full report to stdout."""
        print(f"\n{self._SEPARATOR}")
        print("  CO-EXTRUSION BLOWN FILM LINE — SYSTEM ID & MPC REPORT")
        print(self._SEPARATOR)
        for title, content in self._sections:
            print(f"\n{'─' * 60}")
            print(f"  {title}")
            print(f"{'─' * 60}")
            print(content)
        print(f"\n{self._SEPARATOR}\n")

    # ------------------------------------------------------------------
    def save(self, filename: str = "report.txt") -> None:
        """Save the report to a text file."""
        path = os.path.join(self._dir, filename)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("CO-EXTRUSION BLOWN FILM LINE — SYSTEM ID & MPC REPORT\n")
            fh.write(self._SEPARATOR + "\n")
            for title, content in self._sections:
                fh.write(f"\n{'─' * 60}\n  {title}\n{'─' * 60}\n")
                fh.write(content + "\n")
        logger.info("Report saved: %s", path)

    # ------------------------------------------------------------------
    @staticmethod
    def format_metrics_table(metrics: List[ValidationMetrics]) -> str:
        """Return a formatted string table of validation metrics."""
        header = f"{'Output':<45} {'R²':>8} {'NRMSE':>8} {'MSE':>12}"
        sep    = "-" * 75
        rows   = [header, sep]
        for m in metrics:
            name = m.output_name[-44:]
            rows.append(
                f"{name:<45} {m.r2:>8.4f} {m.nrmse:>8.4f} {m.mse:>12.6f}"
            )
        return "\n".join(rows)