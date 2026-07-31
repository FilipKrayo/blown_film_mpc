"""
accuracy.py
===========
Shared accuracy-gate primitives used by the pipeline to enforce a
minimum per-output R^2 after system identification and again after
model order reduction (see ``AccuracyConfig`` in config.py).

The gate is worst-case, not average: every individual output must
reach the configured threshold. If it doesn't, ``BlownFilmPipeline``
(main.py) escalates the relevant model order and retries, only
raising ``ModelAccuracyError`` once its configured order ceiling is
exhausted.

Author : Blown Film MPC Project
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from sklearn.metrics import r2_score


class ModelAccuracyError(RuntimeError):
    """
    Raised when the model order search exhausts its configured
    ceiling without meeting the required accuracy threshold.
    """


@dataclass
class AccuracyResult:
    """
    Worst-case-oriented accuracy check result for one set of outputs.

    Attributes
    ----------
    stage         : short label identifying which pipeline stage this
                    check was performed at (e.g. "post-identification")
    threshold     : required per-output R² (worst-case)
    per_output_r2 : R² for every output, in ``output_names`` order
    output_names  : output signal names
    n_states      : model order this result was computed at
    """

    stage: str
    threshold: float
    per_output_r2: List[float]
    output_names: List[str]
    n_states: int

    @property
    def worst_index(self) -> int:
        return int(np.argmin(self.per_output_r2))

    @property
    def worst_r2(self) -> float:
        return float(self.per_output_r2[self.worst_index])

    @property
    def worst_output(self) -> str:
        return self.output_names[self.worst_index]

    @property
    def passed(self) -> bool:
        return self.worst_r2 >= self.threshold

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.stage}: worst-case R²={self.worst_r2:.4f} "
            f"(output={self.worst_output!r}, n_states={self.n_states}, "
            f"required>={self.threshold:.4f})"
        )


def evaluate_accuracy(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    output_names: Sequence[str],
    stage: str,
    threshold: float,
    n_states: int,
) -> AccuracyResult:
    """Compute per-output R² and package it into an ``AccuracyResult``."""
    n_out = min(Y_true.shape[1], Y_pred.shape[1], len(output_names))
    r2 = r2_score(
        Y_true[:, :n_out], Y_pred[:, :n_out], multioutput="raw_values"
    )
    return AccuracyResult(
        stage=stage,
        threshold=threshold,
        per_output_r2=[float(v) for v in r2],
        output_names=list(output_names[:n_out]),
        n_states=n_states,
    )
