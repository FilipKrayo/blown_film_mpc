"""
persistence.py
===============
Save/load helpers for the two cacheable pipeline artefacts:

  * the identified + reduced plant model (``ReducedModel``)
  * the tuned MPC weights (``Q``, ``R`` diagonals)

Both are saved as plain-data snapshots via pickle. Note that the live
``MPCController`` itself is never pickled — it holds a CVXPY
``Problem``/``Parameter`` state that isn't safe (or meaningful) to
serialise. Instead only the tuned ``Q``/``R`` weight arrays are saved;
``main.py`` reconstructs a fresh ``MPCController`` from
(``reduced_model``, ``cfg``) and applies the loaded weights via
``MPCController.set_weights()``.

Classes
-------
MPCWeights
    Immutable (Q, R) weight snapshot.
ModelStore
    Save/load a ``ReducedModel``.
ControllerWeightsStore
    Save/load an ``MPCWeights`` snapshot, with shape validation
    against the current model's (n_y, n_u) so a stale/incompatible
    cache is never silently applied.

Author : Blown Film MPC Project
"""

from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass
from typing import Optional

import numpy as np

from model_reduction import ReducedModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MPCWeights:
    """Snapshot of tuned MPC diagonal weights."""

    Q: np.ndarray
    R: np.ndarray


class ModelStore:
    """Save/load a ``ReducedModel`` to/from a pickle file."""

    @staticmethod
    def save(model: ReducedModel, path: str) -> None:
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(model, f)
        logger.info("Saved reduced model to %s", path)

    @staticmethod
    def load(path: str) -> Optional[ReducedModel]:
        """Return the cached ReducedModel, or None if unavailable/invalid."""
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "rb") as f:
                model = pickle.load(f)
        except Exception as exc:
            logger.warning(
                "Failed to load cached model at %s (%s) — ignoring cache.",
                path, exc,
            )
            return None
        if not isinstance(model, ReducedModel):
            logger.warning(
                "Cached file at %s is not a ReducedModel — ignoring cache.",
                path,
            )
            return None
        logger.info("Loaded cached reduced model from %s: %r", path, model)
        return model


class ControllerWeightsStore:
    """Save/load tuned MPC (Q, R) weights to/from a pickle file."""

    @staticmethod
    def save(Q: np.ndarray, R: np.ndarray, path: str) -> None:
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(MPCWeights(Q=Q, R=R), f)
        logger.info("Saved MPC weights to %s", path)

    @staticmethod
    def load(path: str, n_y: int, n_u: int) -> Optional[MPCWeights]:
        """
        Return the cached MPCWeights, or None if unavailable/invalid.

        Shape is validated against the current model's (n_y, n_u) so a
        cache left over from a differently-sized model is never
        silently applied.
        """
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "rb") as f:
                weights = pickle.load(f)
        except Exception as exc:
            logger.warning(
                "Failed to load cached MPC weights at %s (%s) — ignoring cache.",
                path, exc,
            )
            return None
        if not isinstance(weights, MPCWeights):
            logger.warning(
                "Cached file at %s is not an MPCWeights snapshot — ignoring cache.",
                path,
            )
            return None
        if weights.Q.shape != (n_y, n_y) or weights.R.shape != (n_u, n_u):
            logger.warning(
                "Cached MPC weights at %s have shape Q=%s, R=%s, which "
                "doesn't match the current model (n_y=%d, n_u=%d) — "
                "ignoring cache.",
                path, weights.Q.shape, weights.R.shape, n_y, n_u,
            )
            return None
        logger.info("Loaded cached MPC weights from %s", path)
        return weights
