"""
data_manager.py
===============
Responsible for all data I/O, cleaning, scaling and splitting.

Classes
-------
SyntheticDataGenerator
    Generates a stable LTI-driven synthetic dataset for testing.
DataManager
    Loads real CSV/Excel data, preprocesses it, scales it and
    exposes train/test splits as numpy arrays.

Author : Blown Film MPC Project
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from config import ALL_COLUMNS, INPUT_COLS, OUTPUT_COLS, DataConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class IODataset:
    """
    Immutable container for a scaled input/output dataset split.

    Attributes
    ----------
    U_train, Y_train : training arrays  (n_train × n_u/n_y)
    U_test,  Y_test  : test arrays      (n_test  × n_u/n_y)
    input_cols       : list of input column names
    output_cols      : list of output column names
    scaler_u         : fitted RobustScaler for inputs
    scaler_y         : fitted RobustScaler for outputs
    """

    U_train: np.ndarray
    Y_train: np.ndarray
    U_test: np.ndarray
    Y_test: np.ndarray
    input_cols: List[str]
    output_cols: List[str]
    scaler_u: RobustScaler
    scaler_y: RobustScaler

    @property
    def n_inputs(self) -> int:
        return self.U_train.shape[1]

    @property
    def n_outputs(self) -> int:
        return self.Y_train.shape[1]

    @property
    def n_train(self) -> int:
        return self.U_train.shape[0]

    @property
    def n_test(self) -> int:
        return self.U_test.shape[0]

    def __repr__(self) -> str:
        return (
            f"IODataset("
            f"n_train={self.n_train}, n_test={self.n_test}, "
            f"n_inputs={self.n_inputs}, n_outputs={self.n_outputs})"
        )


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

class SyntheticDataGenerator:
    """
    Generates a synthetic input-output dataset driven by a random
    stable discrete-time LTI system.

    Parameters
    ----------
    n_inputs  : number of input channels
    n_outputs : number of output channels
    n_states  : hidden state dimension of the ground-truth system
    cfg       : DataConfig instance
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        n_states: int = 12,
        cfg: DataConfig = DataConfig(),
    ) -> None:
        self._n_u = n_inputs
        self._n_y = n_outputs
        self._n   = n_states
        self._cfg = cfg

    # ------------------------------------------------------------------
    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (U, Y) arrays of shape (n_samples, n_u/n_y).

        The ground-truth system has spectral radius < 1 (stable).
        Inputs are smoothed white noise; outputs are corrupted by
        Gaussian measurement noise.
        """
        rng = np.random.default_rng(self._cfg.random_seed)
        n, m, p = self._n, self._n_u, self._n_y
        T = self._cfg.synthetic_samples

        # Build random stable A_d
        A_raw = rng.standard_normal((n, n))
        rho   = np.max(np.abs(np.linalg.eigvals(A_raw)))
        A_d   = A_raw / (rho + 0.1) * 0.85

        B_d = rng.standard_normal((n, m)) * 0.1
        C_d = rng.standard_normal((p, n)) * 0.5

        # Smooth random inputs
        U = rng.standard_normal((T, m))
        kernel = np.ones(20) / 20
        for j in range(m):
            U[:, j] = np.convolve(U[:, j], kernel, mode="same")

        # Simulate
        Y = np.zeros((T, p))
        x = np.zeros(n)
        noise_std = self._cfg.synthetic_noise_std
        for t in range(T):
            Y[t] = C_d @ x + noise_std * rng.standard_normal(p)
            x    = A_d @ x + B_d @ U[t]

        logger.info(
            "Synthetic dataset generated: "
            "U%s, Y%s", U.shape, Y.shape
        )
        return U, Y


# ---------------------------------------------------------------------------
# Real-data manager
# ---------------------------------------------------------------------------

class DataManager:
    """
    Manages the full data lifecycle for the blown film dataset:
    loading → validation → cleaning → scaling → splitting.

    Parameters
    ----------
    cfg : DataConfig
        Configuration dataclass controlling all preprocessing choices.

    Examples
    --------
    >>> dm = DataManager(DataConfig())
    >>> dataset = dm.load_and_prepare("data.csv")
    """

    def __init__(self, cfg: DataConfig = DataConfig()) -> None:
        self._cfg      = cfg
        self._scaler_u = RobustScaler()
        self._scaler_y = RobustScaler()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load_and_prepare(self, filepath: str) -> IODataset:
        """
        Full pipeline: load → clean → scale → split.

        Parameters
        ----------
        filepath : path to CSV or Excel file

        Returns
        -------
        IODataset
        """
        df = self._load_file(filepath)
        df = self._validate_columns(df)
        df = self._clean(df)
        return self._build_dataset(df)

    def prepare_synthetic(
        self,
        n_inputs: Optional[int] = None,
        n_outputs: Optional[int] = None,
    ) -> IODataset:
        """
        Build an IODataset from synthetic data.

        Parameters
        ----------
        n_inputs  : override number of inputs  (defaults to len(INPUT_COLS))
        n_outputs : override number of outputs (defaults to len(OUTPUT_COLS))
        """
        n_u = n_inputs  or len(INPUT_COLS)
        n_y = n_outputs or len(OUTPUT_COLS)

        gen = SyntheticDataGenerator(
            n_inputs=n_u, n_outputs=n_y, cfg=self._cfg
        )
        U_all, Y_all = gen.generate()
        return self._scale_and_split(
            U_all, Y_all,
            input_cols=[f"u_{i}" for i in range(n_u)],
            output_cols=[f"y_{i}" for i in range(n_y)],
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_file(self, filepath: str) -> pd.DataFrame:
        """Read CSV or Excel into a DataFrame."""
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")

        ext = os.path.splitext(filepath)[-1].lower()
        readers = {
            ".csv":  lambda p: pd.read_csv(p,  parse_dates=[self._cfg.time_column]),
            ".xlsx": lambda p: pd.read_excel(p, parse_dates=[self._cfg.time_column]),
            ".xls":  lambda p: pd.read_excel(p, parse_dates=[self._cfg.time_column]),
        }
        if ext not in readers:
            raise ValueError(f"Unsupported file format: {ext!r}")

        df = readers[ext](filepath)
        logger.info("Loaded %d rows × %d columns from %s",
                    len(df), df.shape[1], filepath)
        return df

    def _validate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only known columns; warn about missing ones."""
        available = [c for c in ALL_COLUMNS if c in df.columns]
        missing   = set(ALL_COLUMNS) - set(df.columns)
        if missing:
            logger.warning(
                "%d expected columns not found in file (will be skipped).",
                len(missing),
            )
        return df[available].copy()

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort, impute, remove outliers."""
        time_col = self._cfg.time_column
        if time_col in df.columns:
            df = df.sort_values(time_col).reset_index(drop=True)

        # Drop fully-NaN columns
        df.dropna(axis=1, how="all", inplace=True)

        # Impute short gaps
        df.ffill(limit=self._cfg.ffill_limit, inplace=True)
        df.bfill(limit=self._cfg.bfill_limit, inplace=True)
        df.dropna(inplace=True)

        # Z-score outlier removal
        num_cols = df.select_dtypes(include=[np.number]).columns
        z = (df[num_cols] - df[num_cols].mean()) / (df[num_cols].std() + 1e-9)
        mask = (np.abs(z) < self._cfg.outlier_zscore).all(axis=1)
        df   = df.loc[mask].reset_index(drop=True)

        logger.info("After cleaning: %d rows remain.", len(df))
        return df

    def _build_dataset(self, df: pd.DataFrame) -> IODataset:
        """Extract I/O arrays from cleaned DataFrame and split."""
        in_cols  = [c for c in INPUT_COLS  if c in df.columns]
        out_cols = [c for c in OUTPUT_COLS if c in df.columns]

        U_all = df[in_cols].values.astype(float)
        Y_all = df[out_cols].values.astype(float)

        return self._scale_and_split(U_all, Y_all, in_cols, out_cols)

    def _scale_and_split(
        self,
        U_all: np.ndarray,
        Y_all: np.ndarray,
        input_cols: List[str],
        output_cols: List[str],
    ) -> IODataset:
        """Scale with RobustScaler and perform chronological split."""
        n_tr = int(len(U_all) * self._cfg.train_fraction)

        U_tr_raw, U_te_raw = U_all[:n_tr], U_all[n_tr:]
        Y_tr_raw, Y_te_raw = Y_all[:n_tr], Y_all[n_tr:]

        U_tr = self._scaler_u.fit_transform(U_tr_raw)
        U_te = self._scaler_u.transform(U_te_raw)
        Y_tr = self._scaler_y.fit_transform(Y_tr_raw)
        Y_te = self._scaler_y.transform(Y_te_raw)

        dataset = IODataset(
            U_train=U_tr, Y_train=Y_tr,
            U_test=U_te,  Y_test=Y_te,
            input_cols=input_cols,
            output_cols=output_cols,
            scaler_u=self._scaler_u,
            scaler_y=self._scaler_y,
        )
        logger.info("Dataset ready: %r", dataset)
        return dataset