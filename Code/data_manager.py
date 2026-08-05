"""
data_manager.py
===============
Responsible for all data I/O, cleaning, scaling and splitting.

Classes
-------
SyntheticDataGenerator
    Generates a physically-structured synthetic dataset from the
    linearised first-principles blown film model.
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

from config import ALL_COLUMNS, INPUT_COLS, OUTPUT_COLS, DataConfig, PhysicalModelConfig

logger = logging.getLogger(__name__)

# No extruder running/screw-speed status flag is exported anywhere in the
# real SCADA data (checked against extrusion_data_legend.xlsx), so melt
# pressure is the only available proxy for "is this extruder actively
# extruding" -- used by _clean() to drop idle/startup rows.
_EXTRUDER_PRESSURE_COLS = [
    "ST110_VARExtr_1_druck_1_IstP",
    "ST110_VARExtr_2_druck_1_IstP",
    "ST110_VARExtr_3_druck_1_IstP",
]


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
    detected_sampling_time : median Δt (s) from real-data timestamps,
                              None for synthetic data
    """

    U_train: np.ndarray
    Y_train: np.ndarray
    U_test: np.ndarray
    Y_test: np.ndarray
    input_cols: List[str]
    output_cols: List[str]
    scaler_u: RobustScaler
    scaler_y: RobustScaler
    detected_sampling_time: Optional[float] = None

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
    Generates a physically-structured synthetic input-output dataset
    from the linearised, discretised first-principles blown film model
    (``physical_model.FirstPrinciplesModel``), replacing the previous
    random dense-LTI ground truth. Inputs are smoothed perturbation
    signals around the model's nominal operating point (README §2);
    outputs are simulated through the linear model with light
    eigenvalue-clipping stabilisation applied purely for bounded data
    generation. The real line is never operated open-loop (IBC/haul-
    off/winder feedback keeps it stable in practice), so this
    substitutes for that closed-loop stabilisation rather than
    representing a modelling error.

    Parameters
    ----------
    cfg      : DataConfig instance
    phys_cfg : PhysicalModelConfig instance (subsystem multiplicities)
    """

    def __init__(
        self,
        cfg: DataConfig = DataConfig(),
        phys_cfg: Optional[PhysicalModelConfig] = None,
    ) -> None:
        self._cfg = cfg
        self._phys_cfg = phys_cfg or PhysicalModelConfig()

    # ------------------------------------------------------------------
    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (U, Y) arrays of shape (n_samples, n_u/n_y), where n_u/
        n_y are the ``FirstPrinciplesModel``'s own input/output counts
        (see config.PhysicalModelConfig), not len(INPUT_COLS/OUTPUT_COLS).
        """
        from physical_model import FirstPrinciplesModel, stabilise_discrete_matrix  # local import avoids a cycle

        pc = self._phys_cfg
        model = FirstPrinciplesModel(
            n_extruders=pc.n_extruders, n_zones=pc.n_zones,
            n_ibc=pc.n_ibc, n_winders=pc.n_winders,
        )
        tau_threshold = (
            pc.fast_time_constant_threshold_factor * self._cfg.sampling_time
            if pc.enable_singular_perturbation else None
        )
        ss, x0, u0, _ = model.to_state_space_model(
            Ts=self._cfg.sampling_time, tau_threshold=tau_threshold,
            min_spectral_gap=pc.singular_perturbation_min_gap,
        )
        A_d = stabilise_discrete_matrix(ss.A)

        rng = np.random.default_rng(self._cfg.random_seed)
        T = self._cfg.synthetic_samples
        m, p = ss.n_inputs, ss.n_outputs

        # Smoothed perturbation inputs, scaled relative to each input's
        # own nominal magnitude (small excitation around the operating point).
        dU = rng.standard_normal((T, m))
        kernel = np.ones(20) / 20
        for j in range(m):
            dU[:, j] = np.convolve(dU[:, j], kernel, mode="same")
        U = dU * (0.02 * np.maximum(np.abs(u0), 1.0))

        Y = np.zeros((T, p))
        x = np.zeros(ss.n_states)   # perturbation state (x - x0)
        noise_std = self._cfg.synthetic_noise_std
        for t in range(T):
            Y[t] = ss.C @ x + ss.D @ U[t] + noise_std * rng.standard_normal(p)
            x    = A_d @ x + ss.B @ U[t]

        logger.info(
            "Physics-based synthetic dataset generated: U%s, Y%s", U.shape, Y.shape
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
        detected_ts = self._detect_sampling_time(df)
        return self._build_dataset(df, detected_sampling_time=detected_ts)

    def prepare_synthetic(self) -> IODataset:
        """
        Build an IODataset from physics-based synthetic data. Input/
        output counts come from the FirstPrinciplesModel's own
        multiplicities (config.PhysicalModelConfig), not len(INPUT_COLS)/
        len(OUTPUT_COLS).
        """
        gen = SyntheticDataGenerator(cfg=self._cfg)
        U_all, Y_all = gen.generate()
        n_u, n_y = U_all.shape[1], Y_all.shape[1]
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

        # Idle/startup rows (any extruder not actively extruding) contaminate
        # the fit -- see _EXTRUDER_PRESSURE_COLS for why pressure is used as
        # the running-status proxy.
        pressure_cols = [c for c in _EXTRUDER_PRESSURE_COLS if c in df.columns]
        if pressure_cols:
            running = (df[pressure_cols] >= self._cfg.min_running_pressure_bar).all(axis=1)
            df = df.loc[running].reset_index(drop=True)

        logger.info("After cleaning: %d rows remain.", len(df))
        return df

    def _detect_sampling_time(self, df: pd.DataFrame) -> Optional[float]:
        """Median Δt (s) between consecutive timestamps, or None if unavailable."""
        time_col = self._cfg.time_column
        if time_col not in df.columns or len(df) < 2:
            return None
        deltas = df[time_col].diff().dropna().dt.total_seconds()
        deltas = deltas[deltas > 0]
        if deltas.empty:
            return None
        return float(deltas.median())

    def _build_dataset(
        self, df: pd.DataFrame, detected_sampling_time: Optional[float] = None,
    ) -> IODataset:
        """Extract I/O arrays from cleaned DataFrame and split."""
        in_cols  = [c for c in INPUT_COLS  if c in df.columns]
        out_cols = [c for c in OUTPUT_COLS if c in df.columns]

        U_all = df[in_cols].values.astype(float)
        Y_all = df[out_cols].values.astype(float)

        return self._scale_and_split(
            U_all, Y_all, in_cols, out_cols,
            detected_sampling_time=detected_sampling_time,
        )

    def _scale_and_split(
        self,
        U_all: np.ndarray,
        Y_all: np.ndarray,
        input_cols: List[str],
        output_cols: List[str],
        detected_sampling_time: Optional[float] = None,
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
            detected_sampling_time=detected_sampling_time,
        )
        logger.info("Dataset ready: %r", dataset)
        return dataset