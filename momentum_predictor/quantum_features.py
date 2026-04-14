from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class QuantumFeatureConfig:
    feature_cols: list[str]
    n_qubits: int = 6
    n_layers: int = 2
    clip_value: float = 3.0
    random_state: int = 42


class QuantumFeatureModule:
    """
    Lightweight quantum-inspired feature mapper.

    This is not a hardware quantum model. It creates a compact nonlinear
    embedding using angle encoding + entanglement-style mixing, which is a
    practical way to test whether "quantum-style" structure helps your signal.

    Output columns:
        - quantum_score
        - quantum_energy
        - quantum_dispersion
        - quantum_alignment
    """

    def __init__(self, config: QuantumFeatureConfig):
        self.config = config
        self.feature_means_: np.ndarray | None = None
        self.feature_stds_: np.ndarray | None = None
        self.proj_: np.ndarray | None = None
        self.mix_: np.ndarray | None = None
        self.is_fitted_: bool = False

    def fit(self, df: pd.DataFrame) -> "QuantumFeatureModule":
        x = self._prepare_input(df, fit=True)

        rng = np.random.default_rng(self.config.random_state)

        # random projection from feature space -> qubit angles
        self.proj_ = rng.normal(
            loc=0.0,
            scale=1.0 / np.sqrt(max(x.shape[1], 1)),
            size=(x.shape[1], self.config.n_qubits),
        )

        # mixing matrix for "entanglement-like" interactions
        mix = rng.normal(
            loc=0.0,
            scale=1.0 / np.sqrt(self.config.n_qubits),
            size=(self.config.n_qubits, self.config.n_qubits),
        )
        self.mix_ = 0.5 * (mix + mix.T)

        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("QuantumFeatureModule must be fitted before transform().")

        x = self._prepare_input(df, fit=False)
        angles = self._encode_angles(x)
        evolved = self._evolve(angles)
        return self._build_output(df.index, evolved)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def _prepare_input(self, df: pd.DataFrame, fit: bool) -> np.ndarray:
        missing = [c for c in self.config.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns for quantum module: {missing}")

        x = df[self.config.feature_cols].astype(float).copy()

        # fill per-column with median
        x = x.fillna(x.median())

        x_np = x.to_numpy(dtype=np.float64)

        if fit:
            self.feature_means_ = np.nanmean(x_np, axis=0)
            self.feature_stds_ = np.nanstd(x_np, axis=0) + 1e-6

        if self.feature_means_ is None or self.feature_stds_ is None:
            raise RuntimeError("Feature normalization stats are missing.")

        x_np = (x_np - self.feature_means_) / self.feature_stds_
        x_np = np.clip(x_np, -self.config.clip_value, self.config.clip_value)

        return x_np

    def _encode_angles(self, x: np.ndarray) -> np.ndarray:
        if self.proj_ is None:
            raise RuntimeError("Projection matrix is missing.")

        base = x @ self.proj_

        # map into angle space
        angles = np.tanh(base) * np.pi

        return angles

    def _evolve(self, angles: np.ndarray) -> np.ndarray:
        if self.mix_ is None:
            raise RuntimeError("Mixing matrix is missing.")

        state = angles.copy()

        for _ in range(self.config.n_layers):
            sin_part = np.sin(state)
            cos_part = np.cos(state)

            interaction = sin_part @ self.mix_
            state = 0.6 * state + 0.25 * interaction + 0.15 * cos_part

        return state

    def _build_output(self, index: pd.Index, state: np.ndarray) -> pd.DataFrame:
        # bounded nonlinear summaries
        amp = np.sin(state)
        phase = np.cos(state)

        quantum_score = amp.mean(axis=1)
        quantum_energy = np.mean(amp**2 + phase**2, axis=1)
        quantum_dispersion = np.std(amp, axis=1)

        signed_alignment = np.mean(amp * phase, axis=1)

        out = pd.DataFrame(
            {
                "quantum_score": quantum_score.astype(np.float32),
                "quantum_energy": quantum_energy.astype(np.float32),
                "quantum_dispersion": quantum_dispersion.astype(np.float32),
                "quantum_alignment": signed_alignment.astype(np.float32),
            },
            index=index,
        )

        return out


def add_quantum_features(
    df_train: pd.DataFrame,
    df_other: pd.DataFrame,
    feature_cols: Iterable[str],
    *,
    n_qubits: int = 6,
    n_layers: int = 2,
    clip_value: float = 3.0,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, QuantumFeatureModule]:
    """
    Fit on train only, transform train + other.
    """
    feature_cols = list(feature_cols)

    module = QuantumFeatureModule(
        QuantumFeatureConfig(
            feature_cols=feature_cols,
            n_qubits=n_qubits,
            n_layers=n_layers,
            clip_value=clip_value,
            random_state=random_state,
        )
    )

    q_train = module.fit_transform(df_train)
    q_other = module.transform(df_other)

    train_out = pd.concat([df_train.reset_index(drop=True), q_train.reset_index(drop=True)], axis=1)
    other_out = pd.concat([df_other.reset_index(drop=True), q_other.reset_index(drop=True)], axis=1)

    return train_out, other_out, module