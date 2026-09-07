"""
Unsupervised anomaly detectors used in the ensemble.
Isolation Forest and LOF are sklearn-based and therefore trained on a sample.
"""
from __future__ import annotations

import joblib
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from pyod.models.ecod import ECOD
from pyspark.sql import DataFrame
from src.utils.logging import get_logger

logger = get_logger(__name__)


class IsolationForestModel:
    """Wrapper around sklearn IsolationForest with consistent scoring interface."""

    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.feature_cols: List[str] = params["feature_cols"]
        self.model: Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler] = None

    def fit(self, X: np.ndarray) -> "IsolationForestModel":
        logger.info("Fitting IsolationForest on %s samples × %s features", *X.shape)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = IsolationForest(
            n_estimators=self.params.get("n_estimators", 200),
            max_samples=self.params.get("max_samples", 256),
            contamination=self.params.get("contamination", "auto"),
            max_features=self.params.get("max_features", 1.0),
            bootstrap=self.params.get("bootstrap", False),
            n_jobs=self.params.get("n_jobs", -1),
            random_state=self.params.get("random_state", 42),
            verbose=0,
        )
        self.model.fit(X_scaled)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Higher = more anomalous (we invert sklearn’s sign)."""
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not fitted")
        X_scaled = self.scaler.transform(X)
        # sklearn returns higher = more normal → invert
        return -self.model.decision_function(X_scaled)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        return self.decision_function(X)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, directory / "iforest.joblib")
        joblib.dump(self.scaler, directory / "scaler.joblib")
        joblib.dump(self.feature_cols, directory / "feature_cols.joblib")
        joblib.dump(self.params, directory / "params.joblib")
        logger.info("IsolationForest saved to %s", directory)

    @classmethod
    def load(cls, directory: str | Path) -> "IsolationForestModel":
        directory = Path(directory)
        params = joblib.load(directory / "params.joblib")
        obj = cls(params)
        obj.model = joblib.load(directory / "iforest.joblib")
        obj.scaler = joblib.load(directory / "scaler.joblib")
        obj.feature_cols = joblib.load(directory / "feature_cols.joblib")
        return obj


class LOFModel:
    """
    Local Outlier Factor with novelty=True so it can score unseen points.
    Extremely memory/CPU heavy → always train on a modest sample.
    """

    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.feature_cols: List[str] = params["feature_cols"]
        self.model: Optional[LocalOutlierFactor] = None
        self.scaler: Optional[StandardScaler] = None

    def fit(self, X: np.ndarray) -> "LOFModel":
        logger.info("Fitting LOF on %s samples × %s features", *X.shape)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = LocalOutlierFactor(
            n_neighbors=self.params.get("n_neighbors", 35),
            algorithm=self.params.get("algorithm", "auto"),
            leaf_size=self.params.get("leaf_size", 30),
            metric=self.params.get("metric", "minkowski"),
            p=self.params.get("p", 2),
            contamination=self.params.get("contamination", "auto"),
            novelty=True,          # critical for scoring new data
            n_jobs=self.params.get("n_jobs", -1),
        )
        self.model.fit(X_scaled)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Higher = more anomalous (invert sklearn sign)."""
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not fitted")
        X_scaled = self.scaler.transform(X)
        return -self.model.decision_function(X_scaled)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        return self.decision_function(X)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, directory / "lof.joblib")
        joblib.dump(self.scaler, directory / "scaler.joblib")
        joblib.dump(self.feature_cols, directory / "feature_cols.joblib")
        joblib.dump(self.params, directory / "params.joblib")
        logger.info("LOF saved to %s", directory)

    @classmethod
    def load(cls, directory: str | Path) -> "LOFModel":
        directory = Path(directory)
        params = joblib.load(directory / "params.joblib")
        obj = cls(params)
        obj.model = joblib.load(directory / "lof.joblib")
        obj.scaler = joblib.load(directory / "scaler.joblib")
        obj.feature_cols = joblib.load(directory / "feature_cols.joblib")
        return obj 
# Autoencoder (PyTorch)
# ------------------------------------------------------------------


class _AENetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, dropout: float = 0.1):
        super().__init__()
        dims = [input_dim] + hidden_dims
        encoder_layers = []
        for i in range(len(dims) - 1):
            encoder_layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder (mirror)
        decoder_dims = list(reversed(dims))
        decoder_layers = []
        for i in range(len(decoder_dims) - 1):
            decoder_layers += [
                nn.Linear(decoder_dims[i], decoder_dims[i + 1]),
                nn.ReLU() if i < len(decoder_dims) - 2 else nn.Identity(),
            ]
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class AutoencoderModel:
    """Reconstruction-error based anomaly detector."""

    def __init__(self, params: dict):
        self.params = params
        self.feature_cols = params["feature_cols"]
        self.device = torch.device(params.get("device", "cpu"))
        self.model: Optional[_AENetwork] = None
        self.scaler: Optional[StandardScaler] = None
        self.threshold_: Optional[float] = None   # optional, set later if needed

    def fit(self, X: np.ndarray) -> "AutoencoderModel":
        logger.info("Fitting Autoencoder on %s samples × %s features", *X.shape)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X).astype(np.float32)

        input_dim = X_scaled.shape[1]
        self.model = _AENetwork(
            input_dim=input_dim,
            hidden_dims=self.params.get("hidden_dims", [64, 32, 16]),
            dropout=self.params.get("dropout", 0.1),
        ).to(self.device)

        dataset = TensorDataset(torch.from_numpy(X_scaled))
        loader = DataLoader(
            dataset,
            batch_size=self.params.get("batch_size", 1024),
            shuffle=True,
            drop_last=False,
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.params.get("learning_rate", 1e-3),
            weight_decay=self.params.get("weight_decay", 1e-5),
        )
        criterion = nn.MSELoss()

        best_loss = float("inf")
        patience = self.params.get("early_stopping_patience", 5)
        patience_counter = 0

        self.model.train()
        for epoch in range(self.params.get("epochs", 30)):
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                recon = self.model(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.size(0)
            epoch_loss /= len(dataset)

            if epoch_loss < best_loss - 1e-5:
                best_loss = epoch_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        self.model.load_state_dict(best_state)
        self.model.eval()
        logger.info("Autoencoder training finished. Best loss: %.6f", best_loss)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Mean squared reconstruction error – higher = more anomalous."""
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not fitted")
        X_scaled = self.scaler.transform(X).astype(np.float32)
        self.model.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(X_scaled).to(self.device)
            recon = self.model(tensor)
            mse = torch.mean((recon - tensor) ** 2, dim=1)
        return mse.cpu().numpy()

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        return self.decision_function(X)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), directory / "ae_weights.pt")
        joblib.dump(self.scaler, directory / "scaler.joblib")
        joblib.dump(self.feature_cols, directory / "feature_cols.joblib")
        joblib.dump(self.params, directory / "params.joblib")
        # also save architecture meta
        joblib.dump(
            {"input_dim": len(self.feature_cols),
             "hidden_dims": self.params.get("hidden_dims", [64, 32, 16]),
             "dropout": self.params.get("dropout", 0.1)},
            directory / "arch.joblib",
        )
        logger.info("Autoencoder saved to %s", directory)

    @classmethod
    def load(cls, directory: str | Path) -> "AutoencoderModel":
        directory = Path(directory)
        params = joblib.load(directory / "params.joblib")
        obj = cls(params)
        arch = joblib.load(directory / "arch.joblib")
        obj.model = _AENetwork(
            input_dim=arch["input_dim"],
            hidden_dims=arch["hidden_dims"],
            dropout=arch["dropout"],
        )
        obj.model.load_state_dict(torch.load(directory / "ae_weights.pt", map_location="cpu"))
        obj.model.eval()
        obj.scaler = joblib.load(directory / "scaler.joblib")
        obj.feature_cols = joblib.load(directory / "feature_cols.joblib")
        return obj


# ------------------------------------------------------------------
# ECOD (pyod)
# ------------------------------------------------------------------


class ECODModel:
    """Empirical Cumulative Distribution-based Outlier Detection."""

    def __init__(self, params: dict):
        self.params = params
        self.feature_cols = params["feature_cols"]
        self.model: Optional[ECOD] = None
        self.scaler: Optional[StandardScaler] = None

    def fit(self, X: np.ndarray) -> "ECODModel":
        logger.info("Fitting ECOD on %s samples × %s features", *X.shape)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = ECOD(
            contamination=self.params.get("contamination", 0.001),
            n_jobs=self.params.get("n_jobs", -1),
        )
        self.model.fit(X_scaled)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Higher = more anomalous."""
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not fitted")
        X_scaled = self.scaler.transform(X)
        return self.model.decision_function(X_scaled)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        return self.decision_function(X)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, directory / "ecod.joblib")
        joblib.dump(self.scaler, directory / "scaler.joblib")
        joblib.dump(self.feature_cols, directory / "feature_cols.joblib")
        joblib.dump(self.params, directory / "params.joblib")
        logger.info("ECOD saved to %s", directory)

    @classmethod
    def load(cls, directory: str | Path) -> "ECODModel":
        directory = Path(directory)
        params = joblib.load(directory / "params.joblib")
        obj = cls(params)
        obj.model = joblib.load(directory / "ecod.joblib")
        obj.scaler = joblib.load(directory / "scaler.joblib")
        obj.feature_cols = joblib.load(directory / "feature_cols.joblib")
        return obj


# ------------------------------------------------------------------
# Robust Statistical Scoring
# ------------------------------------------------------------------
class RobustStatisticalModel:
    """
    Fully statistical anomaly score based on MAD / IQR / robust z-scores.
    Can be fitted on a sample or (preferably) on the full train split
    using Spark approximate quantiles for true scalability.
    """

    def __init__(self, params: dict):
        self.params = params
        self.feature_cols = params["feature_cols"]
        self.stats_: Dict[str, Dict[str, float]] = {}   # per-feature median, mad, q1, q3
        self.aggregation = params.get("aggregation", "mean")

    def fit(self, X: np.ndarray, feature_names: List[str] = None) -> "RobustStatisticalModel":
        """Fit on a numpy sample (fallback). Prefer fit_spark for large data."""
        feature_names = feature_names or self.feature_cols
        logger.info("Fitting RobustStatistical on %s samples", X.shape[0])
        for i, col in enumerate(feature_names):
            col_data = X[:, i]
            median = np.median(col_data)
            mad = np.median(np.abs(col_data - median)) + 1e-8
            q1, q3 = np.percentile(col_data, [25, 75])
            self.stats_[col] = {
                "median": float(median),
                "mad": float(mad),
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(q3 - q1 + 1e-8),
            }
        return self

    def fit_spark(self, spark_df: DataFrame) -> "RobustStatisticalModel":
        """
        Preferred: compute robust statistics with Spark approx quantiles.
        No data movement to driver except the tiny stats dictionary.
        """
        logger.info("Fitting RobustStatistical with Spark approxQuantile")
        for col in self.feature_cols:
            # approxQuantile is cheap and distributed
            quantiles = spark_df.approxQuantile(col, [0.25, 0.5, 0.75], 0.001)
            q1, median, q3 = quantiles
            # MAD approximated via another pass (or skip if too expensive)
            # For speed we use IQR-based robust scale as primary
            mad = (q3 - q1) / 1.349          # consistent with normal
            self.stats_[col] = {
                "median": float(median),
                "mad": float(mad + 1e-8),
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(q3 - q1 + 1e-8),
            }
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Per-feature robust z-score (or IQR distance), then aggregate.
        Higher = more anomalous.
        """
        scores = []
        for i, col in enumerate(self.feature_cols):
            s = self.stats_[col]
            col_data = X[:, i]
            if self.params.get("use_robust_z", True):
                z = np.abs(col_data - s["median"]) / s["mad"]
            else:
                z = np.abs(col_data - s["median"]) / s["iqr"]
            scores.append(z)
        scores = np.vstack(scores).T          # (n_samples, n_features)

        if self.aggregation == "max":
            return np.max(scores, axis=1)
        elif self.aggregation == "sum":
            return np.sum(scores, axis=1)
        else:  # mean
            return np.mean(scores, axis=1)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        return self.decision_function(X)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.stats_, directory / "stats.joblib")
        joblib.dump(self.feature_cols, directory / "feature_cols.joblib")
        joblib.dump(self.params, directory / "params.joblib")
        logger.info("RobustStatistical saved to %s", directory)

    @classmethod
    def load(cls, directory: str | Path) -> "RobustStatisticalModel":
        directory = Path(directory)
        params = joblib.load(directory / "params.joblib")
        obj = cls(params)
        obj.stats_ = joblib.load(directory / "stats.joblib")
        obj.feature_cols = joblib.load(directory / "feature_cols.joblib")
        return obj