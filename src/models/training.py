"""
Training utilities for unsupervised anomaly models.
All fitting happens on a sampled subset of the train split only.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, ArrayType

from src.models.anomaly_models import IsolationForestModel, LOFModel
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _sample_for_training(
    spark_df: DataFrame,
    feature_cols: List[str],
    sample_size: int,
    label_col: str = "is_laundering",
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stratified-ish sample: keep ALL positive (laundering) rows that fit,
    then fill the rest with random negatives. Returns (X, y) as numpy.
    Never materialises the full dataset.
    """
    # Ensure columns exist
    missing = [c for c in feature_cols if c not in spark_df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    pos = spark_df.filter(F.col(label_col) == 1)
    neg = spark_df.filter(F.col(label_col) == 0)

    pos_cnt = pos.count()
    target_pos = min(pos_cnt, sample_size // 5)          # at most 20 % positives
    target_neg = sample_size - target_pos

    pos_sample = pos.orderBy(F.rand(seed)).limit(target_pos)
    neg_sample = neg.orderBy(F.rand(seed)).limit(target_neg)

    sample_df = pos_sample.unionByName(neg_sample)

    # Collect only the sample (fits in driver memory by design)
    pdf = sample_df.select(feature_cols + [label_col]).toPandas()
    X = pdf[feature_cols].astype(np.float32).values
    y = pdf[label_col].values
    logger.info("Training sample shape: %s (positives: %s)", X.shape, y.sum())
    return X, y


def train_isolation_forest(
    train_df: DataFrame,
    model_cfg: dict,
    artifact_dir: str | Path,
) -> IsolationForestModel:
    X, _ = _sample_for_training(
        train_df,
        feature_cols=model_cfg["feature_cols"],
        sample_size=model_cfg["train_sample_size"],
    )
    model = IsolationForestModel(model_cfg)
    model.fit(X)
    model.save(artifact_dir)
    return model


def train_lof(
    train_df: DataFrame,
    model_cfg: dict,
    artifact_dir: str | Path,
) -> LOFModel:
    X, _ = _sample_for_training(
        train_df,
        feature_cols=model_cfg["feature_cols"],
        sample_size=model_cfg["train_sample_size"],
    )
    model = LOFModel(model_cfg)
    model.fit(X)
    model.save(artifact_dir)
    return model