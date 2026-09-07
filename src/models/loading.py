"""
Load saved models and apply them to Spark DataFrames in a scalable way.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StructType, StructField, StringType

from src.models.anomaly_models import IsolationForestModel, LOFModel
from src.utils.logging import get_logger

logger = get_logger(__name__)


def score_with_sklearn_model(
    spark_df: DataFrame,
    model,                          # IsolationForestModel or LOFModel
    score_col: str,
    batch_size: int = 50_000,
) -> DataFrame:
    """
    Score a Spark DataFrame with a sklearn model by mapping over partitions.
    Keeps memory bounded; never collects the full dataset.
    """
    feature_cols = model.feature_cols
    bc_model = spark_df.sql_ctx.sparkSession.sparkContext.broadcast(model)

    def _score_partition(iterator):
        model_local = bc_model.value
        rows = list(iterator)
        if not rows:
            return iter([])
        pdf = pd.DataFrame(rows, columns=rows[0].__fields__)
        X = pdf[feature_cols].astype(np.float32).values
        scores = model_local.decision_function(X)
        pdf[score_col] = scores
        # yield original columns + new score
        for record in pdf.to_dict(orient="records"):
            yield tuple(record[c] for c in pdf.columns)

    # Preserve schema + add score column
    new_schema = spark_df.schema.add(StructField(score_col, DoubleType(), True))
    scored = spark_df.rdd.mapPartitions(_score_partition).toDF(schema=new_schema)
    return scored