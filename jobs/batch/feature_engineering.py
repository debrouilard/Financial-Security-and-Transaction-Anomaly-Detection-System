"""
Part 1 – Feature engineering on already-split data.
Velocity & behavioural features are computed independently on each split
to avoid leakage. Global stats (if any) must be fitted on train only.
"""
from pathlib import Path
from pyspark.sql import SparkSession
from src.data.ingestion import create_spark_session
from src.features.temporal import add_temporal_features
from src.features.velocity import add_velocity_features
from src.features.behavioral import add_behavioral_features
from src.utils.config import load_config, get_data_config
from src.utils.logging import get_logger

logger = get_logger("jobs.feature_engineering")

def _add_features(df, windows, lookback):
    df = add_temporal_features(df)
    df = add_velocity_features(df, windows_hours=windows)
    df = add_behavioral_features(df, lookback_days=lookback)
    return df

def run_feature_engineering():
    cfg = load_config()
    data_cfg = get_data_config(cfg)
    windows = data_cfg.get("velocity_windows_hours", [1, 6, 24])
    lookback = data_cfg.get("behavioral_lookback_days", 7)

    spark = create_spark_session()
    processed = Path(data_cfg["processed_path"])
    feature_root = Path(data_cfg["feature_path"])
    feature_root.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        logger.info("Engineering features for %s", split)
        df = spark.read.parquet(str(processed / split))
        featured = _add_features(df, windows, lookback)
        featured.write.mode("overwrite").parquet(str(feature_root / split))
        logger.info("%s features written", split)

    spark.stop()
    logger.info("Feature engineering completed")

if __name__ == "__main__":
    run_feature_engineering()