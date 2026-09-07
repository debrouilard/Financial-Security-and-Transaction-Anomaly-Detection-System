"""
Part 3 entry point – train Autoencoder + ECOD + Robust Statistical Scoring,
then score train / val / test.

Run:  python -m jobs.batch.train_ae_ecod_robust
"""
from pathlib import Path
import yaml

from src.data.ingestion import create_spark_session
from src.models.training import (
    train_autoencoder,
    train_ecod,
    train_robust_statistical,
)
from src.models.loading import score_with_sklearn_model
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger("jobs.train_ae_ecod_robust")


def run():
    cfg = load_config()
    with open("config/model.yaml") as f:
        model_yaml = yaml.safe_load(f)
    model_cfg = model_yaml["anomaly_models"]
    artifacts = model_yaml["artifacts"]

    spark = create_spark_session()
    feature_root = Path(cfg["data"]["feature_path"])
    scored_root = Path("data/scored_ae_ecod_robust")
    scored_root.mkdir(parents=True, exist_ok=True)

    train_df = spark.read.parquet(str(feature_root / "train"))
    logger.info("Train rows available: %s", train_df.count())

    # ------------------------------------------------------------------
    # Train the three models
    # ------------------------------------------------------------------
    logger.info("=== Training Autoencoder ===")
    ae_model = train_autoencoder(
        train_df,
        model_cfg["autoencoder"],
        artifacts["autoencoder"],
    )

    logger.info("=== Training ECOD ===")
    ecod_model = train_ecod(
        train_df,
        model_cfg["ecod"],
        artifacts["ecod"],
    )

    logger.info("=== Training Robust Statistical ===")
    robust_model = train_robust_statistical(
        train_df,
        model_cfg["robust_statistical"],
        artifacts["robust_statistical"],
        prefer_spark=True,
    )

    # ------------------------------------------------------------------
    # Score every split
    # ------------------------------------------------------------------
    for split in ["train", "val", "test"]:
        logger.info("Scoring split: %s", split)
        df = spark.read.parquet(str(feature_root / split))

        df = score_with_sklearn_model(df, ae_model,     score_col="ae_score")
        df = score_with_sklearn_model(df, ecod_model,   score_col="ecod_score")
        df = score_with_sklearn_model(df, robust_model, score_col="robust_score")

        out = scored_root / split
        df.write.mode("overwrite").parquet(str(out))
        logger.info("Wrote scored data → %s", out)

    spark.stop()
    logger.info("Part 3 (AE + ECOD + Robust) finished successfully")


if __name__ == "__main__":
    run()