"""
Part 2 entry point – train Isolation Forest + LOF on the train split only,
then score train / val / test and write scored Parquet files.

Run:  python -m jobs.batch.train_if_lof
"""
from pathlib import Path

from src.data.ingestion import create_spark_session
from src.models.training import train_isolation_forest, train_lof
from src.models.loading import score_with_sklearn_model
from src.models.anomaly_models import IsolationForestModel, LOFModel
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger("jobs.train_if_lof")


def run():
    cfg = load_config()
    model_cfg = cfg["anomaly_models"]          # from config/model.yaml merged or loaded
    # For simplicity we load model.yaml explicitly
    import yaml
    with open("config/model.yaml") as f:
        model_yaml = yaml.safe_load(f)
    model_cfg = model_yaml["anomaly_models"]
    artifacts = model_yaml["artifacts"]

    spark = create_spark_session()
    feature_root = Path(cfg["data"]["feature_path"])
    scored_root = Path("data/scored_if_lof")
    scored_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load train features and train both models
    # ------------------------------------------------------------------
    train_df = spark.read.parquet(str(feature_root / "train"))
    logger.info("Train rows available: %s", train_df.count())

    if_dir = Path(artifacts["isolation_forest"])
    lof_dir = Path(artifacts["lof"])

    logger.info("=== Training Isolation Forest ===")
    if_model = train_isolation_forest(
        train_df,
        model_cfg["isolation_forest"],
        if_dir,
    )

    logger.info("=== Training LOF ===")
    lof_model = train_lof(
        train_df,
        model_cfg["lof"],
        lof_dir,
    )

    # ------------------------------------------------------------------
    # 2. Score every split (train / val / test)
    # ------------------------------------------------------------------
    for split in ["train", "val", "test"]:
        logger.info("Scoring split: %s", split)
        df = spark.read.parquet(str(feature_root / split))

        df = score_with_sklearn_model(df, if_model, score_col="iforest_score")
        df = score_with_sklearn_model(df, lof_model, score_col="lof_score")

        out = scored_root / split
        df.write.mode("overwrite").parquet(str(out))
        logger.info("Wrote scored data → %s", out)

    spark.stop()
    logger.info("Part 2 (IF + LOF) finished successfully")


if __name__ == "__main__":
    run()