"""
Part 1 entry point – PySpark loading, cleaning, chronological split.
Run: python -m jobs.batch.preprocessing
"""
from pathlib import Path
from src.data.ingestion import create_spark_session, load_raw_transactions
from src.data.validation import validate_schema, basic_quality_checks
from src.data.preprocessing import clean_and_enrich, chronological_split
from src.utils.config import load_config, get_data_config
from src.utils.logging import get_logger

logger = get_logger("jobs.preprocessing")

def run_preprocessing():
    cfg = load_config()
    data_cfg = get_data_config(cfg)

    spark = create_spark_session(
        app_name=cfg["spark"]["app_name"],
        master=cfg["spark"].get("master", "local[*]")
    )

    # 1. Load
    raw = load_raw_transactions(spark, path=data_cfg["raw_path"])

    # 2. Validate + quality
    raw = validate_schema(raw)
    clean = basic_quality_checks(raw)

    # 3. Enrich
    enriched = clean_and_enrich(clean)

    # 4. Chronological split (labels stay with the rows)
    train, val, test = chronological_split(
        enriched,
        train_ratio=data_cfg["train_ratio"],
        val_ratio=data_cfg["val_ratio"],
    )

    # 5. Persist (Parquet for speed & schema preservation)
    out = Path(data_cfg["processed_path"])
    out.mkdir(parents=True, exist_ok=True)

    train.write.mode("overwrite").parquet(str(out / "train"))
    val.write.mode("overwrite").parquet(str(out / "val"))
    test.write.mode("overwrite").parquet(str(out / "test"))

    logger.info("Preprocessing finished. Splits written to %s", out)
    spark.stop()

if __name__ == "__main__":
    run_preprocessing()