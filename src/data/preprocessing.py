from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from src.utils.logging import get_logger

logger = get_logger(__name__)

def clean_and_enrich(df: DataFrame) -> DataFrame:
    """
    Lightweight cleaning + derived columns that do not cause leakage.
    All operations stay distributed.
    """
    df = df.withColumn("amount", F.coalesce(F.col("amount_received"), F.col("amount_paid")))
    df = df.withColumn("is_cross_currency",
                       (F.col("receiving_currency") != F.col("payment_currency")).cast("int"))
    df = df.withColumn("is_same_bank",
                       (F.col("from_bank") == F.col("to_bank")).cast("int"))
    df = df.withColumn("hour_of_day", F.hour("timestamp"))
    df = df.withColumn("day_of_week", F.dayofweek("timestamp"))  # 1=Sun … 7=Sat
    df = df.withColumn("date", F.to_date("timestamp"))

    # Log-transform amount (safe, no leakage)
    df = df.withColumn("log_amount", F.log1p(F.col("amount")))

    logger.info("Basic enrichment completed")
    return df

def chronological_split(
    df: DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[DataFrame, DataFrame, DataFrame]:
    """
    Strict chronological split by timestamp.
    Prevents future information from leaking into train/val.
    Test set is left completely untouched until final evaluation.
    """
    # Global min/max timestamps
    stats = df.agg(
        F.min("timestamp").alias("min_ts"),
        F.max("timestamp").alias("max_ts")
    ).collect()[0]
    min_ts, max_ts = stats["min_ts"], stats["max_ts"]
    total_seconds = (max_ts - min_ts).total_seconds()

    train_end = min_ts + F.expr(f"INTERVAL {int(total_seconds * train_ratio)} SECONDS")
    val_end   = min_ts + F.expr(f"INTERVAL {int(total_seconds * (train_ratio + val_ratio))} SECONDS")

    # Because we cannot use Python datetime arithmetic directly on Column,
    # we use a safer percentile-based approach that still respects order.
    # Alternative robust method:
    ordered = df.orderBy("timestamp")
    total = ordered.count()
    train_cut = int(total * train_ratio)
    val_cut   = int(total * (train_ratio + val_ratio))

    # Add row number (expensive but correct for chronological integrity)
    w = Window.orderBy("timestamp")
    ordered = ordered.withColumn("rn", F.row_number().over(w))

    train_df = ordered.filter(F.col("rn") <= train_cut).drop("rn")
    val_df   = ordered.filter((F.col("rn") > train_cut) & (F.col("rn") <= val_cut)).drop("rn")
    test_df  = ordered.filter(F.col("rn") > val_cut).drop("rn")

    logger.info(f"Chronological split -> train: {train_df.count()}, "
                f"val: {val_df.count()}, test: {test_df.count()}")
    return train_df, val_df, test_df