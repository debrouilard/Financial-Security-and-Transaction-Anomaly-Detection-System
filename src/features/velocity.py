from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from typing import List

def add_velocity_features(
    df: DataFrame,
    windows_hours: List[int] = [1, 6, 24],
) -> DataFrame:
    """
    Account-level velocity (count & sum) over rolling time windows.
    Computed with Spark Window – stays distributed.
    Must be applied AFTER chronological split or with care
    to avoid leakage (see jobs/batch/feature_engineering.py).
    """
    # Sort for deterministic window
    df = df.orderBy("from_account", "timestamp")

    for h in windows_hours:
        # Count of outgoing txs in last h hours
        w = (
            Window.partitionBy("from_account")
            .orderBy(F.col("timestamp").cast("long"))
            .rangeBetween(-h * 3600, -1)   # exclusive of current row
        )
        df = df.withColumn(f"out_cnt_{h}h", F.count("*").over(w))
        df = df.withColumn(f"out_sum_{h}h", F.sum("amount").over(w))
        df = df.withColumn(f"out_avg_{h}h", F.avg("amount").over(w))

        # Incoming velocity
        w_in = (
            Window.partitionBy("to_account")
            .orderBy(F.col("timestamp").cast("long"))
            .rangeBetween(-h * 3600, -1)
        )
        df = df.withColumn(f"in_cnt_{h}h", F.count("*").over(w_in))
        df = df.withColumn(f"in_sum_{h}h", F.sum("amount").over(w_in))

    return df