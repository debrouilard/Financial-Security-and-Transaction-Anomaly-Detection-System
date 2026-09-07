from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

def add_behavioral_features(df: DataFrame, lookback_days: int = 7) -> DataFrame:
    """
    Simple behavioural signals that can be computed online later.
    Uses only past data relative to current transaction.
    """
    # Unique counterparties in lookback
    w = (
        Window.partitionBy("from_account")
        .orderBy(F.col("timestamp").cast("long"))
        .rangeBetween(-lookback_days * 86400, -1)
    )
    df = df.withColumn("unique_counterparties_7d",
                       F.approx_count_distinct("to_account").over(w))
    df = df.withColumn("unique_payment_formats_7d",
                       F.approx_count_distinct("payment_format").over(w))

    # Same-bank ratio
    df = df.withColumn("same_bank_ratio_7d",
                       F.avg("is_same_bank").over(w))

    return df