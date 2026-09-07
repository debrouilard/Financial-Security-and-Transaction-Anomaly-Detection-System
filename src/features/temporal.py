from pyspark.sql import DataFrame
from pyspark.sql import functions as F

def add_temporal_features(df: DataFrame) -> DataFrame:
    """Pure temporal features – no account history, safe for all splits."""
    return (
        df.withColumn("hour_sin", F.sin(2 * 3.1415926535 * F.col("hour_of_day") / 24))
          .withColumn("hour_cos", F.cos(2 * 3.1415926535 * F.col("hour_of_day") / 24))
          .withColumn("dow_sin",  F.sin(2 * 3.1415926535 * F.col("day_of_week") / 7))
          .withColumn("dow_cos",  F.cos(2 * 3.1415926535 * F.col("day_of_week") / 7))
          .withColumn("is_weekend", (F.col("day_of_week").isin(1, 7)).cast("int"))
          .withColumn("is_night",   (F.col("hour_of_day").between(0, 5)).cast("int"))
    )