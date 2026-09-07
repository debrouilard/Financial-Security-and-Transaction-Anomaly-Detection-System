from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, IntegerType, TimestampType
)
from typing import Optional
from src.utils.config import get_data_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

def create_spark_session(app_name: str = "AML-Anomaly-Detection", master: str = "local[*]") -> SparkSession:
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark

def get_aml_schema() -> StructType:
    """Explicit schema for IBM AML Large to avoid inference cost on huge files."""
    return StructType([
        StructField("Timestamp", StringType(), True),
        StructField("From Bank", IntegerType(), True),
        StructField("Account", StringType(), True),          # originator
        StructField("To Bank", IntegerType(), True),
        StructField("Account.1", StringType(), True),        # beneficiary (common Kaggle naming)
        StructField("Amount Received", DoubleType(), True),
        StructField("Receiving Currency", StringType(), True),
        StructField("Amount Paid", DoubleType(), True),
        StructField("Payment Currency", StringType(), True),
        StructField("Payment Format", StringType(), True),
        StructField("Is Laundering", IntegerType(), True),
    ])

def load_raw_transactions(
    spark: SparkSession,
    path: Optional[str] = None,
    format: str = "csv",
) -> DataFrame:
    """
    Load IBM AML Large dataset with PySpark only.
    Supports CSV (header) or Parquet. Never materialises to Pandas.
    """
    cfg = get_data_config()
    path = path or cfg["raw_path"]
    logger.info(f"Loading raw transactions from {path}")

    if format == "parquet" or path.endswith(".parquet"):
        df = spark.read.parquet(path)
    else:
        df = (
            spark.read
            .option("header", "true")
            .option("inferSchema", "false")
            .schema(get_aml_schema())
            .csv(path)
        )

    # Standardise column names early
    rename_map = {
        "Timestamp": "timestamp",
        "From Bank": "from_bank",
        "Account": "from_account",
        "To Bank": "to_bank",
        "Account.1": "to_account",
        "Amount Received": "amount_received",
        "Receiving Currency": "receiving_currency",
        "Amount Paid": "amount_paid",
        "Payment Currency": "payment_currency",
        "Payment Format": "payment_format",
        "Is Laundering": "is_laundering",
    }
    for old, new in rename_map.items():
        if old in df.columns:
            df = df.withColumnRenamed(old, new)

    # Ensure timestamp is proper TimestampType
    df = df.withColumn(
        "timestamp",
        F.to_timestamp(F.col("timestamp"), "yyyy/MM/dd HH:mm")
    )

    logger.info(f"Loaded {df.count()} rows, columns: {df.columns}")
    return df