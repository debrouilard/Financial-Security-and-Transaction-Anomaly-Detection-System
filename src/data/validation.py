from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from src.utils.logging import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = [
    "timestamp", "from_bank", "from_account", "to_bank", "to_account",
    "amount_received", "receiving_currency", "amount_paid",
    "payment_currency", "payment_format", "is_laundering"
]

def validate_schema(df: DataFrame) -> DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df

def basic_quality_checks(df: DataFrame) -> DataFrame:
    """Drop obvious bad rows without collecting to driver."""
    initial = df.count()
    df = df.filter(
        F.col("timestamp").isNotNull()
        & F.col("from_account").isNotNull()
        & F.col("to_account").isNotNull()
        & (F.col("amount_received") >= 0)
        & (F.col("amount_paid") >= 0)
        & F.col("is_laundering").isin(0, 1)
    )
    # Remove exact duplicates on key fields
    df = df.dropDuplicates([
        "timestamp", "from_account", "to_account",
        "amount_received", "amount_paid", "payment_format"
    ])
    final = df.count()
    logger.info(f"Quality filter: {initial} -> {final} rows "
                f"({initial - final} removed)")
    return df