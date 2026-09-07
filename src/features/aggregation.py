from pyspark.sql import DataFrame
from pyspark.sql import functions as F

def add_global_aggregates(df: DataFrame) -> DataFrame:
    """
    Light global / currency / format statistics that can be pre-computed
    on the training set only and joined later (prevents leakage).
    """
    # Placeholder – real global stats should be fitted on train only
    # and applied via broadcast join in the job layer.
    return df