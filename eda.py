import pandas as pd
import numpy as np


def load_dataset(filepath):
    df = pd.read_csv(filepath)
    return df


def basic_info(df):
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "memory_mb": df.memory_usage(deep=True).sum() / 1024**2
    }
