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


def describe_numeric(df):
    return df.describe()


def describe_categorical(df):
    cat_cols = df.select_dtypes(include=['object']).columns
    summary = {}
    for col in cat_cols:
        summary[col] = {
            "unique": df[col].nunique(),
            "top_values": df[col].value_counts().head(5).to_dict()
        }
    return summary


def missing_data(df):
    missing = df.isnull().sum()
    pct = df.isnull().sum() / len(df) * 100
    return pd.DataFrame({"count": missing, "percent": pct}).sort_values("percent", ascending=False)
