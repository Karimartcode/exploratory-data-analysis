import pandas as pd
import numpy as np


def load_dataset(filepath):
    df = pd.read_csv(filepath)
    return df
