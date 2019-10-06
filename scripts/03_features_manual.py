"""Extract features manually."""
# %%
import sys

import pandas as pd

sys.path.append(".")
from feature_engineering.config import CNF_TARGET_COL

# %% LOAD DATA
X_base_complete = pd.read_parquet("data/processed/X_base_complete.parquet")
Y_target = pd.read_parquet("data/processed/Y_target.parquet")

# Concat features and target variable
df_complete = pd.concat([X_base_complete, Y_target], axis=1)


# %% EXTRACT FEATURES
def create_lags(df, column, title, num_lags):
    """Create lagged features"""
    df[f"{column}_{title}_lag_{num_lags}"] = df[column].shift(periods=num_lags)
    return df


# create lags: daily up to one week, weekly up to one month
lags = list(range(1, 7)) + list(range(7, 29, 7))
print(f"creating lags for shifting periods: {lags}")
for i in lags:
    df_complete = create_lags(df_complete, CNF_TARGET_COL, "target_daily", i)

# %% DROP TARGET COLUMN
X_manual_complete = df_complete.drop(columns=[CNF_TARGET_COL])

# %% SAVE RESULTS
X_manual_complete.to_parquet("data/processed/X_manual_complete.parquet")

# %%
