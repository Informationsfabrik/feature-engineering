"""Partition data: keep a test set hidden from feature extraction."""
# %%
import sys

import pandas as pd

sys.path.append(".")
from feature_engineering.config import (
    CNF_TRAIN_TEST_SPLIT,
    CNF_TEST_VALID_SPLIT,
    CNF_TARGET_COL,
)


# %% LOAD GENERATED FEATURE SETS
def load_data(name):
    df_complete = pd.read_parquet(f"data/processed/X_{name}_complete.parquet")

    return {"name": name, "data": df_complete}


# initialize feature set list
feature_dfs = [
    load_data("base"), # BASE
    load_data("tpot"), # TPOT
    load_data("tsfresh"), # TSFRESH
    load_data("featuretools"), # FEATURETOOLS
    load_data("manual"), # MANUAL
]

# %% LOAD TARGET VARIABLE
Y_target = pd.read_parquet("data/processed/Y_target.parquet")

# %% DEFINE CUTOFF DATE
cutoff_date = feature_dfs[0]["data"].reset_index().loc[:, "Date"].min()


# %% PARTITION DATA
def partition_data(feature_df):
    """Partition feature dataframe in train, valid, test."""

    name = feature_df["name"]
    df = feature_df["data"].copy()

    df["Date"] = df.index.values

    # concat features and target variable
    df = pd.concat([df, Y_target], axis=1)

    df = df[df["Date"] >= cutoff_date].copy()

    # cutoff points for partitioning data
    ts_end = df["Date"].max()
    ts_end_valid = ts_end - pd.Timedelta(days=CNF_TEST_VALID_SPLIT)
    ts_end_train = ts_end - pd.Timedelta(days=CNF_TRAIN_TEST_SPLIT)

    # partitioning into train, valid, test
    df_train = df[df.Date < ts_end_train].drop("Date", axis=1)
    df_valid = df[(df.Date >= ts_end_train) & (df.Date < ts_end_valid)].drop(
        "Date", axis=1
    )
    df_test = df[df.Date >= ts_end_valid].drop("Date", axis=1)

    assert len(df) == len(df_train) + len(df_valid) + len(
        df_test
    ), "partitioning change quantity of data"

    # extract target variable
    Y_target_train = df_train.loc[:, [CNF_TARGET_COL]]
    Y_target_valid = df_valid.loc[:, [CNF_TARGET_COL]]
    Y_target_test = df_test.loc[:, [CNF_TARGET_COL]]

    # remove target
    X_df_train = df_train.drop(columns=[CNF_TARGET_COL])
    X_df_valid = df_valid.drop(columns=[CNF_TARGET_COL])
    X_df_test = df_test.drop(columns=[CNF_TARGET_COL])

    # save partitioned target in separate files
    partition_result = {
        "name": name,
        "X_train": X_df_train,
        "X_valid": X_df_valid,
        "X_test": X_df_test,
        "Y_train": Y_target_train,
        "Y_valid": Y_target_valid,
        "Y_test": Y_target_test,
    }

    return partition_result


# %% CREATE SETS FOR EACH ALGORITHM
def save_partitioned_sets(df_result):
    # save partitioned features
    df_result["X_train"].to_parquet(
        f"data/processed/X_{df_result['name']}_train.parquet"
    )
    df_result["X_valid"].to_parquet(
        f"data/processed/X_{df_result['name']}_valid.parquet"
    )
    df_result["X_test"].to_parquet(f"data/processed/X_{df_result['name']}_test.parquet")

    # save partitioned target variable
    df_result["Y_train"].to_parquet(f"data/processed/Y_target_train.parquet")
    df_result["Y_valid"].to_parquet(f"data/processed/Y_target_valid.parquet")
    df_result["Y_test"].to_parquet(f"data/processed/Y_target_test.parquet")


# select features based on TSFresh feature selection
data_partition_results = map(partition_data, feature_dfs)

# iterate over results to store new feature sets
for data_partition_result in data_partition_results:
    save_partitioned_sets(data_partition_result)

# %%
