"""Create feature sets based on TSFresh feature selection."""
# %%
import sys

import pandas as pd
from tsfresh import feature_selection

sys.path.append(".")
from feature_engineering.config import CNF_TARGET_COL


# %% LOAD DATA
def load_data(name):
    train = pd.read_parquet(f"data/processed/X_{name}_train.parquet")
    valid = pd.read_parquet(f"data/processed/X_{name}_valid.parquet")
    test = pd.read_parquet(f"data/processed/X_{name}_test.parquet")

    # create dict for single feature set
    exp_dict = {"name": name, "train": train, "valid": valid, "test": test}

    return exp_dict


# initialize feature set list
feature_dfs = [
    load_data("tpot"),  # TPOT
    load_data("tpot_base"),  # TPOT BASE
    load_data("featuretools"),  # FEATURETOOLS
    load_data("featuretools_base"),  # FEATURETOOLS BASE
    load_data("manual"),  # MANUAL
    load_data("manual_base")  # MANUAL BASE
]

# %% LOAD TARGET VARIABLE

Y_train = (
        pd.read_parquet(f"data/processed/Y_target_train.parquet")
        .loc[:, CNF_TARGET_COL]
        .squeeze()
        .copy()
    )


# %% SELECT FEATURES
def select_features(feature_set):
    """Select features in the train set based on TSfresh feature selection."""
    df_train = feature_set["train"]
    df_valid = feature_set["valid"]
    df_test = feature_set["test"]

    # drop cols with NaN values
    df_train = df_train.drop(df_train.columns[df_train.isna().any()].tolist(), axis=1)

    # define Features and Target Dataset
    X_train = df_train.copy()
    
    # TSFresh feature selection
    selected_features = feature_selection.select_features(X_train, Y_train)

    # reduce dfs to selected features
    df_train_sel = df_train[selected_features.columns]
    df_valid_sel = df_valid[selected_features.columns]
    df_test_sel = df_test[selected_features.columns]

    print(f"Shape after Selection: {df_train_sel.shape}")

    return {
        "name": feature_set["name"],
        "train": df_train_sel,
        "valid": df_valid_sel,
        "test": df_test_sel,
    }


# %% CREATE SELECTED FEATURE SETS FOR EACH ALGORITHM
def save_selected_features(df_result):
    df_result["train"].to_parquet(
        f"data/processed/X_{df_result['name']}_selection_train.parquet"
    )
    df_result["valid"].to_parquet(
        f"data/processed/X_{df_result['name']}_selection_valid.parquet"
    )
    df_result["test"].to_parquet(
        f"data/processed/X_{df_result['name']}_selection_test.parquet"
    )


# select features based on TSFresh feature selection
feature_selection_results = map(select_features, feature_dfs)

# iterate over results to store new feature sets
for feature_selection_result in feature_selection_results:
    save_selected_features(feature_selection_result)


# %%
