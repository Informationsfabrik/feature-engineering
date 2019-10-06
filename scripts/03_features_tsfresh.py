"""Feature extraction and selection with TSFRESH."""
# %%
import sys

import pandas as pd
from tsfresh.utilities.dataframe_functions import roll_time_series, impute
from tsfresh import extract_features, feature_extraction, feature_selection

sys.path.append(".")
from feature_engineering.config import CNF_TARGET_COL, CNF_STATUS, CNF_TRAIN_TEST_SPLIT
from feature_engineering.tsfresh_config import (
    TSFRESH_FEATURE_WHITELIST,
    TSFRESH_TIME_WINDOWS,
)

# %% LOAD DATA
X_base_complete = pd.read_parquet("data/processed/X_algorithm_input_complete.parquet")
Y_target = pd.read_parquet("data/processed/Y_target.parquet")

# Concat features and target variable
df_complete = pd.concat([X_base_complete, Y_target], axis=1).reset_index()

# %% EXTRACTION OF TRAIN SET
# Cutoff points for partitioning data
ts_end = df_complete["Date"].max()
# noinspection PyUnresolvedReferences
ts_end_train = ts_end - pd.Timedelta(days=CNF_TRAIN_TEST_SPLIT)

X_train = df_complete.loc[df_complete["Date"] <= ts_end_train, :].drop(
    columns=[CNF_TARGET_COL]
)

Y_train = (
    df_complete.loc[df_complete["Date"] <= ts_end_train, [CNF_TARGET_COL, "Date"]]
    .set_index("Date")
    .squeeze()
)

X_complete = df_complete.drop(columns=[CNF_TARGET_COL])
Y_complete = df_complete[[CNF_TARGET_COL, "Date"]].set_index("Date").squeeze()

# %% DEFINITION OF FEATURE EXTRACTION MECHANISMS

# Define Features based on which new Features will be generated
FEATURES = X_train.drop(columns=["Date"]).columns

settings = feature_extraction.EfficientFCParameters()

# If in dev mode, then only use few features for faster execution
if CNF_STATUS == "dev":
    for key in list(settings.keys()):
        if key not in TSFRESH_FEATURE_WHITELIST:
            del settings[key]
else:
    settings = None


# %% APPLY TSFRESH FEATURE ENGINEERING


def create_new_features(df_x, s_y, x_train_cols=[]):
    """
    Create new Features from Input-Dataframe by using TSFRESH
    :param df_x: Dataframe containing Time-Series
    :param s_y: Series of Target-Var
    :param x_train_cols:
    :return: Dataframe containing created Features
    """

    # add id column (same id for every row, because only one time series is
    # considered in this dataset)
    df_x["id"] = 1

    # create roll time series for generating time series features
    df_x_rolled = roll_time_series(
        df_x,
        column_id="id",
        column_sort="Date",
        column_kind=None,
        rolling_direction=1,
        max_timeshift=TSFRESH_TIME_WINDOWS - 1,
    )

    x = df_x.set_index("Date")

    # for each variable in input df new features are generated
    for current_feature in FEATURES:
        # noinspection PyTypeChecker
        generated_features = extract_features(
            df_x_rolled,
            column_id="id",
            n_jobs=3,
            column_kind=None,
            column_value=current_feature,
            impute_function=impute,
            default_fc_parameters=settings,
        )

        x = pd.concat([x, generated_features], axis=1)
        print(f"\nNew shape of Feature-Matrix: {x.shape}")

    print(f"\nAmount of Features before selection: {len(x.columns)}")

    # check if features of train set are already selected
    if len(x_train_cols) == 0:
        # select relevant features for train set
        selected_features = feature_selection.select_features(x, s_y)
        print(
            f"\nAmount of Features after selection: "
            f"{len(selected_features.columns)}"
        )
    else:
        # no selection is needed, features are already selected for train set
        selected_features = x[x_train_cols]

    return selected_features


X_tsfresh_train = create_new_features(X_train, Y_train)
X_tsfresh_complete = create_new_features(
    X_complete, Y_complete, X_tsfresh_train.columns
)

# %% SAVE RESULTS
X_tsfresh_complete.to_parquet("data/processed/X_tsfresh_complete.parquet")

# %%
