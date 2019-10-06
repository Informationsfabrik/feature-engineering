"""Prepare data set."""
# %%
import sys

import numpy as np
import pandas as pd

sys.path.append(".")
from feature_engineering.utils.preprocessing import parse_date
from feature_engineering.config import CNF_TARGET_COL

# %% LOAD DATA
input_df_train = (
    pd.read_csv(
        "data/input/rossmann/train.csv",
        sep=",",
        encoding="utf-8",
        parse_dates=["Date"],
        date_parser=parse_date,
        dtype={
            "StateHoliday": str,
            "DayOfWeek": str,
            "Assortment": str,
            "StoreType": str,
            "CompetitionOpenSinceMonth": int,
            "CompetitionOpenSinceYear": int,
        },
    )
    .query("Store == 1")
    .drop(columns=["Customers"])
    .assign(year=lambda d: d.Date.dt.year)
    .assign(month=lambda d: d.Date.dt.month)
    .sort_values(by="Date")
)

input_df_store = (
    pd.read_csv("data/input/rossmann/store.csv", sep=",", encoding="utf-8")
    .query("Store == 1")  # TODO: use more than one store
    .drop(columns=["Promo2SinceWeek", "Promo2SinceYear", "PromoInterval"])  # removes columns with unstructured and missing data
)
# Drop for considered store == 1(later keep columns and impute)

# Join Dataframes
input_df = (
    input_df_train.set_index("Store")
    .join(
        input_df_store.set_index("Store"),
        how="left",
        lsuffix="_train",
        rsuffix="_store",
    )
    .reset_index()
)

# %% ONE-HOT-ENCODING
time_series = pd.get_dummies(
    input_df, columns=["DayOfWeek", "StateHoliday", "Assortment", "StoreType"],
    dtype=np.int64
)

# %% CREATE LAGGED TARGET VARIABLE
time_series["target_lagged_by_1"] = time_series[CNF_TARGET_COL].shift(periods=1)

# Cut off first row due to null value for lagged target variable
time_series.drop(time_series.head(1).index, inplace=True)

# %% EXPORT BASE DATA SET
# Extract and save target variable
Y_target = time_series.loc[:, ["Date", CNF_TARGET_COL]].set_index("Date")
Y_target.to_parquet("data/processed/Y_target.parquet")

# Remove target variable and save base features
X_base_complete = time_series.drop(columns=[CNF_TARGET_COL, "target_lagged_by_1"]).set_index("Date")
X_base_complete.to_parquet("data/processed/X_base_complete.parquet")

# %% EXPORT INPUT DATA SET FOR FEATURE ALGORITHMS

# Remove target variable and save data for algorithms as input
X_algorithm_input_complete = time_series.drop(columns=[CNF_TARGET_COL]).set_index("Date")
X_algorithm_input_complete.to_parquet("data/processed/X_algorithm_input_complete.parquet")


#%%
