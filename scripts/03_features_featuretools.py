"""Applying Featuretools to engineer features."""
# %%
import sys

import pandas as pd
import featuretools as ft
import featuretools.selection

sys.path.append(".")
from feature_engineering.featuretools_config import (
    FT_MAX_DEPTH
)

# %% LOAD DATA
X_base_complete = pd.read_parquet(
    "data/processed/X_algorithm_input_complete.parquet"
).reset_index()
print(f"Shape before Featuretools: {X_base_complete.shape}")

date_col = X_base_complete["Date"].squeeze()

# %% DEFINITION OF BASEDATA
# Assign ID to provide unique continuous index for observations
transactions = X_base_complete.assign(ts_id=range(0, len(X_base_complete))).copy()

# %% BUILD ENTITIES AND RELATIONSHIPS
es = ft.EntitySet()

# Transactions
es = es.entity_from_dataframe(
    entity_id="transactions", dataframe=transactions, index="ts_id", time_index="Date"
)

# Customers
es = es.normalize_entity(
    base_entity_id="transactions",
    new_entity_id="stores",
    index="Store",
    additional_variables=[
        "CompetitionDistance",
        "CompetitionOpenSinceMonth",
        "CompetitionOpenSinceYear",
        "Promo2",
    ],
)

# %% DEFINITION OF FEATURE EXTRACTION MECHANISM
# Using timesteps as target Entity to create features for each transaction
feature_matrix, feature_defs = ft.dfs(
    entityset=es,
    target_entity="transactions",
    max_depth=FT_MAX_DEPTH,
    verbose=True,
    features_only=False
)

# Encode categorial features
feature_matrix_enc, _ = ft.encode_features(feature_matrix, feature_defs)

# Remove low information features
features_cleaned = featuretools.selection.remove_low_information_features(feature_matrix_enc)

# Transform dtypes of all columns into float of df
X_featuretools_complete = features_cleaned
X_featuretools_complete = (
    X_featuretools_complete.assign(Date=date_col).reset_index().set_index("Date")
)
print(f"Shape after applying Featuretools: {X_featuretools_complete.shape}")

# %% SAVE RESULTS
X_featuretools_complete.to_parquet("data/processed/X_featuretools_complete.parquet")

# %%
