"""Extract features with TPOT."""
# %%
import sys
from copy import copy

import pandas as pd
import pickle
from tpot import TPOTRegressor

sys.path.append(".")
from feature_engineering.config import CNF_TARGET_COL, CNF_TRAIN_TEST_SPLIT
from feature_engineering.tpot_config import (
    TPOT_REG_DICT,
    TPOT_MAX_TIME_MINS,
    TPOT_POPULATION_SIZE,
    TPOT_EARLY_STOP,
    TPOT_GENERATIONS,
    TPOT_CV,
    TPOT_RANDOM_STATE,
    TPOT_VERBOSITY,
    TPOT_N_JOBS,
)

# %% LOAD DATA
X_base_complete = pd.read_parquet("data/processed/X_algorithm_input_complete.parquet")
Y_target = pd.read_parquet("data/processed/Y_target.parquet")

# Concat features and target variable
df_complete = pd.concat([X_base_complete, Y_target], axis=1).reset_index()

# %% EXTRACTION OF TRAIN SET
# cutoff points for partitioning data
ts_end = df_complete["Date"].max()
ts_end_train = ts_end - pd.Timedelta(days=CNF_TRAIN_TEST_SPLIT)

# partitioning into train, valid, test
X_train = (
    df_complete[df_complete["Date"] <= ts_end_train]
    .drop(columns=[CNF_TARGET_COL, "Date"])
)

Y_train = (
    df_complete[df_complete["Date"] <= ts_end_train]
    .filter(items=[CNF_TARGET_COL, "Date"])
    .set_index("Date")
    .squeeze()
    .copy()
)

X_complete = df_complete.drop(columns=[CNF_TARGET_COL, "Date"])
Y_complete = df_complete[[CNF_TARGET_COL, "Date"]].set_index("Date").squeeze()

# %% OPTIMIZE PRE-PROCESSING PIPELINE
tpot_clf = TPOTRegressor(
    max_time_mins=TPOT_MAX_TIME_MINS,
    population_size=TPOT_POPULATION_SIZE,
    early_stop=TPOT_EARLY_STOP,
    generations=TPOT_GENERATIONS,
    cv=TPOT_CV,
    random_state=TPOT_RANDOM_STATE,
    verbosity=TPOT_VERBOSITY,
    n_jobs=TPOT_N_JOBS,
    config_dict=TPOT_REG_DICT,
)
tpot_clf.fit(X_train, Y_train)

# %% APPLY FEATURE EXTRACTION STEPS OF PIPELINE
pipeline_full = copy(tpot_clf.fitted_pipeline_)  # optimal pipeline as learned by TPOT
# test result
assert len(pipeline_full) >= 2, "learned pipeline does not have more than 1 step!"
# pipeline_full.steps[-1] returns tuple of 'xgbregressor' and its config
assert pipeline_full.steps[-1][0] == "xgbregressor", "last step is not model!"

# extract preprocessing from pipeline
pipeline_preprocessing = copy(pipeline_full)
pipeline_preprocessing.steps = pipeline_full.steps[:-1]
# test result
prepr_steps = [key for key, value in pipeline_preprocessing.steps]
assert "xgbregressor" not in prepr_steps, "xgbmodel learned before last pipeline step!"

X_tpot_complete = pipeline_preprocessing.transform(X_complete)

# %% EXPORT PIPELINE
with open('data/processed/tpot_pipeline.pickle', 'wb') as handle:
    pickle.dump(prepr_steps, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %% SAVE RESULTS
X_tpot_complete = pd.DataFrame.from_records(X_tpot_complete)
# Change type of integer col names to strings
X_tpot_complete.columns = X_tpot_complete.columns.map(str)
# Add InvoiceDate to dataframe
Y_complete = Y_complete.reset_index()
X_tpot_complete["Date"] = Y_complete["Date"]
X_tpot_complete = X_tpot_complete.set_index("Date")
# Save data in Parquet file
X_tpot_complete.to_parquet("data/processed/X_tpot_complete.parquet")

# %%
