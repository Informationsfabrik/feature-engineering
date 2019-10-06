"""Comparing the features that were extracted in the previous steps."""
# %%
import sys
from math import sqrt

import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import itertools as iter

sys.path.append(".")
from feature_engineering.compare_config import (
    CMP_XGB_PARAMS,
    CMP_NUM_BOOST_ROUND,
    CMP_EARLY_STOPPING_ROUNDS,
    CMP_SEEDS,
)
from feature_engineering.config import CNF_TARGET_COL

# Cols for evaluation matrix
EVAL_COLUMNS = ["algorithm", "seed", "mse", "rmse", "r2_score"]
PREDS_COLUMNS = ["algorithm", "seed", "y_true", "y_pred"]
FEATURE_IMPORTANCE_COLUMNS = ["algorithm", "seed", "feature", "information_gain"]


# %% LOAD GENERATED FEATURE SETS OF EXPERIMENTS
def load_data(name):
    """Load splitted datasets for one experiment"""
    train = pd.read_parquet(f"data/processed/X_{name}_train.parquet")
    valid = pd.read_parquet(f"data/processed/X_{name}_valid.parquet")
    test = pd.read_parquet(f"data/processed/X_{name}_test.parquet")

    # create dict for single experiment
    exp_dict = {"name": name, "train": train, "valid": valid, "test": test}

    return exp_dict


# initialize experiments list
experiments = [
    load_data("base"),  # BASE
    load_data("tpot"),  # TPOT
    load_data("tpot_selection"),  # TPOT SELECTION
    load_data("tpot_base"),  # TPOT BASE
    load_data("tpot_base_selection"),  # TPOT SELECTION BASE
    load_data("tsfresh"),  # TSFRESH
    load_data("tsfresh_base"),  # TSFRESH BASE
    load_data("featuretools"),  # FEATURETOOLS
    load_data("featuretools_selection"),  # FEATURETOOLS SELECTION
    load_data("featuretools_base"),  # FEATURETOOLS BASE
    load_data("featuretools_base_selection"),  # FEATURETOOLS SELECTION BASE
    load_data("manual"),  # MANUAL
    load_data("manual_selection"),  # MANUAL SELECTION
    load_data("manual_base"),  # MANUAL BASE
    load_data("manual_base_selection"),  # MANUAL SELECTION BASE
]


# %% LOAD TARGET VALUES
def load_target():
    """Load target values independently from generated feature sets"""
    y = {}
    for split in ["train", "valid", "test"]:
        y[split] = (
            pd.read_parquet(f"data/processed/Y_target_{split}.parquet")
            .loc[:, CNF_TARGET_COL]
            .squeeze()
            .copy()
        )
    return y


Y_target = load_target()

# %% TRAIN MODEL AND EVALUATE FEATURES


def evaluate_features(experiment_seed_tuple):
    """
    Evaluate Feature Matrix for each algorithm by using an independent
    XG-Boost Regressor.
    :param experiment: Dict containing Name, Features(train, vaild, test)
    """

    experiment = experiment_seed_tuple[0]
    seed = experiment_seed_tuple[1]

    X_train = experiment["train"]
    X_valid = experiment["valid"]
    X_test = experiment["test"]

    Y_train = Y_target["train"]
    Y_valid = Y_target["valid"]
    Y_test = Y_target["test"]

    def df_to_dmatrix(features, target):
        x = features.drop(columns=["Date"], errors="ignore")  # Ignore if not exist
        y = target
        dmatrix = xgb.DMatrix(x, label=y)
        return dmatrix

    dm_train = df_to_dmatrix(X_train, Y_train)
    dm_valid = df_to_dmatrix(X_valid, Y_valid)
    dm_test = df_to_dmatrix(X_test, Y_test)

    # set seed in XGBoost params
    CMP_XGB_PARAMS["seed"] = seed

    # Determine optimal model size
    evals = [(dm_train, "train"), (dm_valid, "valid")]
    model_bst = xgb.train(
        params=CMP_XGB_PARAMS,
        dtrain=dm_train,
        evals=evals,
        num_boost_round=CMP_NUM_BOOST_ROUND,
        early_stopping_rounds=CMP_EARLY_STOPPING_ROUNDS,
    )
    best_ntree_limit = model_bst.best_ntree_limit

    # OPTIONAL: Append train and valid set and train on both sets

    # Retrain on all training data
    evals2 = [(dm_train, "train"), (dm_test, "test")]
    model_final = xgb.train(
        params=CMP_XGB_PARAMS,
        dtrain=dm_train,
        evals=evals2,
        num_boost_round=best_ntree_limit,
    )

    # Feature importance (Information Gain)
    feature_information_gain = model_final.get_score(importance_type="gain")
    feature_importance = pd.DataFrame(
        list(feature_information_gain.items()), columns=["feature", "information_gain"]
    )
    feature_importance["algorithm"] = experiment["name"]
    feature_importance["seed"] = seed
    # Reorder columns
    feature_importance = feature_importance[FEATURE_IMPORTANCE_COLUMNS]

    # Predict values of test set
    y_pred = model_final.predict(dm_test)
    y_true = dm_test.get_label()
    preds = pd.DataFrame()
    preds["y_true"] = y_true
    preds["y_pred"] = y_pred
    preds["algorithm"] = experiment["name"]
    preds["seed"] = seed
    # Reorder columns
    preds = preds[PREDS_COLUMNS]

    # Calculate and save error metrics
    r2 = r2_score(y_true=y_true, y_pred=y_pred)
    mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
    rmse = sqrt(mse)
    metrics = pd.DataFrame(
        [[experiment["name"], seed, mse, rmse, r2]], columns=EVAL_COLUMNS
    )

    eval_dict = {
        "name": experiment["name"],
        "metrics": metrics,
        "preds": preds,
        "feature_importance": feature_importance,
    }

    return eval_dict


# %% GENERATE AND SAVE EVALUATION RESULTS FOR FEATURE SETS

# build cartesian product with experiments and seeds
experiments_seeds_tuples = list(iter.product(experiments, CMP_SEEDS))

# evaluate features for each experiment feature set
exp_eval_results = list(map(evaluate_features, experiments_seeds_tuples))

# initialize empty df for evaluation metrics
model_metrics = pd.DataFrame(columns=EVAL_COLUMNS)
preds = pd.DataFrame(columns=PREDS_COLUMNS)
feature_importance = pd.DataFrame(columns=FEATURE_IMPORTANCE_COLUMNS)

# TODO: OPTIONAL initialize three global df to fill with exp. results

# save evaluation artefacts for each experiment
for exp_eval_result in exp_eval_results:
    model_metrics = model_metrics.append(exp_eval_result["metrics"])
    preds = preds.append(exp_eval_result["preds"])
    feature_importance = feature_importance.append(
        exp_eval_result["feature_importance"]
    )

# save model metrics in separate file
model_metrics.to_parquet("data/processed/model_metrics.parquet")
model_metrics.to_csv("results/model_metrics.csv")

# save prediction vs. truth data
preds.to_parquet("data/processed/preds.parquet")

# save feature importance data
feature_importance.to_parquet("data/processed/feature_importance.parquet")


# %%
