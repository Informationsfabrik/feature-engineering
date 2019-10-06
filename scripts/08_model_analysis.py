"""Analysis and visualization of model evaluation results."""
# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from feature_engineering.compare_config import CMP_SEEDS

SEED = CMP_SEEDS[0]

sns.set(style="darkgrid")


# %% LOAD DATA
preds = pd.read_parquet(f"data/processed/preds.parquet")
feature_importance = pd.read_parquet("data/processed/feature_importance.parquet")


def load_data(name):

    return {
        "name": name,
        "data": preds[(preds["algorithm"] == name) & (preds["seed"] == SEED)],
    }


# initialize experiments list
experiments = [
    load_data("base"),  # BASE
    load_data("tpot"),  # TPOT
    load_data("tpot_selection"),  # TPOT SELECTION
    load_data("tpot_base"),  # TPOT BASE
    load_data("tpot_selection_base"),  # TPOT SELECTION BASE
    load_data("tsfresh"),  # TSFRESH
    load_data("featuretools"),  # FEATURETOOLS
    load_data("featuretools_selection"),  # FEATURETOOLS SELECTION
    load_data("featuretools_base"),  # FEATURETOOLS BASE
    load_data("featuretools_selection_base"),  # FEATURETOOLS SELECTION BASE
    load_data("manual"),  # MANUAL
    load_data("manual_selection"),  # MANUAL SELECTION
    load_data("featuretools_base"),  # MANUAL BASE
    load_data("featuretools_selection_base"),  # MANUAL SELECTION BASE
]


# %% VISULIZATION
def plot_truth_vs_pred(preds_df):
    plt.figure()
    sns.regplot("y_pred", "y_true", data=preds_df["data"])
    sns.lineplot([0, 10000], [0, 10000], legend="brief")
    plt.xlim(0, 10000)
    plt.ylim(0, 10000)
    plt.legend(
        title=preds_df["name"],
        loc="upper left",
        labels=["Linear Regression", "Original Straight"],
    )
    plt.show()
    return plt


plots = list(map(plot_truth_vs_pred, experiments))

# %% COMPARE SELECTED FEATURES OF FEATURETOOLS
featuretools_df = pd.read_parquet(
    f"data/processed/feature_importance_featuretools.parquet"
).sort_values(by="information_gain")
selection_df = pd.read_parquet(
    f"data/processed/feature_importance_featuretools_selection.parquet"
).sort_values(by="information_gain")

print(featuretools_df.shape)
print(selection_df.shape)


# %% PLOT FEATURE IMPORTANCE
def plot_feature_importance(experiment):

    name = experiment["name"]

    plt.figure(figsize=[10, 8])
    df = feature_importance[
        (feature_importance["algorithm"] == name) & (feature_importance["seed"] == SEED)
    ]
    df_m_imp = df.sort_values(by="information_gain", ascending=False).head(10)
    df_m_imp["feature"] = df_m_imp["feature"].apply(lambda d: d + "_")
    sns.barplot("information_gain", "feature", data=df_m_imp)
    plt.title(name)
    plt.show()

    return plt


feature_importance_plots = list(map(plot_feature_importance, experiments))

# %%
