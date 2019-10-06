"""Define experimental datasets to compare them with prior generated feature sets"""
# %%

import pandas as pd


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
    load_data("tsfresh"),  # TSFRESH
    load_data("featuretools"),  # FEATURETOOLS
    load_data("manual"),  # MANUAL
]


# %% EXPERIMENT 1: ADD BASE FEATURES TO GENERATED FEATURE SETS

# Load base features
base_features = load_data("base")


def join_feat_frames(gen_feat, base_feat):
        """Extend base features for a specific split (train, valid, test)"""
        # keep prior column names
        new_columns = gen_feat.columns.append(base_feat.columns)

        # append base features to current experiment df
        joined_feat = pd.concat([gen_feat, base_feat], axis=1, ignore_index=True)
        joined_feat.columns = new_columns
        # Prevent duplicated columns
        result_df = joined_feat.loc[:, ~joined_feat.columns.duplicated()]

        return result_df


def join_generated_base_feat(gen_feat):
    """Extend generated features with base ones"""
    joined_feat_train = join_feat_frames(gen_feat["train"], base_features["train"])
    joined_feat_valid = join_feat_frames(gen_feat["valid"], base_features["valid"])
    joined_feat_test = join_feat_frames(gen_feat["test"], base_features["test"])

    extend_result = {
        "name": gen_feat["name"],
        "train": joined_feat_train,
        "valid": joined_feat_valid,
        "test": joined_feat_test,
    }

    return extend_result


def save_base_extended_dfs(df_result):
    """Save partioned feat frames in parquet file"""
    df_result["train"].to_parquet(
        f"data/processed/X_{df_result['name']}_base_train.parquet"
    )
    df_result["valid"].to_parquet(
        f"data/processed/X_{df_result['name']}_base_valid.parquet"
    )
    df_result["test"].to_parquet(
        f"data/processed/X_{df_result['name']}_base_test.parquet"
    )


# append base features to each df with generated features
joined_feat_frames = map(join_generated_base_feat, feature_dfs)

# iterate over results to store new data
for joined_feat_frame in joined_feat_frames:
    save_base_extended_dfs(joined_feat_frame)

# %%
