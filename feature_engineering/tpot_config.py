"""Dictionaries that configure TPOT."""

import numpy as np

# How many minutes TPOT has to optimize the pipeline.
TPOT_MAX_TIME_MINS = None  # default
# Number of individuals to retain in the GP population every generation.
TPOT_POPULATION_SIZE = 100  # default
# How many generations TPOT checks whether there is no improvement in optimization.
TPOT_EARLY_STOP = None  # default
# Number of iterations to run the pipeline optimization process.
TPOT_GENERATIONS = 100  # default
# Number of folds to evaluate each pipeline over in cross-validation during optimization
TPOT_CV = 5  # default
# Random number generator seed for reproducibility.
TPOT_RANDOM_STATE = 42
# How much information TPOT communicates while it is running.
TPOT_VERBOSITY = 2
# Number of CPUs for evaluating pipelines in parallel during optimization.
TPOT_N_JOBS = 3
# config of functions that can be used when optimizing for regression.
TPOT_REG_DICT = {
    # model
    "xgboost.XGBRegressor": {
        # 'n_estimators': [10, 50, 100],
        # 'max_depth': range(1, 11),
        # 'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        # 'subsample': np.arange(0.05, 1.01, 0.05),
        # 'min_child_weight': range(1, 21),
        # 'nthread': [1]
    },
    # Preprocesssors
    "sklearn.preprocessing.Binarizer": {"threshold": np.arange(0.0, 1.01, 0.05)},
    "sklearn.decomposition.FastICA": {"tol": np.arange(0.0, 1.01, 0.05)},
    "sklearn.cluster.FeatureAgglomeration": {
        "linkage": ["ward", "complete", "average"],
        "affinity": ["euclidean", "l1", "l2", "manhattan", "cosine"],
    },
    "sklearn.preprocessing.MaxAbsScaler": {},
    "sklearn.preprocessing.MinMaxScaler": {},
    "sklearn.preprocessing.Normalizer": {"norm": ["l1", "l2", "max"]},
    "sklearn.kernel_approximation.Nystroem": {
        "kernel": [
            "rbf",
            "cosine",
            "chi2",
            "laplacian",
            "polynomial",
            "poly",
            "linear",
            "additive_chi2",
            "sigmoid",
        ],
        "gamma": np.arange(0.0, 1.01, 0.05),
        "n_components": range(1, 11),
    },
    "sklearn.decomposition.PCA": {
        "svd_solver": ["randomized"],
        "iterated_power": range(1, 11),
    },
    "sklearn.preprocessing.PolynomialFeatures": {
        "degree": [2],
        "include_bias": [False],
        "interaction_only": [False],
    },
    "sklearn.kernel_approximation.RBFSampler": {"gamma": np.arange(0.0, 1.01, 0.05)},
    "sklearn.preprocessing.RobustScaler": {},
    "sklearn.preprocessing.StandardScaler": {},
    "tpot.builtins.ZeroCount": {},
    "tpot.builtins.OneHotEncoder": {
        "minimum_fraction": [0.05, 0.1, 0.15, 0.2, 0.25],
        "sparse": [False],
        "threshold": [10],
    },
    # Selectors
    "sklearn.feature_selection.SelectFwe": {
        "alpha": np.arange(0, 0.05, 0.001),
        "score_func": {"sklearn.feature_selection.f_regression": None},
    },
    "sklearn.feature_selection.SelectPercentile": {
        "percentile": range(1, 100),
        "score_func": {"sklearn.feature_selection.f_regression": None},
    },
    "sklearn.feature_selection.VarianceThreshold": {
        "threshold": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    },
    "sklearn.feature_selection.SelectFromModel": {
        "threshold": np.arange(0, 1.01, 0.05),
        "estimator": {
            "sklearn.ensemble.ExtraTreesRegressor": {
                "n_estimators": [100],
                "max_features": np.arange(0.05, 1.01, 0.05),
            }
        },
    },
}

# TPOT_CLF_DICT = {
#     "xgboost.XGBClassifier": {
#         "n_estimators": [10],
#         "max_depth": [5],
#         "learning_rate": [0.1],
#         "subsample": [0.8],
#         "nthread": [4],
#     },
#     # Preprocesssors
#     "sklearn.preprocessing.Binarizer": {"threshold": np.arange(0.0, 1.01, 0.05)},
#     "sklearn.decomposition.FastICA": {"tol": np.arange(0.0, 1.01, 0.05)},
#     "sklearn.cluster.FeatureAgglomeration": {
#         "linkage": ["ward", "complete", "average"],
#         "affinity": ["euclidean", "l1", "l2", "manhattan", "cosine"],
#     },
#     "sklearn.preprocessing.MaxAbsScaler": {},
#     "sklearn.preprocessing.MinMaxScaler": {},
#     "sklearn.preprocessing.Normalizer": {"norm": ["l1", "l2", "max"]},
#     "sklearn.kernel_approximation.Nystroem": {
#         "kernel": [
#             "rbf",
#             "cosine",
#             "chi2",
#             "laplacian",
#             "polynomial",
#             "poly",
#             "linear",
#             "additive_chi2",
#             "sigmoid",
#         ],
#         "gamma": np.arange(0.0, 1.01, 0.05),
#         "n_components": range(1, 11),
#     },
#     "sklearn.decomposition.PCA": {
#         "svd_solver": ["randomized"],
#         "iterated_power": range(1, 11),
#     },
#     "sklearn.preprocessing.PolynomialFeatures": {
#         "degree": [2],
#         "include_bias": [False],
#         "interaction_only": [False],
#     },
#     "sklearn.kernel_approximation.RBFSampler": {"gamma": np.arange(0.0, 1.01, 0.05)},
#     "sklearn.preprocessing.RobustScaler": {},
#     "sklearn.preprocessing.StandardScaler": {},
#     "tpot.builtins.ZeroCount": {},
#     "tpot.builtins.OneHotEncoder": {
#         "minimum_fraction": [0.05, 0.1, 0.15, 0.2, 0.25],
#         "sparse": [False],
#         "threshold": [10],
#     },
#     # Selectors
#     "sklearn.feature_selection.SelectFwe": {
#         "alpha": np.arange(0, 0.05, 0.001),
#         "score_func": {"sklearn.feature_selection.f_classif": None},
#     },
#     "sklearn.feature_selection.SelectPercentile": {
#         "percentile": range(1, 100),
#         "score_func": {"sklearn.feature_selection.f_classif": None},
#     },
#     "sklearn.feature_selection.VarianceThreshold": {
#         "threshold": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
#     },
#     "sklearn.feature_selection.RFE": {
#         "step": np.arange(0.05, 1.01, 0.05),
#         "estimator": {
#             "sklearn.ensemble.ExtraTreesClassifier": {
#                 "n_estimators": [100],
#                 "criterion": ["gini", "entropy"],
#                 "max_features": np.arange(0.05, 1.01, 0.05),
#             }
#         },
#     },
#     "sklearn.feature_selection.SelectFromModel": {
#         "threshold": np.arange(0, 1.01, 0.05),
#         "estimator": {
#             "sklearn.ensemble.ExtraTreesClassifier": {
#                 "n_estimators": [100],
#                 "criterion": ["gini", "entropy"],
#                 "max_features": np.arange(0.05, 1.01, 0.05),
#             }
#         },
#     },
# }
