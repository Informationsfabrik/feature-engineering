# Params for XG-Boost used to compare Featuresets
CMP_XGB_PARAMS = {
    'booster': 'gbtree',
    'colsample_bytree': 0.9,
    'learning_rate': 0.05,
    'max_depth': 5,
    'min_child_weight': 5,
    'eval_metric': 'rmse',
    'seed': 3210
}

# Number of boosting rounds
CMP_NUM_BOOST_ROUND = 1000

# Early Stopping rounds
CMP_EARLY_STOPPING_ROUNDS = 100

# Seeds
CMP_SEEDS = [123, 234, 456, 34, 1334, 978, 1919, 98, 2647, 12, 9, 455, 23, 44, 665]