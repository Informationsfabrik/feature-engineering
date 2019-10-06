# Current status -> dev settings for fast execution
CNF_STATUS = "prod" # dev or prod

# Column of Dataset which is used as Target column for XG Boost
CNF_TARGET_COL = "Sales"

# Direction and amount of steps the Target column is shifted
CNF_SHIFT_SIZE = -1

# Cut off days from end of Dataset -> Part before used as Train-Set, Part after
# is used for Valid and Test-Set
CNF_TRAIN_TEST_SPLIT = 200 # Days

# Cut off days from end of Dataset -> Part before used as Valid-Set, Part after
# is used as Test-Set
CNF_TEST_VALID_SPLIT = 100 # Days