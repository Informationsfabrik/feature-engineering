"""Visualizing time series."""
# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(style="darkgrid")

# %% LOAD DATA
X_base_complete = pd.read_parquet("data/processed/X_base_complete.parquet")
Y_target = pd.read_parquet("data/processed/Y_target.parquet")

# Concat features and target variable
df_complete = pd.concat([X_base_complete, Y_target], axis=1).reset_index()

# %% PLOT TIME-SERIES
g = sns.relplot(x="Date", y="Sales", kind="line", data=df_complete)
g.fig.autofmt_xdate()
plt.show()

# %% OVERLAY MONTHLY SUM
plt.figure()

monthly_agg = df_complete.groupby(["month", "year"]).sum().reset_index().copy()

sns.relplot(
    x="month", y="Sales", hue="year", data=monthly_agg, kind="line", legend="full"
)
plt.show()

# %% OVERLAY MONTHLY INTERVAL
plt.figure()

sns.relplot(x="month", y="Sales", hue="year", kind="line", data=df_complete, legend="full")
plt.show()

# %%
