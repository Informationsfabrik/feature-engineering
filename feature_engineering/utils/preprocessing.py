import pandas as pd


def time_features(dt_series: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({
        'year': dt_series.dt.year,
        'month': dt_series.dt.month,
        'week': dt_series.dt.week,
        'weekday': dt_series.dt.weekday,
        'hour': dt_series.dt.hour,
    })
    df_dummies = pd.get_dummies(df, prefix='weekday', columns=['weekday'])
    return df_dummies


def parse_date(x):
    return pd.datetime.strptime(x, '%Y-%m-%d')
