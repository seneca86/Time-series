# %%
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib as mpl
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from pathlib import Path

mpl.rcParams["figure.dpi"] = 150

Path("plots").mkdir(parents=True, exist_ok=True)

# %%
df_raw = pd.read_csv(".lesson/assets/mj-clean.csv", parse_dates=[5])
print(df_raw.dtypes)
df_raw.describe()
df_raw.head()
# %%
seconds_in_a_year = 356 * 24 * 60 * 60
df_train = (
    df_raw.groupby(["quality", "date"], as_index=False)[["quality", "date", "ppg"]]
    .agg("mean")
    .assign(days=lambda x: (x.date - min(x.date)))
    .assign(years=lambda x: x.days.dt.total_seconds() / seconds_in_a_year)
    .assign(type="train")
)

# %%
sns.set_theme()
lineplot = sns.relplot(x="date", y="ppg", kind="line", row="quality", data=df_train)
lineplot.savefig("plots/lineplot.png")
plt.clf()
# %%
for q in df_train.quality.unique():
    model = smf.ols("ppg ~ years", data=df_train.query("quality == @q"))
    results = model.fit()
    print(f"quality = {q}")
    print(results.summary())
    print("+++++")
# %%
df_ols = pd.DataFrame()
for q in df_train.quality.unique():
    df_q_train = df_train.query("quality == @q")
    model = smf.ols("ppg ~ years", data=df_q_train)
    results = model.fit()
    df_prediction = (
        df_q_train.assign(ppg=results.fittedvalues)
        .assign(type="predict")
        .assign(quality=q)
    )
    df_fit_predict = pd.concat([df_q_train, df_prediction])
    df_ols = pd.concat([df_ols, df_fit_predict])

# %%
df_ols.reset_index(drop=True, inplace=True)
lineplot = sns.relplot(
    x="date", y="ppg", kind="line", row="quality", hue="type", data=df_ols
)
lineplot.savefig("plots/lineplot_predict.png")
plt.clf()
# %% Moving averages
s1 = pd.Series(np.arange(10))
# %%
s1.rolling(window=3).mean()
# %%
df_rolling = pd.DataFrame()
for q in df_train.quality.unique():
    df_q_train = df_train.query("quality==@q")
    df_q_rolling = df_q_train.copy()
    df_q_rolling["ppg"] = df_q_rolling["ppg"].rolling(window=30).mean()
    df_q_rolling["type"] = "rolling"
    df_q = pd.concat([df_q_train, df_q_rolling])
    df_rolling = pd.concat([df_rolling, df_q])

# %%
df_rolling.reset_index(drop=True, inplace=True)
lineplot = sns.relplot(
    x="date", y="ppg", kind="line", row="quality", hue="type", data=df_rolling
)
lineplot.savefig("plots/lineplot_rolling.png")
plt.clf()
# %% EWMA
df_ewma = pd.DataFrame()
for q in df_train.quality.unique():
    df_q_train = df_train.query("quality==@q")
    df_q_ewma = df_q_train.copy()
    df_q_ewma["ppg"] = df_q_ewma["ppg"].ewm(span=30).mean()
    df_q_ewma["type"] = "ewma"
    df_q = pd.concat([df_q_train, df_q_ewma])
    df_ewma = pd.concat([df_ewma, df_q])
# %%
df_ewma.reset_index(drop=True, inplace=True)
lineplot = sns.relplot(
    x="date", y="ppg", kind="line", row="quality", hue="type", data=df_ewma
)
lineplot.savefig("plots/lineplot_ewma.png")
plt.clf()
# %%
import statsmodels.tsa.stattools as smtsa
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

acf_values = smtsa.acf(df_train.query('quality=="high"').ppg, nlags=365)
# %%
pd.plotting.autocorrelation_plot(
    df_rolling.query('quality=="high"').dropna().query('type=="rolling"').ppg
)
# %%
pd.plotting.autocorrelation_plot(
    df_train.query('quality=="high"').dropna().query('type=="train"').ppg
)
# %%
pd.plotting.autocorrelation_plot(
    df_ewma.query('quality=="high"').dropna().query('type=="ewma"').ppg
)
# %%
df_flights = pd.read_csv(".lesson/assets/airline-passengers.csv")
# %%
df_flights["Passengers_diff"] = df_flights["Passengers"].diff(periods=1)
# %%
plt.plot(df_flights.Passengers)
plt.plot(df_flights.Passengers_diff)
# %%
acf_values = acf(df_flights.dropna()["Passengers_diff"])
print(acf_values)
plot_acf(df_flights.dropna().Passengers_diff, lags=30)
# %%
pacf_values = pacf(df_flights.dropna()["Passengers_diff"])
print(pacf_values)
plot_pacf(df_flights.dropna().Passengers_diff, lags=30)

# %%
