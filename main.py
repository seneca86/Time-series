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
df = pd.read_csv(".lesson/assets/mj-clean.csv", parse_dates=[5])
print(df.dtypes)
df.describe()
df.head()
# %%
seconds_in_a_year = 356 * 24 * 60 * 60
df_daily = (
    df.groupby(["quality", "date"], as_index=False)[["quality", "date", "ppg"]]
    .agg("mean")
    .assign(days=lambda x: (x.date - min(x.date)))
    .assign(years=lambda x: x.days.dt.total_seconds() / seconds_in_a_year)
)

# %%
sns.set_theme()
lineplot = sns.relplot(x="date", y="ppg", kind="line", row="quality", data=df_daily_predict)
lineplot.savefig("plots/lineplot.png")
plt.clf()
# %%
for q in df_daily.quality.unique():
    model = smf.ols("ppg ~ years", data=df_daily.query("quality == @q"))
    results = model.fit()
    print(f"quality = {q}")
    print(results.summary())
    print("+++++")
# %%
df = pd.DataFrame()
for q in df_daily.quality.unique():
    df_daily_predict = df_daily.query("quality == @q").assign(type="train")
    model = smf.ols("ppg ~ years", data=df_daily_predict)
    results = model.fit()
    df_prediction = (
        df_daily_predict.assign(ppg=results.fittedvalues)
        .assign(type="predict")
        .assign(quality=q)
    )
    df_fit_predict = pd.concat([df_daily_predict, df_prediction])
    df = pd.concat([df, df_fit_predict])

# %%
df.reset_index(inplace=True)
lineplot = sns.relplot(
    x="date", y="ppg", kind="line", row="quality", hue="type", data=df
)
lineplot.savefig("plots/lineplot_predict.png")
plt.clf()
# %% Moving averages
s1 = pd.Series(np.arange(10))
# %%
s1.rolling(window=3).mean()
# %%
df.rolling(window=30).mean('')