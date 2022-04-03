# Time series

## Introduction

A time series is a sequence of measurements from a system that varies in time.

The example we will use is taken from a US researcher called Zachary M. Jones, who collected data from a web site called “Price of Weed” that crowdsources the price, quantity, quality, and location of cannabis transactions (http://www.priceofweed.com/). The goal of his project is to investigate the effect of policy decisions on markets.

## Exploratory analysis

Let's first import a few libraries that we will use during the lesson, and load the dataframe from a text file.

```python
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
```

```python
df_raw = pd.read_csv(".lesson/assets/mj-clean.csv", parse_dates=[5])
print(df_raw.dtypes)
df_raw.describe()
df_raw.head()
```

The DataFrame has a row for each reported transaction and the following
columns:

• city: string city name

• state: two-letter state abbreviation.

• price: price paid in dollars

• amount: quantity purchased in grams

• quality: high, medium, or low quality, as reported by the purchaser

• date: date of report, presumed to be shortly after date of purchase

• ppg: price per gram, in dollars

• state.name: string state name

• lat: approximate latitude of the transaction, based on city name

• lon: approximate longitude of the transaction

Next thing we should do it to transform each group into an equally spaced series by computing the mean daily price per gram. We also should transform the date into a continuous variable, for instance the years from the beginning of the series.

```python
seconds_in_a_year = 356 * 24 * 60 * 60
df_train = (
    df_raw.groupby(["quality", "date"], as_index=False)[["quality", "date", "ppg"]]
    .agg("mean")
    .assign(days=lambda x: (x.date - min(x.date)))
    .assign(years=lambda x: x.days.dt.total_seconds() / seconds_in_a_year)
    .assign(type="train")
)
```

`groupby` and `agg` are DataFrame methods that group a dataframe along certain variables and then calculate the summary statistics of others. Since the values of quality are low, medium, and high, we get three groups with those names.

It is always good to visualize the dataframe to understand the type of dataset we are dealing with.

```python
sns.set_theme()
lineplot = sns.relplot(x="date", y="ppg", kind="line", row="quality", data=df_train)
lineplot.savefig("plots/lineplot.png")
plt.clf()
```

Visually, it looks like the price of high quality cannabis is declining during this period, and the price of medium quality is increasing. The price of low quality might also be increasing, but it is harder to tell, since it seems to be more volatile.

## Linear regression for time series

Although there are methods specific to time series analysis, for many problems a simple way to get started is by applying general-purpose tools like linear regression. Let's build a loop that cycles through the different qualities of drug and computes a least squares fit for the prices, returning the model and results objects from StatsModels:

```python
for q in df_train.quality.unique():
    model = smf.ols("ppg ~ years", data=df_train.query("quality == @q"))
    results = model.fit()
    print(f"quality = {q}")
    print(results.summary())
```

The estimated slopes indicate that the price of high quality cannabis dropped by about 71 cents per year during the observed interval; for medium quality it increased by 28 cents per year, and for low quality it increased by 57 cents per year. These estimates are all statistically significant with very small p-values.

We can enhance the loop in a way that the predictions are stored into a dataframe that we can plot:

```python
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
```

Let's plot the prediction:

```python
df_ols.reset_index(drop=True, inplace=True)
lineplot = sns.relplot(
    x="date", y="ppg", kind="line", row="quality", hue="type", data=df_ols
)
lineplot.savefig("plots/lineplot_predict.png")
plt.clf()
```

The model seems like a good linear fit for the data; nevertheless, linear regression is not the most appropriate choice for this data:

* First, there is no reason to expect the long-term trend to be linear. In general, prices are determined by supply and demand in a non-linear way

* Second, the linear regression model gives equal weight to all data, recent and past; often we should give more weight to recent data.

* Finally, one of the assumptions of linear regression is that the residuals are uncorrelated noise. With time series data, this assumption is almost always false because successive values are correlated.

Let's look for an alternative that is more appropriate for time series data.

## Moving averages and exponential smoothing



```python
s1 = pd.Series(np.arange(10))
s1.rolling(window=3).mean()
```



