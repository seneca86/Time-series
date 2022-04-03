## Time series

### Introduction

A time series is a sequence of measurements from a system that varies in time.

The example we will use is taken from a US researcher called Zachary M. Jones, who collected data from a web site called “Price of Weed” that crowdsources the price, quantity, quality, and location of cannabis transactions (http://www.priceofweed.com/). The goal of his project is to investigate the effect of policy decisions on markets.

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
