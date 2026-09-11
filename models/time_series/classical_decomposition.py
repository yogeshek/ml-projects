# Time series data refers to data that is collected, recorded orobserved over time in a sequential order
## Characteristics:
# 1. Chronological order: Observations are orderd in time
# 2. Sequential Dependency: The order of the data matters
# 3. Temporal components. Trend, Seasonaltity, Cycle, noise

# Time series analysis
    # Statistical techniques: meaningfull insights
    # forecasting
# Time series decomposition
    # Trend
    # sesonality 
    # noise
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
df = pd.read_csv(url,parse_dates=['Month'], index_col='Month')
print(df)

# df['Passengers'].plot(figsize=(12,5))
# plt.show()

stock_data = pd.read_excel('./data/Reliance.xlsx')
# print(stock_data)
stock_data.info()
stock_data.index = stock_data['Date']
print(stock_data)
stock_2020 = stock_data[stock_data['Date'].dt.year==2020]
print(stock_2020)
# stock_data['Close'].plot(figsize=(8,6))
# plt.show()

from statsmodels.tsa.seasonal import seasonal_decompose

sd= seasonal_decompose(stock_2020['Price'], model='multiplicative', period=10)

trend = sd.trend
seasonal = sd.seasonal
resd = sd.resid
# plt.figure(figsize=(14,10))
# plt.subplot(311)
# plt.plot(trend)
# plt.subplot(312)
# plt.plot(seasonal)
# plt.subplot(313)
# plt.plot(resd)
# plt.show()

estimated = trend + seasonal
# plt.plot(stock_2020['Close'],color="blue")
# plt.plot(estimated, color="red")
# plt.show()

from sklearn.metrics import mean_squared_error
print(mean_squared_error(stock_2020['Close'], estimated.fillna(0)))

##### STL DECOMPOSITION

# Seasonal Trend Loess(STL) Decomposition

from statsmodels.tsa.seasonal import STL

result = STL(stock_2020['Price'], period=10).fit()

plt.plot(result)
plt.show()

# plt.figure(figsize=(8,12))
# plt.subplot(511)
# plt.plot(estimated, color='blue')
# plt.subplot(512)
# plt.plot(result.trend + result.seasonal, color='red')
# plt.show()