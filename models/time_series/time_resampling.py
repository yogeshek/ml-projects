import pandas as pd


# date range
week_range = pd.date_range(start='2026-01-01', end='2026-05-31', freq='W')
# print(week_range)
bussiness_days = pd.date_range(start='2026-06-01', end='2026-07-31', freq='C')
# print(bussiness_days)

df= pd.read_csv('./data/Preprocessing3.csv') #, parse_dates=True, index_col='Date')
# print(df)
# print(df.iloc[5])
# print(df.loc[5])
# print(df.index.is_monotonic_increasing) # index is 0,1... always true


df.index =pd.to_datetime(df['Date'])
df = df.drop('Date', axis=1)
# print(df)

# print(df.iloc[5])
# print(df.loc['2009-01-03'])

# print(df.sort_index())
# print(df.index)
# print(df.index.is_monotonic_increasing) # index is Date, may or may not be true
df.sort_index()

monthly = df['Sale Price'].resample(rule='M').min()
print(monthly)
import matplotlib.pyplot as plt
df['Sale Price'].resample(rule='W').max().plot(kind='bar',figsize=(6,8))
plt.show()

