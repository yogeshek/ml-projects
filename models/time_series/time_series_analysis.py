import pandas as pd
from datetime import datetime

print(datetime(2025,5,5))

my_dates = ["27/12/2025","05/11/2025", "12/05/2026"]
print(my_dates)

a = pd.to_datetime(my_dates, format = "%d/%m/%Y")
print(a)
print(a.month)
print(a.year)
print(a.day)


# url = "https://drive.google.com/uc?id=1ugXf9514sOZx5izMY7Mt6_HX8doCQLcO"
# df = pd.read_csv(url)
# print(df)

# df = pd.read_csv(r"C:/Users/YogeshEkambaramK/Downloads/Preprocessing3.csv")
# print(df)

# import os
# os.makedirs("data", exist_ok=True)
# import shutil
# shutil.copy(r"C:/Users/YogeshEkambaramK/Downloads/Preprocessing3.csv", "data/")
print("*************")
df= pd.read_csv('./data/Preprocessing3.csv')
# print(df)

# df.info()
# print(df['Date'].max())
# print(df['Date'].dt.day)
df['Date']=pd.to_datetime(df['Date'])
# df.info()
# print(df['Date'].max())
# print(df['Date'].dt.year)

df_new = pd.DataFrame(df['Date'])
# print(df_new)
# print(df_new.info())

df_new['year']=df_new["Date"].dt.year
df_new['month']=df_new['Date'].dt.month
df_new['day']=df_new['Date'].dt.day

print(df_new)

##################################33

# import yfinance as yf
# ticker_symbol = 'RELIANCE.NS'
# data = yf.download("ticker_symbol", start="2020-01-01", end="2025-01-01")
# print(data)
 ### another way of donloadin
 
import pandas as pd

# ticker = "RELIANCE.NS"
# url = f"https://query1.finance.yahoo.com/v7/finance/download/RELIANCE.NS?period1=1577836800&period2=1735689600&interval=1d&events=history"
# data = pd.read_csv(url)
# print(data)

### manually import

import matplotlib.pyplot as plt
stock_data = pd.read_excel('./data/Reliance.xlsx')
print(stock_data)


x = stock_data.loc[(stock_data['Date'].dt.year==2024) & (stock_data['Date'].dt.month==1),'Date']
y = stock_data.loc[(stock_data['Date'].dt.year==2024) & (stock_data['Date'].dt.month==1),'Close']

plt.plot(x, y)
plt.xlabel('Date')
plt.show()
