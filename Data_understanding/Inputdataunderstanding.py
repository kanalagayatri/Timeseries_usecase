# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 18:06:49 2018

@author: Gayatri.k
"""


import numpy as np
import pandas as pd
import calendar
import matplotlib.pyplot as plt
import statsmodels.api as sm


df = pd.read_csv('\\trends1.csv')
df.head()
df.info()
df.date = pd.to_datetime(df.date)
df.set_index('date', inplace=True)
df=df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.columns= [ 'top1', 'top2', 'top3', 'date']
df.head()
#Graph plot of the data
df.plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.xlabel('Year', fontsize=20);
plt.ylabel('Tops', fontsize = 20)    
plt.savefig("\\Inputdata.png")      
df.corr()

#individual plots
df[['top 1']].plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.xlabel('Year', fontsize=20);
          
df[['top 2']].plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.xlabel('Year', fontsize=20);

df[['top 3']].plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.xlabel('Year', fontsize=20);

#Removing noise and plot the data          
top1 = df[['top 1']]
top1.rolling(12).mean().plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.xlabel('Year', fontsize=20);

top2 = df[['top 2']]
top2.rolling(12).mean().plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.xlabel('Year', fontsize=20);
          
top3 = df[['top 3']]
top3.rolling(12).mean().plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.xlabel('Year', fontsize=20);
          
df_rm = pd.concat([top1.rolling(12).mean(), top2.rolling(12).mean(), top3.rolling(12).mean()], axis=1)
df_rm.plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.xlabel('Year', fontsize=20)         

top3.diff().plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.xlabel('Year', fontsize=20)


df.diff().plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.xlabel('Year', fontsize=20);
          
df.diff().corr()  





ts_log = np.log(df)
moving_avg = pd.rolling_mean(ts_log,12)
plt.plot(ts_log)
plt.plot(moving_avg, color='red')      

ts_log_moving_avg_diff = ts_log - moving_avg
ts_log_moving_avg_diff.head(12)  

ts_log_moving_avg_diff.dropna(inplace=True)
ts_log_diff = ts_log - ts_log.shift()
plt.plot(ts_log_diff)



df = pd.read_csv('\\trends1.csv')
df.head()
df.info()
df.date = pd.to_datetime(df.date)
df=df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.columns= [ 'top1', 'top2', 'top3', 'date']
df['date']=pd.to_datetime(df['date'])
df['Month'] = df['date'].dt.month
df['Month'] = df['Month'].apply(lambda x: calendar.month_abbr[x])
df['Year'] = df['date'].dt.year
top1=df[['top1','date','Month','Year']].copy()
top2=df[['top2','date','Month','Year']].copy()
top3=df[['top3','date','Month','Year']].copy()
monthly_sales_data = pd.pivot_table(top1, values = "top1", columns = "Year", index = "Month")
monthly_sales_data = monthly_sales_data.reindex(index = ['Jan','Feb','Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
#print monthly_sales_data
monthly_sales_data.plot()

yearly_sales_data = pd.pivot_table(top1, values = "top1", columns = "Month", index = "Year")
yearly_sales_data = yearly_sales_data[['Jan','Feb','Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']]
#print yearly_sales_data
yearly_sales_data.boxplot()

top3=top3.set_index('date')
sales_ts=top3['top3']
decomposition = sm.tsa.seasonal_decompose(sales_ts, model='multiplicative')
fig = decomposition.plot()
fig.set_figwidth(12)
fig.set_figheight(8)
fig.suptitle('Decomposition of multiplicative time series')
plt.show()
