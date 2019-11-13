# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 11:25:35 2018

@author: Gayatri.k
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as smt
import statsmodels.graphics.gofplots as sm
from statsmodels.tsa.stattools import adfuller
import warnings
from pandas import Series
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMAResults
from math import sqrt
from math import log
from math import exp
from scipy.stats import boxcox
import statsmodels.api as sma
import itertools
import sys



df = pd.read_csv('D:\\Usecase\\trends1.csv')
df.columns = ['Sno', 'top1', 'top2', 'top3', 'date']
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

split_point = len(df.top3) - 12
dataset, validation = df[0:split_point], df[split_point:]

X1 = df.top1
X2 = df.top2
X3 = df.top3
X1=np.log(X1)
X2=np.log(X2)
X3=np.log(X3)
split = len(df.top1) / 2
X11, X12 = X1[0:split], X1[split:]
X21, X22 = X2[0:split], X2[split:]
X31, X32 = X3[0:split], X3[split:]
mean11, mean12 = X11.mean(), X12.mean()
mean21, mean22 = X21.mean(), X22.mean()
mean31, mean32 = X31.mean(), X32.mean()
var11, var12 = X11.var(), X12.var()
var21, var22 = X21.var(), X22.var()
var31, var32 = X31.var(), X32.var()
print('mean11=%f, mean12=%f' % (mean11, mean12))
print('variance11=%f, variance12=%f' % (var11, var12))
print('mean21=%f, mean22=%f' % (mean21, mean22))
print('variance21=%f, variance22=%f' % (var21, var22))
print('mean31=%f, mean32=%f' % (mean31, mean32))
print('variance31=%f, variance32=%f' % (var31, var32))


X1=np.diff(df.top1) #1st level of differentiation d = 1
X2=np.diff(df.top2) #1st level of differentiation d = 1
X3=np.diff(df.top3) #1st level of differentiation d = 1
result1 = adfuller(X1)
result2 = adfuller(X2)
result3 = adfuller(X3)

print('ADF Statistic 1: %f' % result1[0])
print('p-value 1: %f' % result1[1])
print('Critical Values 1:')
for key, value in result1[4].items():
	print('\t%s: %.3f' % (key, value))

print('ADF Statistic 2 %f' % result2[0])
print('p-value 2: %f' % result2[1])
print('Critical Values 2:')
for key, value in result2[4].items():
	print('\t%s: %.3f' % (key, value))
    
print('ADF Statistic 3: %f' % result3[0])
print('p-value 3: %f' % result3[1])
print('Critical Values 3:')
for key, value in result3[4].items():
	print('\t%s: %.3f' % (key, value))
    
    
def ts_diagnostics(y, lags=None, title='', filename=''):
    '''
    Calculate acf, pacf, qq plot and Augmented Dickey Fuller test for a given time series
    '''
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
        
    # weekly moving averages (5 day window because of workdays)
    rolling_mean = pd.rolling_mean(y, window=12)
    rolling_std = pd.rolling_std(y, window=12)
    
    fig = plt.figure(figsize=(14, 12))
    layout = (3, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))
    qq_ax = plt.subplot2grid(layout, (2, 0))
    hist_ax = plt.subplot2grid(layout, (2, 1))
    
    # time series plot
    y.plot(ax=ts_ax)
    rolling_mean.plot(ax=ts_ax, color='crimson');
    rolling_std.plot(ax=ts_ax, color='darkslateblue');
    plt.legend(loc='best')
    ts_ax.set_title(title, fontsize=24);
    
    # acf and pacf
    smt.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
    smt.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5) 
    
    # qq plot
    sm.qqplot(y, line='s', ax=qq_ax)
    qq_ax.set_title('QQ Plot')
    
    # hist plot
    y.plot(ax=hist_ax, kind='hist', bins=25);
    hist_ax.set_title('Histogram');
    plt.tight_layout();
    plt.savefig('D:/Usecase/{}.png'.format(filename))
    plt.show()
    
    # perform Augmented Dickey Fuller test
    print('Results of Dickey-Fuller test:')
    dftest = adfuller(y, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['test statistic', 'p-value', '# of lags', '# of observations'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    return   

ts_diagnostics(X1,lags=10,title="TOP1",filename="TOP1")
ts_diagnostics(X2,lags=10,title="TOP2",filename="TOP2")
ts_diagnostics(X3,lags=10,title="TOP3",filename="TOP3")

plt.figure()
plt.subplot(211)
smt.plot_acf(df.top1, ax=plt.gca(), lags=30, alpha=0.5)
plt.subplot(212)
smt.plot_pacf(df.top1, ax=plt.gca(), lags=30, alpha=0.5)
plt.show()




'''
evaluate an ARIMA model for a given order (p,d,q) and return RMSE
'''
def evaluate_arima_model(X, arima_order):
	# prepare training dataset
	X = X.astype('float32')
	train_size = int(len(X) * 0.50)
	train, test = X[0:train_size], X[train_size:]
	history = [x for x in train]
	# make predictions
	predictions = list()
	for t in range(len(test)):
		model = ARIMA(history, order=arima_order)
		model_fit = model.fit(disp=0)
		yhat = model_fit.forecast()[0]
		predictions.append(yhat)
		history.append(test[t])
	# calculate out of sample error
	mse = mean_squared_error(test, predictions)
	rmse = sqrt(mse)
	return rmse

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
	dataset = dataset.astype('float32')
	best_score, best_cfg = float("inf"), None
	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p,d,q)
				try:
					mse = evaluate_arima_model(dataset, order)
					if mse < best_score:
						best_score, best_cfg = mse, order
					print('ARIMA%s MSE=%.3f' % (order,mse))
				except:
					continue
	print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))

# load dataset
series = dataset.top3
# evaluate parameters
p_values = range(0,5)
d_values = range(0,2)
q_values = range(0,5)
warnings.filterwarnings("ignore")
evaluate_models(series.values, p_values, d_values, q_values)
#For top1 (4,1,0)
#For top2 (1,1,0)
#For top3 (0,1,1)



# invert box-cox transform
def boxcox_inverse(value, lam):
	if lam == 0:
		return exp(value)
	return exp(log(lam * value + 1) / lam)

# load data
series = dataset.top3
# prepare data
X = series.values
X = X.astype('float32')
train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]
# walk-foward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
	# transform
	transformed, lam = boxcox(history)
	if lam < -5:
		transformed, lam = history, 1
	# predict
	model = ARIMA(transformed, order=(0,1,1))
	model_fit = model.fit(disp=0)
	yhat = model_fit.forecast()[0]
	# invert transformed prediction
	yhat = boxcox_inverse(yhat, lam)
	predictions.append(yhat)
	# observation
	obs = test[i]
	history.append(obs)
	print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
# report performance
mse = mean_squared_error(test, predictions)
rmse = sqrt(mse)
print('RMSE: %.3f' % rmse)
#RMSE for top1 = 2.600
#RMSE for top2 = 5.921
#RMSE for top3 = 3.273


'''
Model fitting
'''
series = dataset.top3
# prepare data
X = series.values
X = X.astype('float32')
# transform data
transformed, lam = boxcox(X)
# fit model
model = ARIMA(transformed, order=(0,1,1))
model_fit = model.fit(disp=0)
# save model
model_fit.save('D:\Usecase\model_top3.pkl')
np.save('D:\Usecase\model_lambda_top3.npy', [lam])

'''
Model validation
'''
def boxcox_inverse(value, lam):
	if lam == 0:
		return exp(value)
	return exp(log(lam * value + 1) / lam)
 
# load and prepare datasets
dataset = dataset.top3
X = dataset.values.astype('float32')
history = [x for x in X]
validation = validation.top3
y = validation.values.astype('float32')
# load model
model_fit = ARIMAResults.load('D:\Usecase\model_top3.pkl')
lam = np.load('D:\Usecase\model_lambda_top3.npy')
# make first prediction
predictions = list()
yhat = model_fit.forecast()[0]
yhat = boxcox_inverse(yhat, lam)
predictions.append(yhat)
history.append(y[0])
print('>Predicted=%.3f, Expected=%3.f' % (yhat, y[0]))
# rolling forecasts
for i in range(1, len(y)):
    	# transform
    	transformed, lam = boxcox(history)
       
    	if lam < -5:
    		transformed, lam = history, 1
    	# predict
    	model = ARIMA(transformed, order=(0,1,1))
    	model_fit = model.fit(disp=0)
    	yhat = model_fit.forecast()[0]
    	# invert transformed prediction
    	yhat = boxcox_inverse(yhat, lam)
    	predictions.append(yhat)
    	# observation
    	obs = y[i]
    	history.append(obs)
    	print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
# report performance
mse = mean_squared_error(y, predictions)
rmse = sqrt(mse)
print('RMSE: %.3f' % rmse)
plt.plot(y)
plt.plot(predictions, color='red')
plt.savefig('D:/Usecase/{}.png'.format("TOP3validation"))
plt.show()



