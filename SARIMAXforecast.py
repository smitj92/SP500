# Import libraries
import warnings
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error,mean_squared_error
import math
from statsmodels.tools.eval_measures import rmse

plt.rcParams['figure.figsize'] = (20.0, 10.0)
plt.rcParams.update({'font.size': 12})
plt.style.use('ggplot')

################################################
# import the data
dataRead = pd.read_csv('C:/Users/Smit/Dataset/yahoo/stockMarket.csv',
                       engine='python', skipfooter=3)
data = pd.DataFrame()
data['Date'] = dataRead.Date
data['ClosingVal'] = dataRead.Close
data['Date']=pd.to_datetime(data['Date'], format='%d/%m/%Y')
data.set_index(['Date'], inplace=True)

# Plot the data
data.plot()
plt.ylabel('Stock index')
plt.xlabel('Date')
plt.show()
################################################
# Defining d and q parameters
q = d = range(0, 2)
# Defining p parameters
p = range(0, 4)
# Generating different combinations
pdq = list(itertools.product(p, d, q))

# different combinations of seasonal p, q and q
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
################################################

train_data = data['2013-06-28':'2017-06-28']
test_data = data['2017-06-29':'2018-06-26']
warnings.filterwarnings("ignore") # specify to ignore warning messages

AIC = []
SARIMAX_model = []
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(train_data,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            print('SARIMAX{}x{} - AIC:{}'.format(param, param_seasonal,
                  results.aic), end='\r')
            AIC.append(results.aic)
            SARIMAX_model.append([param, param_seasonal])
        except:
            continue

print('The smallest AIC is {} for model SARIMAX{}x{}'.format(min(AIC),
      SARIMAX_model[AIC.index(min(AIC))][0],
      SARIMAX_model[AIC.index(min(AIC))][1]))
###########################################################
# model fitting
mod = sm.tsa.statespace.SARIMAX(train_data,
                                order=SARIMAX_model[AIC.index(min(AIC))][0],
                                seasonal_order=SARIMAX_model[AIC.index(min(AIC))][1],
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()

###########################################################
results.plot_diagnostics(figsize=(20, 14))
plt.show()
###########################################################
##### PREDICTIONS

pred0 = results.get_prediction(start='2016-06-28', dynamic=False)
#pred0 = results.get_prediction(start='2017-06-28', dynamic=False)
pred0_ci = pred0.conf_int()

pred1 = results.get_prediction(start='2016-06-28', dynamic=True)
pred1_ci = pred1.conf_int()

date1 = '2017-11-19'
date2 = '2018-07-26'
#date1 = '2018-06-28'
#date2 = '2018-09-25'
mydates = pd.date_range(date1, date2).tolist()

pred2 = results.get_forecast(steps=250,index=mydates) ## steps = 12
pred2_ci = pred2.conf_int()
print(pred2.predicted_mean['2017-11-19':'2018-07-26'])
#print(pred2.predicted_mean['2018-06-28':'2018-09-25'])
###########################################################

ax = data.plot(figsize=(20, 16))
pred0.predicted_mean.plot(ax=ax, label='1-step-ahead Forecast (get_predictions, dynamic=False)')
pred1.predicted_mean.plot(ax=ax, label='Dynamic Forecast (get_predictions, dynamic=True)')
pred2.predicted_mean.plot(ax=ax, label='Dynamic Forecast (get_forecast)')
ax.fill_between(pred2_ci.index, pred2_ci.iloc[:, 0], pred2_ci.iloc[:, 1], color='k', alpha=.1)
plt.ylabel('Stock Market Index')
plt.xlabel('Date')
plt.legend()
plt.show()

###########################################################

prediction = pred2.predicted_mean['2017-11-19':'2018-07-26'].values
#prediction = pred2.predicted_mean['2017-06-27':'2018-03-03'].values
# flatten nested list
truth = list(itertools.chain.from_iterable(test_data.values))

error = rmse(truth, prediction)
error

# Mean Absolute Percentage Error
MAPE = np.mean(np.abs((truth - prediction) / truth)) * 100
mae = mean_absolute_error(prediction,truth)

print('The Mean Absolute Percentage Error for the forecast of year 2018-19 is {:.2f}%'.format(MAPE))

mse = mean_squared_error(prediction,truth)
rmse = math.sqrt(mse)


