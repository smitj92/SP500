print('STOCK PREDICTION USING RNN LSTM')
import numpy as np
import pandas as pd
from sklearn import preprocessing
from functools import reduce
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic
from sklearn.metrics import mean_absolute_error,mean_squared_error
import math
%matplotlib inline
####
# load the dataset
####
SMdata= pd.read_csv('C:/Users/Smit/Dataset/yahoo/stockMarket.csv')
TWratio = pd.read_csv('C:/Users/Smit/PosRatioTweets.csv')
TWvol = pd.read_csv('C:/Users/Smit/TWvol.csv')
#merge all the datasets
data_frames = [SMdata, TWratio, TWvol]
data_csv = reduce(lambda  left,right: pd.merge(left,right,on=['Date'],
                                            how='outer'), data_frames)
data_csv = data_csv.drop(data_csv.index[1260:1640])
data_csv.isna().count()
data_csv = data_csv.drop('Unnamed: 0', axis=1)
data_csv = data_csv.drop('DateTime', axis=1)
data_csv = data_csv.fillna(0)
data_csv[['Close']].plot()
plt.show()
plt.clf()
########################################################
#VAR model
drop = ['Open','High','Low','Adj Close','Volume','Negative','Positive','Count']
data_csv = data_csv.drop(drop,axis=1)
data_csv = data_csv.set_index('Date')
data_csv.info()
nobs = 30
df_train, df_test = data_csv[0:-nobs], data_csv[-nobs:]
model = VAR(data_csv)
for i in [1,2,3,4,5,6,7,8,9]:
    result = model.fit(i)
    print('Lag Order =', i)
    print('AIC : ', result.aic)
    print('BIC : ', result.bic)
    print('FPE : ', result.fpe)
    print('HQIC: ', result.hqic, '\n')
x = model.select_order(maxlags=12)
x.summary()
model_fitted = model.fit(7)
model_fitted.summary()
forecast_input = df_train.values
fc = model_fitted.forecast(y=forecast_input, steps=nobs)
df_forecast = pd.DataFrame(fc, index=data_csv.index[-nobs:], columns=data_csv.columns + '_2d')
df_forecast
plt.plot(df_train['Close'])

mse = mean_squared_error(df_forecast['Close_2d'],df_test['Close'])
rmse = math.sqrt(mse)
mae = mean_absolute_error(df_forecast['Close_2d'],df_test['Close'])
MAPE = np.mean(np.abs((df_test['Close'] - df_forecast['Close_2d']) / df_forecast['Close_2d'])) * 100
print('The Mean Absolute Percentage Error is {:.2f}%'.format(MAPE))

msee = mean_squared_error(df_forecast['Ratio_2d'],df_test['Ratio'])
rmsee = math.sqrt(msee)
maee = mean_absolute_error(df_forecast['Ratio_2d'],df_test['Ratio'])
MAPEE = np.mean(np.abs((df_test['Ratio'] - df_forecast['Ratio_2d']) / df_forecast['Ratio_2d'])) * 100
print('The Mean Absolute Percentage Error is {:.2f}%'.format(MAPEE))

df_test['predicted_close'] = df_forecast['Close_2d']
df_test['predicted_ratio'] = df_forecast['Ratio_2d']

plt.plot(df_test['Close'], label='Actual Close')
plt.plot(df_test['predicted_close'], label='Predicted Close')
plt.legend()

plt.plot(df_test['Ratio'], label='Actual Ratio')
plt.plot(df_test['predicted_ratio'], label=['Predicted Ratio'])
plt.legend()