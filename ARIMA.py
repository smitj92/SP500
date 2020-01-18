%matplotlib inline
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import statsmodels.api as sm  
import seaborn as sb
from scipy import stats
sb.set_style('darkgrid')
from pmdarima import auto_arima 
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_absolute_error,mean_squared_error
import math

#import the csv and store in a dataframe
stock_data = pd.read_csv('C:/Users/Smit/Dataset/yahoo/stockMarket.csv')
newdata = stock_data.set_index('Date')
newdata = newdata.iloc[:,3]
newdata = pd.DataFrame(newdata)
summary = auto_arima(newdata['Close'],start_p=0,
                     start_q=0,max_p=3,max_q=3,seasonal=False,trace=True)
summary.summary()

trian = newdata.iloc[:1200]
test = newdata.iloc[1200:]
start = len(trian)
end = len(trian) + len(test) - 1
model_arima = ARIMA(trian['Close'],order=(0,1,0))
result_arima = model_arima.fit()
prediction= result_arima.predict(start=start,end=end,typ='levels')
prediction= pd.DataFrame(prediction)

test['prediction'] = prediction.values
test.plot()

mae = mean_absolute_error(prediction,test['Close'])
MAPE = np.mean(np.abs((test['Close'] - test['prediction']) / test['prediction'])) * 100
print('The Mean Absolute Percentage Error is {:.2f}%'.format(MAPE))
newdata.mean()
mse = mean_squared_error(prediction,test['Close'])
rmse = math.sqrt(mse)
