import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

#stock market dataset
df = pd.read_csv("C:/Users/Smit/Dataset/yahoo/stockMarket.csv", index_col = 0)

#twitter dataset
twitter_data = pd.read_csv('C:/Users/Smit/combined_csv1.csv')
dfVolume = twitter_data.groupby(['Date'])['Date'].count()
dfT = pd.Series.to_frame(dfVolume)
dfT['id'] = list(dfT.index)
dfT.columns = ['Count','DateTime']
dfT.to_csv('C:/Users/Smit/TWvol.csv')

dfGC = pd.DataFrame()
dfGC['StockPrice'] = df.Close
dfGC['TweetsCount'] = dfT.Count
dfGC['TweetsCount'] = dfGC['TweetsCount'].ffill()
#dfGC1 = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv', parse_dates=['date'])
#df['month'] = df.date.dt.month
grangercausalitytests(dfGC[['TweetsCount','StockPrice']], maxlag=3)

#######################################################################

#twitter dataset
dfPosRatio = pd.read_csv('C:/Users/Smit/PosRatioTweets.csv', index_col = 1)

dfGC1 = pd.DataFrame()
dfGC1['StockPrice'] = df.Close
dfGC1['Ratio'] = dfPosRatio.Ratio
dfGC1['Ratio'] = dfGC1['Ratio'].ffill()
grangercausalitytests(dfGC1[['Ratio','StockPrice']], maxlag=3)


#######################################################################
# stock price vs crude oil
dfSP = pd.read_csv("C:/Users/Smit/Dataset/yahoo/stockMarket.csv")

dfCO = pd.read_csv("C:/Users/Smit/Dataset/CrudeOil/COP.csv")
dfCO.columns = ['Date', 'ClosingVal']

combdf = pd.DataFrame()
combdf['StockPrice'] = dfSP.Close
combdf['CrudeOil'] = dfCO.ClosingVal
combdf['CrudeOil'] = combdf['CrudeOil'].ffill()

grangercausalitytests(combdf[['CrudeOil','StockPrice']], maxlag=3)

#######################################################################
# stock price vs exchange price

dfEP = pd.read_csv("C:/Users/Smit/Dataset/ExchangePrice/EUR_USD.csv")

combdf1 = pd.DataFrame()
combdf1['StockPrice'] = dfSP.Close
combdf1['ExchgP'] = dfEP.Price

grangercausalitytests(combdf1[['ExchgP','StockPrice']], maxlag=3)

#######################################################################

dfGP = pd.read_csv("C:/Users/Smit/Dataset/GoldPrices/GP.csv")

combdf2 = pd.DataFrame()
combdf2['StockPrice'] = dfSP.Close
combdf2['GP'] = dfGP.Price

grangercausalitytests(combdf2[['GP','StockPrice']], maxlag=3)

#######################################################################
