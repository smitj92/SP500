#sentiment analysis
%matplotlib inline
import os  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import statsmodels.api as sm  
import seaborn as sb
import re
sb.set_style('darkgrid')

#import the csv and store in a dataframe
twitter_data = pd.read_csv('C:/Users/Smit/combined_csv1.csv')

#count for volume
dfVolume = twitter_data.groupby(['Date'])['Date'].count()
a = pd.Series.to_frame(dfVolume)
a['id'] = list(a.index)
a.columns = ['Count','DateTime']

a['Count'].plot(figsize=(16, 12))
a['DateTime'] = pd.to_datetime(a.DateTime)
a.sort_values('DateTime') # This now sorts in date order
a.to_csv('C:/Users/Smit/VolTweets.csv')

#count for -ve and +ve sentiments grouped by date
dfDate = pd.DataFrame()
dfDate['Sentiments_Count'] = None
dfDate['Sentiments_Count'] = twitter_data.groupby(['Date', 'Sentiments'])['Sentiments'].count()
dfDate = dfDate.reset_index()
neg = dfDate[dfDate['Sentiments']=='Negative']
pos = dfDate[dfDate['Sentiments']=='Positive']
negcount = neg.groupby('Date').sum().reset_index()
poscount = pos.groupby('Date').sum().reset_index()

allcount = negcount.merge(poscount,on=['Date'],how='outer')
allcount = allcount.set_index('Date')
allcount.fillna(0,inplace=True)
allcount = allcount.rename(columns={'Sentiments_Count_x':'Negative','Sentiments_Count_y':'Positive'})
allcount.isna().sum()
allcount['Ratio'] = allcount['Positive']/(allcount['Positive']+allcount['Negative'])
allcount['Ratio'] = allcount['Ratio'].round(2)
allcount = allcount.sort_index()

allcount = allcount.reset_index()
allcount.to_csv('C:/Users/Smit/PosRatioTweets.csv')#
for i in range(0,len(allcount)):
    if(allcount.loc[i,'Positive']>allcount.loc[i,'Negative']):
        allcount.loc[i,'sentiment'] = 1
    else:
        allcount.loc[i,'sentiment'] = 0
        
allcount = allcount.set_index('Date')

print(allcount.groupby('sentiment').count())

allcount = allcount.reset_index()
dfg = dfGC.reset_index()

for i in range(0, len(dfg)-1):
    value = dfg.loc[i+1, 'StockPrice'] - dfg.loc[i, 'StockPrice']
    if(value > 1):
        dfg.loc[i+1,'trend'] = 'Up Trend'
    else:
        dfg.loc[i+1,'trend'] = 'Down Trend'
        
merged = allcount.merge(dfg,on=['Date'],how='outer')
merged.drop(merged[merged['trend'].isna() == True].index, inplace = True)
merged.reset_index(drop=True)
        
# SVM model
from sklearn.svm import SVC
SVM = SVC(C=1.0, kernel='linear')
merged.drop(merged[merged['sentiment'].isna() == True].index, inplace = True)
merged.reset_index(drop=True)
X = merged[['StockPrice','sentiment']]
Y = merged['trend']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4)
SVM.fit(X_train, y_train)
y_pred = SVM.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)

# RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier()
RF.fit(X_train, y_train)
y_pred_rf = RF.predict(X_test)
accuracy_rf = accuracy_score(y_test,y_pred_rf)

from sklearn.model_selection import cross_val_score
clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(RF, X, Y, cv=10)
scores.mean()
