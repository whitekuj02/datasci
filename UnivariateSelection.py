import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler

# scaler
scaler = MinMaxScaler()

data = pd.read_csv("C://Users//white//Desktop//archive//housing.csv")

# remove nan data
data.fillna(0, inplace=True)

X = data.select_dtypes(include=np.number)  #only number data not string data
y = data.iloc[:, -1]    #target column i.e price range

X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns) # minmaxscaler and convert to pandas DataFrame

pd.set_option('display.max_columns', None)
print(X.head())
#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k="all")
fit = bestfeatures.fit(X, y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(10,'Score'))  #print 10 best features
