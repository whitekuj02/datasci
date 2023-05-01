import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# scaler
scaler = MinMaxScaler()

data = pd.read_csv("C://Users//white//Desktop//archive//housing.csv")

# remove nan data
data.fillna(0, inplace=True)

X = data.select_dtypes(include=np.number)  #only number data not string data
y = data.iloc[:, -1]    #target column i.e price range

X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)# minmaxscaler and convert to pandas DataFrame

#get correlations of each features in dataset
corrmat = data.corr(numeric_only=True)
top_corr_features = corrmat.index
plt.figure(figsize=(20, 20))

#plot heat map
g=sns.heatmap(data[top_corr_features].corr(), annot=True, cmap="RdYlGn")
plt.show()