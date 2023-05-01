import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# scaler
scaler = MinMaxScaler()

data = pd.read_csv("C://Users//white//Desktop//archive//housing.csv")

# remove nan data
data.fillna(0, inplace=True)

X = data.select_dtypes(include=np.number)  #only number data not string data
y = data.iloc[:, -1]    #target column i.e price range

X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns) # minmaxscaler and convert to pandas DataFrame

model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()