import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sklearn.datasets as sk

scaler = StandardScaler() #standardScaler

data = pd.read_csv("C://Users//white//Desktop//archive//housing.csv")
data.fillna(0, inplace=True) # NaN data replace 0
X = data.select_dtypes(include=np.number) # only number data

data_scaled = scaler.fit_transform(X) # scaler

print(data.head())
# data2 = sk.fetch_california_housing(data_home=None, download_if_missing=True, return_X_y =False, as_frame=False)
#
# df = pd.DataFrame(data2.data, columns=data2.feature_names)
# df['target'] = data2.target
# pd.set_option('display.max_columns', None)
#
# print(df)
# data_scaled = scaler.fit_transform(df)

pca = PCA(n_components=2) #PCA feature 2
pca.fit(data_scaled)
df_pca = pca.transform(data_scaled)
print(df_pca.shape)

# df_pca = pd.DataFrame(data=df_pca)
# df_pca["target"] = df.target
# print(df_pca.head())

df_pca = pd.DataFrame(data=df_pca)
df_pca["target"] = data.ocean_proximity # target data (ocean_proximity) concat
print(df_pca.head())
print(df_pca['target'].unique())


# scatter print
colors = ['red', 'green', 'blue', 'yellow', 'purple']
targets = ['NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND']

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_title('2 Component PCA')

for target, color in zip(targets,colors):
    indicesToKeep = df_pca['target'] == target
    ax.scatter(df_pca.loc[indicesToKeep, 0], df_pca.loc[indicesToKeep, 1], c = color, s = 5)

ax.legend(targets)
ax.grid()
plt.show()