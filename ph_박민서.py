# pandas and numpy for data manipulation
import numpy as np
import pandas as pd

# ignore warnings from pandas
import warnings
warnings.filterwarnings('ignore')

# matplotlib for plot
import matplotlib.pyplot as plt

import seaborn as sns

# scikit-learn for linear_model, PolynomialFeatures, RobustScaler, LinearRegressionm, Pipeline, preprocessing
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn import preprocessing

# Read the data file
data=pd.read_excel('C://Users//white//Desktop//bmi_data_phw1.xlsx', sheet_name="dataset")

X=data.iloc[0,4] #independent columns
Y=data.iloc[:,-1] #target column, i.e., BMI

# Check if data is correctly printed
#print(data.head())

# make a dataframe
df = pd.DataFrame(data)

#=========================================================================================

# print dataset statistical data, feature names, data types
print("Dataset Statistical data, feature names, data types")
print(df.head())
print(df.dtypes)
print('='*50)

#=========================================================================================

bmi = df[['BMI']]
#print(bmi.head())

weight = df[['Weight (Pounds)']]
#print(weight.head())
height = df[['Height (Inches)']]
#print(height.head())

bmi_min = np.min(df['BMI'])
bmi_Max = np.max(df['BMI'])

#========================================

# plot height histograms for each BMI value(bins=10)
for i in range(int(bmi_min), int(bmi_Max)+1):
    mask = (i<=df['BMI']) & (df['BMI']<i+1)
    plt.hist(df['Height (Inches)'][mask], bins=10, alpha=0.5, label = 'Height')
# plot weight histograms for each BMI value(bins=10)
    plt.hist(df['Weight (Pounds)'][mask], bins=10, alpha=0.5, label = 'Weight')
    plt.legend()
    plt.title(f'BMI range: {i}~{i+1}')
    plt.show()

#=========================================================================================

# plot scaling results for height & weight using StandardScaler, MinMaxScaler, RobustScaler

plt.title('StandardScale, MinMaxScale, RobustScale')
plt.xlabel('Height (Inches)')
plt.ylabel('Weight (Pounds)')

# StandardScale
stScale = preprocessing.StandardScaler()
st_df = df.iloc[:, 2:4]
st_df2 = stScale.fit_transform(st_df)
plt.scatter(st_df2[:,0:1],st_df2[:,1:2], color='red')

# MinMaxScaler
minMaxSc = preprocessing.MinMaxScaler()
mM_df = df.iloc[:, 2:4]
mM_df2 = minMaxSc.fit_transform(mM_df)
plt.scatter(mM_df2[:,0:1],mM_df2[:,1:2], color='blue')

# RobustScaler
robust = preprocessing.RobustScaler()
rb_df = df.iloc[:, 2:4]
rb_df2 = robust.fit_transform(rb_df)
plt.scatter(rb_df2[:,0:1],rb_df2[:,1:2], color='green')

plt.xlim([-3,8])
plt.ylim([-5,5])
plt.show()

#=========================================================================================

# find outlier
# assume we don't know BMI
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# computing linear regression
lr_dt = data[['Height (Inches)', 'Weight (Pounds)']]
lr_dt_no_NaN= lr_dt.dropna()

lr_ht = lr_dt_no_NaN['Height (Inches)'].to_numpy()
lr_wt = lr_dt_no_NaN['Weight (Pounds)'].to_numpy()
lr = linear_model.LinearRegression()
lr.fit(lr_ht[:, np.newaxis], lr_wt)

# define test weight
lr_test_wt = lr.predict(lr_ht[:, np.newaxis])

x = np.array([lr_ht.min()-1, lr_ht.max()+1])
y = lr.predict(x[:, np.newaxis])

plt.scatter(lr_ht, lr_wt)
plt.plot(x, y, color='blue')
plt.scatter(lr_ht, lr_test_wt, color="red")
plt.show()

# e = w-w'
e = lr_wt - lr_test_wt
e_norm = stScale.fit_transform(e[:, np.newaxis])

plt.hist(e_norm, bins=10)
plt.xlabel("ze")
plt.ylabel('frequency')
plt.xticks([-2,-1,0,1,2])
plt.show()

# computing a
dt_bmi_is_0 = sum(data["BMI"]==0)
dt_bmi_is_4 = sum(data["BMI"]==4)

e_norm = np.sort(e_norm, axis=0)
# print(e_norm)
print(e_norm[dt_bmi_is_0])
print('-'*15)

# e_norm-dt_bmi_is_4
e_norm_bmi4 = len(e_norm)-dt_bmi_is_4-1
print(e_norm[e_norm_bmi4])
print('-'*15)

a = (np.abs(e_norm[dt_bmi_is_0]) + np.abs(e_norm[e_norm_bmi4]))/2
print('a={}'.format(a))
print('='*50)

#====================================================================

# Dividing gender
lr_sex = data[['Sex', 'Height (Inches)', 'Weight (Pounds)']]
lr_fm = lr_sex.loc[lr_sex["Sex"]=='Female']
lr_m = lr_sex.loc[lr_sex["Sex"]=='Male']

# dividing height weight by gender
lr_ht_fm = lr_fm['Height (Inches)'].to_numpy()
lr_wt_fm = lr_fm['Weight (Pounds)'].to_numpy()
lr_ht_m = lr_m['Height (Inches)'].to_numpy()
lr_wt_m = lr_m['Weight (Pounds)'].to_numpy()

lr_fm = linear_model.LinearRegression()
lr_fm.fit(lr_ht_fm[:, np.newaxis], lr_wt_fm)
lr_m = linear_model.LinearRegression()
lr_m.fit(lr_ht_m[:, np.newaxis], lr_wt_m)

# make test data for female, male
lr_test_ht_fm = lr_ht_fm
lr_test_ht_m = lr_ht_m
lr_test_wt_fm = lr_fm.predict(lr_test_ht_fm[:, np.newaxis])
lr_test_wt_m = lr_m.predict(lr_test_ht_m[:, np.newaxis])

x_fm = np.array([lr_ht_fm.min()-1, lr_ht_fm.max()+1])
y_fm = lr_fm.predict(x_fm[:, np.newaxis])

x_m = np.array([lr_ht_m.min()-1, lr_ht_m.max()+1])
y_m = lr_m.predict(x_m[:, np.newaxis])

plt.subplot(1,2,1)
plt.scatter(lr_ht_fm, lr_wt_fm)
plt.scatter(lr_test_ht_fm, lr_test_wt_fm, color="red")
plt.plot(x_fm, y_fm, color="black")

plt.subplot(1,2,2)
plt.scatter(lr_ht_m, lr_wt_m)
plt.scatter(lr_test_ht_m, lr_test_wt_m, color="red")
plt.plot(x_m, y_m, color="black")

plt.show()

#==================================

# histogram

e_fm = lr_wt_fm - lr_test_wt_fm
e_m = lr_wt_m - lr_test_wt_m

e_fm_norm = stScale.fit_transform(e_fm[:, np.newaxis])
e_m_norm = stScale.fit_transform(e_m[:, np.newaxis])

plt.subplot(1,2,1)
plt.hist(e_fm_norm, bins=10)
plt.xticks([-2,-1,0,1,2])
plt.title("e_Female")
plt.xlabel("ze")
plt.ylabel("frequency")

plt.subplot(1,2,2)
plt.hist(e_m_norm, bins=10)
plt.xticks([-2,-1,0,1,2])
plt.title("e_Male")
plt.xlabel("ze")
plt.ylabel("frequency")

plt.show()

#============================================================

# computing a for female
dt_bmi_0_fm = sum(dt_bmi_is_0 & (data['Sex'] == 'Female'))
dt_bmi_4_fm = sum(dt_bmi_is_4 & (data['Sex'] == 'Female'))

e_fm_norm = np.sort(e_fm_norm, axis=0)
print(e_fm_norm[dt_bmi_0_fm])

e_norm_bmi4_fm = len(e_fm_norm)-dt_bmi_4_fm-1
print(e_fm_norm[e_norm_bmi4_fm])

a_fm = ( np.abs(e_fm_norm[dt_bmi_0_fm]) + np.abs(e_fm_norm[e_norm_bmi4_fm]) )/2
print("a_Female = {}".format(a_fm))

# computing a for male
dt_bmi_0_m = sum(dt_bmi_is_0 & data['Sex']=='Male')
dt_bmi_4_m = sum(dt_bmi_is_4 & data['Sex']=='Male')

e_m_norm = np.sort(e_m_norm, axis=0)
print(e_m_norm[dt_bmi_0_m])

e_norm_bmi4_m = len(e_m_norm)-dt_bmi_4_m-1
print(e_m_norm[e_norm_bmi4_m])

a_m = ( np.abs(e_m_norm[dt_bmi_0_m]) + np.abs(e_m_norm[e_norm_bmi4_m]) )/2
print("a_Male = {}".format(a_m))

