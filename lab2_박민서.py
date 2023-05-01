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
data=pd.read_csv('C://Users//white//Desktop//bmi_data_lab2.csv')

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

# Identifying all dirty records
# 1. number of likely-wrong values // height: 6, weight: 3
# 2. missing height weight bmi values // height: 6, weight: 1, bmi: 5
# 3. both 1.&2. // 3

# Remove all likely-wrong values - i.e. making them NaN
# height
for i in range(0,len(height)):
    if( (df['Height (Inches)'][i]<=0) | (df['Height (Inches)'][i]>100) ):
        df['Height (Inches)'][i]=np.nan
# print(df['Height (Inches)'].head(20))

# weight
for i in range(0,len(weight)):
    if( (df['Weight (Pounds)'][i]<=0) | (df['Weight (Pounds)'][i]>200) ):
        df['Weight (Pounds)'][i]=np.nan
#print(df['Weight (Pounds)'].tail(25))

# bmi
for i in range(0,len(bmi)):
    if( (df['BMI'][i]<=0) | (df['BMI'][i]>4) ):
        df['BMI'][i]=np.nan
#print(df['BMI'].head(15))
#print('='*50)

# Print # of rows with NaN // 각 행에 NaN이 몇개 있는지 출력
print("# of rows with NaN")
print( df.isnull().sum(axis=1) )
print('-'*30)

# #ofNaN for each column // 각 열에 몇개 있는지 출력
print("# of NaN for each column")
print( df['Height (Inches)'].isna().sum() )
print('-'*30)
print( df['Weight (Pounds)'].isna().sum() )
print('-'*30)
print( df['BMI'].isna().sum() )
print('='*50)

# Extract all rows without NaN
print("Extracting all rows without NaN")
df_without_row_NaN = df.dropna(how="any")
print(df_without_row_NaN)
print('='*50)

# Fill NaN with mean, median, or using ffill / bfill methods
print("Fill NaN with mean, median, or ffill/bfill methods")
df_fillNa = df.copy()
df_fillNa["BMI"].fillna( df_fillNa["BMI"].mean(), inplace=True )
df_fillNa["Height (Inches)"].fillna( df_fillNa["Height (Inches)"].median(), inplace=True )
df_fillNa["Weight (Pounds)"].fillna( df_fillNa["Weight (Pounds)"].median(), inplace=True )

df_fill_NaN = df_fillNa.dropna(how="any")
print(df_fill_NaN)
print('='*50)

#=========================================================================================

# Clean all dirty records using Linear regression for (height, weight) values
df['Height (Inches)'][df['Height (Inches)']>100] = np.nan
df['Height (Inches)'][df['Height (Inches)']<=0] = np.nan
df['Weight (Pounds)'][(df['Weight (Pounds)']>200)] = np.nan
df['Weight (Pounds)'][(df['Weight (Pounds)']<=0)] = np.nan

#======================================
# Linear Regression
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
lr_df = df[ ['Sex', 'Height (Inches)', 'Weight (Pounds)']]

# Computing replacemnet values using E - groups divided by gender
# female
lr_df_fm = lr_df.loc[lr_df["Sex"] == "Female"]
lr_df_fm_NaN = lr_df_fm.dropna()

lr_ht_fm = lr_df_fm_NaN["Height (Inches)"].to_numpy()
lr_wt_fm = lr_df_fm_NaN["Weight (Pounds)"].to_numpy()
lr_fm = linear_model.LinearRegression()
lr_fm.fit(lr_ht_fm[:, np.newaxis], lr_wt_fm)

lr_test_ht_fm = lr_df_fm[lr_df_fm["Weight (Pounds)"].isna()]
lr_test_ht_fm = lr_test_ht_fm['Height (Inches)'].to_numpy()
lr_test_wt_fm = lr_fm.predict(lr_test_ht_fm[:, np.newaxis])

x_sex_fm = np.array([lr_ht_fm.min()-1, lr_ht_fm.max()+1])
y_sex_fm = lr_fm.predict(x_sex_fm[:, np.newaxis])

plt.title('Female')
plt.scatter(lr_ht_fm, lr_wt_fm)
plt.scatter(lr_test_ht_fm, lr_test_wt_fm, color='red')
plt.plot(x_sex_fm, y_sex_fm, color='blue')
plt.show()

# male
lr_df_m = lr_df.loc[lr_df["Sex"] == "Male"]
lr_df_m_NaN = lr_df_m.dropna()

lr_ht_m = lr_df_m_NaN["Height (Inches)"].to_numpy()
lr_wt_m = lr_df_m_NaN["Weight (Pounds)"].to_numpy()
lr_m = linear_model.LinearRegression()
lr_m.fit(lr_ht_m[:, np.newaxis], lr_wt_m)

lr_test_ht_m = lr_df_m[lr_df_m["Weight (Pounds)"].isna()]
lr_test_ht_m = lr_test_ht_m['Height (Inches)'].to_numpy()
lr_test_wt_m = lr_m.predict(lr_test_ht_m[:, np.newaxis])

x_sex_m = np.array([lr_ht_m.min()-1, lr_ht_m.max()+1])
y_sex_m = lr_m.predict(x_sex_m[:, np.newaxis])

plt.title('Male')
plt.scatter(lr_ht_m, lr_wt_m)
plt.scatter(lr_test_ht_m, lr_test_wt_m, color='red')
plt.plot(x_sex_m, y_sex_m, color='blue')
plt.show()

# Computing replacemnet values using E - groups divided by BMI
lr_df_BMI = df[["Height (Inches)", "Weight (Pounds)", "BMI"]]
lr_df_BMI_2 = lr_df_BMI.loc[lr_df_BMI["BMI"] <= 2]
lr_df_BMI_3 = lr_df_BMI.loc[lr_df_BMI["BMI"] >= 3]

lr_df_BMI_2_no_NaN = lr_df_BMI_2.dropna()
lr_df_BMI_3_no_NaN = lr_df_BMI_3.dropna()

# bmi <= 2
lr_ht_BMI_2 = lr_df_BMI_2_no_NaN["Height (Inches)"].to_numpy()
lr_wt_BMI_2 = lr_df_BMI_2_no_NaN["Weight (Pounds)"].to_numpy()
lr_BMI_2 = linear_model.LinearRegression()
lr_BMI_2.fit(lr_ht_BMI_2[:, np.newaxis], lr_wt_BMI_2)

lr_test_ht_BMI_2 = lr_df_BMI_2[lr_df_BMI_2["Weight (Pounds)"].isna()]
lr_test_ht_BMI_2 = lr_test_ht_BMI_2['Height (Inches)'].to_numpy()
lr_test_wt_BMI_2 = lr_BMI_2.predict(lr_test_ht_BMI_2[:, np.newaxis])

x_BMI_2 = np.array([lr_ht_BMI_2.min()-1, lr_ht_BMI_2.max()+1])
y_BMI_2 = lr_BMI_2.predict(x_BMI_2[:, np.newaxis])

plt.title('BMI <= 2')
plt.scatter(lr_ht_BMI_2, lr_wt_BMI_2)
plt.scatter(lr_test_ht_BMI_2, lr_test_wt_BMI_2, color='red')
plt.plot(x_BMI_2, y_BMI_2, color='blue')
plt.show()

# bmi >= 3
lr_ht_BMI_3 = lr_df_BMI_3_no_NaN["Height (Inches)"].to_numpy()
lr_wt_BMI_3 = lr_df_BMI_3_no_NaN["Weight (Pounds)"].to_numpy()
lr_BMI_3 = linear_model.LinearRegression()
lr_BMI_3.fit(lr_ht_BMI_3[:, np.newaxis], lr_wt_BMI_3)

lr_test_ht_BMI_3 = lr_df_BMI_3[lr_df_BMI_3["Weight (Pounds)"].isna()]
lr_test_ht_BMI_3 = lr_test_ht_BMI_3['Height (Inches)'].to_numpy()
lr_test_wt_BMI_3 = lr_BMI_3.predict(lr_test_ht_BMI_3[:, np.newaxis])

x_BMI_3 = np.array([lr_ht_BMI_3.min()-1, lr_ht_BMI_3.max()+1])
y_BMI_3 = lr_BMI_3.predict(x_BMI_3[:, np.newaxis])

plt.title('BMI >= 3')
plt.scatter(lr_ht_BMI_3, lr_wt_BMI_3)
plt.scatter(lr_test_ht_BMI_3, lr_test_wt_BMI_3, color='red')
plt.plot(x_BMI_3, y_BMI_3, color='blue')
plt.show()