import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn import linear_model
import seaborn as sns

data = pd.read_csv("C://Users//white//Desktop//bmi_data_lab2.csv")

# feature names & data types
print(data.columns)

# height & weight histograms
plt.subplot(2, 1, 1)
plt.title("height")

height_data = data["Height (Inches)"].to_numpy()
height_bins = np.linspace(62, 75, 11)
plt.hist(height_data, bins=height_bins, rwidth=0.8)
plt.xticks(height_bins)

plt.subplot(2, 1, 2)
plt.title("weight")

weight_data = data["Weight (Pounds)"].to_numpy()
weight_bins = np.linspace(90, 170, 11)
plt.hist(weight_data, bins=weight_bins, rwidth=0.8)
plt.xticks(weight_bins)
plt.show()

# Scaler

minmax = MinMaxScaler()
stand = StandardScaler()
robust = RobustScaler()

data_minmax = minmax.fit_transform(data.iloc[:, 1:3])
df_minmax = pd.DataFrame(data_minmax, columns=["Height", "Weight"])
print(df_minmax.head())

fig_minmax, (ax1_minmax, ax2_minmax) = plt.subplots(ncols=2, figsize=(6, 5))

ax1_minmax.set_title('before scaling')
sns.kdeplot(data["Height (Inches)"], ax=ax1_minmax)
sns.kdeplot(data["Weight (Pounds)"], ax=ax1_minmax)

ax2_minmax.set_title('after scaling minmax')
sns.kdeplot(df_minmax["Height"], ax=ax2_minmax)
sns.kdeplot(df_minmax["Weight"], ax=ax2_minmax)

plt.show()

data_stand = stand.fit_transform(data.iloc[:, 1:3])
df_stand = pd.DataFrame(data_stand, columns=["Height", "Weight"])
print(df_stand.head())

fig_stand, (ax1_stand, ax2_stand) = plt.subplots(ncols=2, figsize=(6, 5))

ax1_stand.set_title('before scaling')
sns.kdeplot(data["Height (Inches)"], ax=ax1_stand)
sns.kdeplot(data["Weight (Pounds)"], ax=ax1_stand)

ax2_stand.set_title('after scaling stand')
sns.kdeplot(df_stand["Height"], ax=ax2_stand)
sns.kdeplot(df_stand["Weight"], ax=ax2_stand)

plt.show()

data_robust = robust.fit_transform(data.iloc[:, 1:3])
df_robust = pd.DataFrame(data_robust, columns=["Height", "Weight"])
print(df_robust.head())

fig_robust, (ax1_robust, ax2_robust) = plt.subplots(ncols=2, figsize=(6, 5))

ax1_robust.set_title('before scaling')
sns.kdeplot(data["Height (Inches)"], ax=ax1_robust)
sns.kdeplot(data["Weight (Pounds)"], ax=ax1_robust)

ax2_robust.set_title('after scaling robust')
sns.kdeplot(df_robust["Height"], ax=ax2_robust)
sns.kdeplot(df_robust["Weight"], ax=ax2_robust)

plt.show()


# NaN data preprocessing

data.loc[(data["Sex"] != "Female") & (data["Sex"] != "Male"), "Sex"] = np.nan
data.loc[(data["Age"] < 0) | (data["Age"] > 100), "Age"] = np.nan
data.loc[(data["Height (Inches)"] > 75) | (data["Height (Inches)"] < 62), "Height (Inches)"] = np.nan
data.loc[(data["Weight (Pounds)"] > 170) | (data["Weight (Pounds)"] < 90), "Weight (Pounds)"] = np.nan
data.loc[~data["BMI"].between(0, 5), "BMI"] = np.nan

# drop NaN row
data_without_row_nan = data.dropna(how="any")

print(data_without_row_nan)

#data[["Height (Inches)", "Weight (Pounds)"]] = minmax.fit_transform(data[["Height (Inches)", "Weight (Pounds)"]])

# fill NaN data using fillna

data_fillna = data.copy()
data_fillna["Sex"].fillna(method="ffill", inplace=True)
data_fillna["Age"].fillna(method="bfill", inplace=True)
data_fillna["Height (Inches)"].fillna(data_fillna["Height (Inches)"].mean(), inplace=True)
data_fillna["Weight (Pounds)"].fillna(data_fillna["Weight (Pounds)"].mean(), inplace=True)
data_fillna["BMI"].fillna(data_fillna["BMI"].median(), inplace=True)

data_fill_nan = data_fillna.dropna(how="any")

print(data_fill_nan)

#linear regression
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

reg_data = data[["Height (Inches)", "Weight (Pounds)"]]
reg_data_rm_nan = reg_data.dropna()

reg_height = reg_data_rm_nan["Height (Inches)"].to_numpy()
reg_weight = reg_data_rm_nan["Weight (Pounds)"].to_numpy()
reg = linear_model.LinearRegression()
reg.fit(reg_height[:, np.newaxis], reg_weight)

reg_data_test_height = reg_data[reg_data["Weight (Pounds)"].isna()]

reg_data_test_height = reg_data_test_height["Height (Inches)"].to_numpy()

reg_data_test_weight = reg.predict(reg_data_test_height[:, np.newaxis])

px = np.array([reg_height.min()-1, reg_height.max()+1])
py = reg.predict(px[:, np.newaxis])

plt.scatter(reg_height, reg_weight)
plt.scatter(reg_data_test_height, reg_data_test_weight, color='r')
plt.plot(px, py, color='b')
plt.axis([62, 75, 90, 170])
#plt.axis([0, 1, 0, 1])
plt.show()

# grouping gender

reg_data_gender = data[["Sex", "Height (Inches)", "Weight (Pounds)"]]
reg_data_gender = reg_data_gender.loc[reg_data_gender["Sex"] == "Female"]

reg_data_gender_rm_nan = reg_data_gender.dropna()

reg_height_female = reg_data_gender_rm_nan["Height (Inches)"].to_numpy()
reg_weight_female = reg_data_gender_rm_nan["Weight (Pounds)"].to_numpy()
reg = linear_model.LinearRegression()
reg.fit(reg_height_female[:, np.newaxis], reg_weight_female)

reg_data_test_height_female = reg_data_gender[reg_data_gender["Weight (Pounds)"].isna()]

reg_data_test_height_female = reg_data_test_height_female["Height (Inches)"].to_numpy()

reg_data_test_weight_female = reg.predict(reg_data_test_height_female[:, np.newaxis])

px_gender = np.array([reg_height_female.min()-1, reg_height_female.max()+1])
py_gender = reg.predict(px_gender[:, np.newaxis])

plt.scatter(reg_height_female, reg_weight_female)
plt.scatter(reg_data_test_height_female, reg_data_test_weight_female, color='r')
plt.plot(px_gender, py_gender, color='b')
plt.axis([62, 75, 90, 170])
#plt.axis([0, 1, 0, 1])
plt.show()
# grouping BMI

reg_data_BMI = data[["Height (Inches)", "Weight (Pounds)", "BMI"]]
reg_data_BMI = reg_data_BMI.loc[reg_data_BMI["BMI"] > 2]

print(reg_data_BMI)
reg_data_BMI_rm_nan = reg_data_BMI.dropna()

reg_height_BMI = reg_data_BMI_rm_nan["Height (Inches)"].to_numpy()
reg_weight_BMI = reg_data_BMI_rm_nan["Weight (Pounds)"].to_numpy()
reg = linear_model.LinearRegression()
reg.fit(reg_height_BMI[:, np.newaxis], reg_weight_BMI)

reg_data_test_height_BMI = reg_data_BMI[reg_data_BMI["Weight (Pounds)"].isna()]
print(reg_data_test_height_BMI)
reg_data_test_height_BMI = reg_data_test_height_BMI["Height (Inches)"].to_numpy()

reg_data_test_weight_BMI = reg.predict(reg_data_test_height_BMI[:, np.newaxis])

px_BMI = np.array([reg_height_BMI.min()-1, reg_height_BMI.max()+1])
py_BMI = reg.predict(px_BMI[:, np.newaxis])

plt.scatter(reg_height_BMI, reg_weight_BMI)
plt.scatter(reg_data_test_height_BMI, reg_data_test_weight_BMI, color='r')
plt.plot(px_BMI, py_BMI, color='b')
plt.axis([62, 75, 90, 170])
#plt.axis([0, 1, 0, 1])
plt.show()
