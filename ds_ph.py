import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn import linear_model
import seaborn as sns

data = pd.read_excel("C://Users//white//Desktop//bmi_data_phw1.xlsx", sheet_name="dataset")

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



#linear regression
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

reg_data = data[["Height (Inches)", "Weight (Pounds)"]]
reg_data_rm_nan = reg_data.dropna()

reg_height = reg_data_rm_nan["Height (Inches)"].to_numpy()
reg_weight = reg_data_rm_nan["Weight (Pounds)"].to_numpy()
reg = linear_model.LinearRegression()
reg.fit(reg_height[:, np.newaxis], reg_weight)

reg_data_test_height = reg_height

reg_data_test_weight = reg.predict(reg_data_test_height[:, np.newaxis])

px = np.array([reg_height.min()-1, reg_height.max()+1])
py = reg.predict(px[:, np.newaxis])

plt.scatter(reg_height, reg_weight)
plt.scatter(reg_data_test_height, reg_data_test_weight, color='r')
plt.plot(px, py, color='b')
plt.axis([62, 75, 90, 170])
plt.show()

# compute e = w-w`

e = reg_weight - reg_data_test_weight

e_normalize = stand.fit_transform(e[:, np.newaxis])

plt.hist(e_normalize, bins=np.linspace(-2.5, 2.5, 11), rwidth=0.8)
plt.xticks([-2, -1, 0, 1, 2])
plt.xlabel("Ze")
plt.ylabel("frequency")
plt.show()

# compute a

data_BMI_0 = sum(data["BMI"] == 0)
data_BMI_4 = sum(data["BMI"] == 4)

e_normalize = np.sort(e_normalize, axis=0)
print(e_normalize[data_BMI_0])
print(e_normalize[len(e_normalize) - data_BMI_4 - 1])

a = (np.abs(e_normalize[data_BMI_0]) + np.abs(e_normalize[len(e_normalize) - data_BMI_4 - 1]))/2

print("a = {}".format(a))

# divide gender
reg_data_gender = data[["Sex", "Height (Inches)", "Weight (Pounds)"]]
reg_data_female = reg_data_gender.loc[reg_data_gender["Sex"] == "Female"]
reg_data_male = reg_data_gender.loc[reg_data_gender["Sex"] == "Male"]

reg_height_female = reg_data_female["Height (Inches)"].to_numpy()
reg_weight_female = reg_data_female["Weight (Pounds)"].to_numpy()
reg_height_male = reg_data_male["Height (Inches)"].to_numpy()
reg_weight_male = reg_data_male["Weight (Pounds)"].to_numpy()

reg_female = linear_model.LinearRegression()
reg_male = linear_model.LinearRegression()
reg_female.fit(reg_height_female[:, np.newaxis], reg_weight_female)
reg_male.fit(reg_height_male[:, np.newaxis], reg_weight_male)

reg_data_test_height_female = reg_height_female
reg_data_test_height_male = reg_height_male

reg_data_test_weight_female = reg_female.predict(reg_data_test_height_female[:, np.newaxis])
reg_data_test_weight_male = reg_male.predict(reg_data_test_height_male[:, np.newaxis])

px_female = np.array([reg_height_female.min()-1, reg_height_female.max()+1])
py_female = reg_female.predict(px_female[:, np.newaxis])

px_male = np.array([reg_height_male.min()-1, reg_height_male.max()+1])
py_male = reg_male.predict(px_male[:, np.newaxis])

plt.subplot(1, 2, 1)
plt.scatter(reg_height_female, reg_weight_female)
plt.scatter(reg_data_test_height_female, reg_data_test_weight_female, color='r')
plt.plot(px_female, py_female, color='black')

plt.subplot(1, 2, 2)
plt.scatter(reg_height_male, reg_weight_male)
plt.scatter(reg_data_test_height_male, reg_data_test_weight_male, color='r')
plt.plot(px_male, py_male, color='black')
plt.show()

# histogram

e_female = reg_weight_female - reg_data_test_weight_female
e_male = reg_weight_male - reg_data_test_weight_male

e_female_normalize = stand.fit_transform(e_female[:, np.newaxis])
e_male_normalize = stand.fit_transform(e_male[:, np.newaxis])

plt.subplot(2, 1, 1)
plt.hist(e_female_normalize, bins=np.linspace(-2.5, 2.5, 11), rwidth=0.8)
plt.xticks([-2, -1, 0, 1, 2])
plt.title("female_e")
plt.xlabel("Ze")
plt.ylabel("frequency")

plt.subplot(2, 1, 2)
plt.hist(e_male_normalize, bins=np.linspace(-2.5, 2.5, 11), rwidth=0.8)
plt.xticks([-2, -1, 0, 1, 2])
plt.title("male_e")
plt.xlabel("Ze")
plt.ylabel("frequency")
plt.show()

# compute a

data_BMI_0_female = sum((data["BMI"] == 0) & (data["Sex"] == "Female"))
data_BMI_4_female = sum((data["BMI"] == 4) & (data["Sex"] == "Female"))

e_female_normalize = np.sort(e_female_normalize, axis=0)
print(e_female_normalize[data_BMI_0_female])
print(e_female_normalize[len(e_female_normalize) - data_BMI_4_female - 1])

a_female = (np.abs(e_female_normalize[data_BMI_0_female]) + np.abs(e_female_normalize[len(e_female_normalize) - data_BMI_4_female- 1]))/2

print("female a = {}".format(a_female))

data_BMI_0_male = sum((data["BMI"] == 0) & (data["Sex"] == "male"))
data_BMI_4_male = sum((data["BMI"] == 4) & (data["Sex"] == "male"))

e_male_normalize = np.sort(e_male_normalize, axis=0)

print(e_male_normalize[data_BMI_0_male])
print(e_male_normalize[len(e_male_normalize) - data_BMI_4_male - 1])

a_male = (np.abs(e_male_normalize[data_BMI_0_male]) + np.abs(e_male_normalize[len(e_male_normalize) - data_BMI_4_male- 1]))/2

print("male a = {}".format(a_male))
