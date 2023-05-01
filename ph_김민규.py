import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LinearRegression

pd.set_option('display.max_rows', 160)

df = pd.read_excel(r'C:\Users\김민규\Downloads\bmi_data_phw1 (1).xlsx')

bmi_groups = df.groupby('BMI')

#Height
for bmi, bmi_data in bmi_groups:
   
    plt.hist(bmi_data['Height (Inches)'], bins=10)
    plt.title(f'BMI: {bmi}')
    plt.xlabel('Height')
    plt.ylabel('Count')
    
    plt.show()

#Weight
bmi_groups = df.groupby('BMI')


for bmi, bmi_data in bmi_groups:
   
    plt.hist(bmi_data['Weight (Pounds)'], bins=10)
    plt.title(f'BMI: {bmi}')
    plt.xlabel('Weight')
    plt.ylabel('Count')
    
   
    plt.show()




height_values = df['Height (Inches)'].values.reshape(-1, 1)
weight_values = df['Weight (Pounds)'].values.reshape(-1, 1)


height_values1 = df['Height (Inches)'].values
weight_values1 = df['Weight (Pounds)'].values

sns.kdeplot(height_values1)
plt.title('Height KDE plot')
plt.xlabel('Height (inches)')
plt.show()

sns.kdeplot(weight_values1)
plt.title('Weight KDE plot')
plt.xlabel('Weight (pounds)')
plt.show()

scalers = {
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'RobustScaler': RobustScaler()
}
scaled_height_values = {}
scaled_weight_values = {}
for scaler_name, scaler in scalers.items():
    scaled_height_values[scaler_name] = scaler.fit_transform(height_values)
    scaled_weight_values[scaler_name] = scaler.fit_transform(weight_values)

#Scale
for scaler_name in scalers.keys():
    plt.figure()
    sns.kdeplot(height_values.flatten(), label='Original values')
    sns.kdeplot(scaled_height_values[scaler_name].flatten(), label='Scaled values')
    plt.title(f'Scaling results for height ({scaler_name})')
    plt.legend()
    plt.show()

#Scale
for scaler_name in scalers.keys():
    plt.figure()
    sns.kdeplot(weight_values.flatten(), label='Original values')
    sns.kdeplot(scaled_weight_values[scaler_name].flatten(), label='Scaled values')
    plt.title(f'Scaling results for weight ({scaler_name})')
    plt.legend()
    plt.show()




X = df[["Height (Inches)"]]
y = df["Weight (Pounds)"]
reg = LinearRegression().fit(X, y)
E = (reg.coef_[0], reg.intercept_)

# compute e = w - w' for each record in D
df["Weight Residuals"] = df.apply(lambda row: row["Weight (Pounds)"] - (E[0]*row["Height (Inches)"] + E[1]), axis=1)

# compute z-scores of the weight residuals
scaler = StandardScaler()
df["Weight Residuals (z-scores)"] = scaler.fit_transform(df[["Weight Residuals"]])

# plot histogram of z-scores
plt.hist(df["Weight Residuals (z-scores)"], bins=10)
plt.xlabel("z-scores")
plt.ylabel("Frequency")
plt.show()


X = df[["Height (Inches)"]]
y = df["Weight (Pounds)"]
reg = LinearRegression().fit(X, y)
E = (reg.coef_[0], reg.intercept_)

# compute e = w - w' for each record in D
df["Weight Residuals"] = df.apply(lambda row: row["Weight (Pounds)"] - (E[0]*row["Height (Inches)"] + E[1]), axis=1)

# compute z-scores of the weight residuals
scaler = StandardScaler()
df["Weight Residuals (z-scores)"] = scaler.fit_transform(df[["Weight Residuals"]])

# plot histogram of z-scores
plt.hist(df["Weight Residuals (z-scores)"], bins=10)
plt.xlabel("z-scores")
plt.ylabel("Frequency")
plt.show()

# decide threshold alpha
alpha = 4.0

# set BMI values based on z-scores
df["BMI"] = np.where(df["Weight Residuals (z-scores)"] < -alpha, 0, np.where(df["Weight Residuals (z-scores)"] > alpha, 4, df["BMI"]))






df['Gender'] = np.where(df['Index']%2==0, 'F', 'M')

clean_data = df.dropna(subset=['Height (Inches)', 'Weight (Pounds)'])
X = clean_data['Height (Inches)'].values.reshape(-1, 1)
y = clean_data['Weight (Pounds)'].values.reshape(-1, 1)
regressor = LinearRegression()
regressor.fit(X, y)
print(f"Regression equation: y = {regressor.coef_[0][0]:.2f}x + {regressor.intercept_[0]:.2f}")



df['Weight (Pounds) Cleaned'] = df.apply(lambda row: regressor.predict([[row['Height (Inches)']]])[0][0] if pd.notna(row['Weight (Pounds)']) else np.nan, axis=1)
df['e'] = df.apply(lambda row: row['Weight (Pounds)'] - row['Weight (Pounds) Cleaned'] if pd.notna(row['Weight (Pounds)']) else np.nan, axis=1)


e_mean = df['e'].mean()
e_std = df['e'].std()
df['ze'] = df['e'].apply(lambda e: (e - e_mean) / e_std if pd.notna(e) else np.nan)
alpha = 2.0
df['BMI'] = df.apply(lambda row: 0 if row['ze'] < -alpha else (4 if row['ze'] > alpha else row['BMI']), axis=1)
df_female = df[df['Gender'] == 'F']
df_male = df[df['Gender'] == 'M']

# BMI
df_female['BMI Estimated'] = (df_female['Weight (Pounds) Cleaned'] * 0.45) / ((df_female['Height (Inches)'] * 0.025)**2)
df_male['BMI Estimated'] = (df_male['Weight (Pounds) Cleaned'] * 0.45) / ((df_male['Height (Inches)'] * 0.025)**2)