import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LinearRegression
pd.set_option('display.max_rows', 160)


df = pd.read_csv("C://Users//white//Desktop//bmi_data_lab2.csv")
#dataexploration

print(df)
print(df.columns)

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

#scaler
for scaler_name in scalers.keys():
    plt.figure()
    sns.kdeplot(height_values.flatten(), label='Original values')
    sns.kdeplot(scaled_height_values[scaler_name].flatten(), label='Scaled values')
    plt.title(f'Scaling results for height ({scaler_name})')
    plt.legend()
    plt.show()

#scaler
for scaler_name in scalers.keys():
    plt.figure()
    sns.kdeplot(weight_values.flatten(), label='Original values')
    sns.kdeplot(scaled_weight_values[scaler_name].flatten(), label='Scaled values')
    plt.title(f'Scaling results for weight ({scaler_name})')
    plt.legend()
    plt.show()




#MISSING value manipulation

df.loc[df['Height (Inches)'] >= 80, 'Height (Inches)'] = np.nan
df.loc[df['Height (Inches)'] <= 10, 'Height (Inches)'] = np.nan
df.loc[df['Weight (Pounds)'] >= 200, 'Weight (Pounds)'] = np.nan
df.loc[df['Weight (Pounds)'] <= 10, 'Weight (Pounds)'] = np.nan


#printwithoutNan
print(df.dropna(subset=['Height (Inches)']))
nan_Height = df[df['Height (Inches)'].isna()]
print(nan_Height)

print(df.dropna(subset=['Weight (Pounds)']))
nan_Weight = df[df['Weight (Pounds)'].isna()]
print(nan_Weight)




#mean
mean_valueH = df['Height (Inches)'].mean()
df['Height (Inches)'].fillna(mean_valueH, inplace=True)
single_columnH = df['Height (Inches)']
print(single_columnH)


#mean
mean_valueW = df['Weight (Pounds)'].mean()
df['Weight (Pounds)'].fillna(mean_valueW, inplace=True)
single_columnW = df['Weight (Pounds)']
print(single_columnW)





