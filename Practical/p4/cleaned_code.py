import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("city_day.csv")
print(df)

df.shape
df.info()

df.isnull().sum()

missing_data = df.isnull()
missing_data.head()

for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print("")

pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
for col in pollutants:
    df[col].fillna(df[col].mean())

df['Date'] = pd.to_datetime(df['Date'])

df[pollutants].describe()

df.groupby('City')[pollutants].mean()

plt.figure(figsize=(10,5))
plt.plot(df['Date'], df['PM2.5'])
plt.title("PM2.5 Trend Over Time")
plt.xlabel("Date")
plt.ylabel("PM2.5")
plt.show()

df['PM2.5'].hist(bins=30)
plt.title("Distribution of PM2.5")
plt.show()

sns.boxplot(data=df[pollutants])
plt.show()

corr = df[pollutants].corr()
corr

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap of Pollutants")
plt.show()

plt.figure(figsize=(10,5))
sns.boxplot(x='City', y='PM2.5', data=df)
plt.xticks(rotation=90)
plt.show()
