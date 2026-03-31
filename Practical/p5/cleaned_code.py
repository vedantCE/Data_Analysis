import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv("melb_data.csv")
print(df)

df.isnull().sum()

df.isnull().sum() / len(df) * 100

df.mode()

df.describe()

print(df['Car'].median())
print(df['BuildingArea'].median())
print(df['YearBuilt'].median())
print(df['CouncilArea'].mode())
print(df['CouncilArea'].mode()[0])

df.fillna(df['Car'].median(), inplace=True)
df.fillna(df['BuildingArea'].median(), inplace=True)
df.fillna(df['YearBuilt'].median(), inplace=True)
df.fillna(df['CouncilArea'].mode()[0], inplace=True)

df.isnull().sum()

df.describe()

plt.figure(figsize=(10,6))
sns.histplot(df['Price'], bins=50, kde=True)
plt.title('Distribution of House Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(x=df['Price'])
plt.title('Boxplot of House Prices')
plt.xlabel('Price')
plt.show()

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

sns.scatterplot(data=df, x='Rooms', y='Price', ax=axes[0,0])
axes[0,0].set_title('Rooms vs Price')

sns.scatterplot(data=df, x='Distance', y='Price', ax=axes[0,1])
axes[0,1].set_title('Distance vs Price')

sns.scatterplot(data=df, x='BuildingArea', y='Price', ax=axes[0,2])
axes[0,2].set_title('BuildingArea vs Price')

sns.scatterplot(data=df, x='Landsize', y='Price', ax=axes[1,0])
axes[1,0].set_title('Landsize vs Price')

sns.scatterplot(data=df, x='Bathroom', y='Price', ax=axes[1,1])
axes[1,1].set_title('Bathroom vs Price')

sns.scatterplot(data=df, x='Car', y='Price', ax=axes[1,2])
axes[1,2].set_title('Car vs Price')

plt.tight_layout()
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

features = ['Rooms', 'Bathroom', 'Car', 'Lattitude', 'Distance']
X = df[features]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training size:", X_train.shape)
print("Testing size:", X_test.shape)

X_simple_train = X_train[['Rooms']]
X_simple_test = X_test[['Rooms']]

simple_model = LinearRegression()
simple_model.fit(X_simple_train, y_train)

y_pred_simple = simple_model.predict(X_simple_test)

r2_simple = r2_score(y_test, y_pred_simple)
print("Simple Regression R²:", r2_simple)

multiple_model = LinearRegression()
multiple_model.fit(X_train, y_train)

y_pred_multiple = multiple_model.predict(X_test)
r2_multiple = r2_score(y_test, y_pred_multiple)
print("Multiple Regression R²:", r2_multiple)
