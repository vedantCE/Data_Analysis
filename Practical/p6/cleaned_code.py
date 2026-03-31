import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

df = pd.read_csv("HR_capstone_dataset.csv")

df.shape

df.isnull().sum()

df.info()

stratified_random_sample = df.groupby("Department", group_keys=False).apply(
    lambda x: x.sample(frac=0.1, random_state=42)
)

print("Before sampling shape:", df.shape)
print("After sampling shape:", stratified_random_sample.shape)
print(stratified_random_sample["Department"].value_counts())

plt.figure(figsize=(8,5))
sns.boxplot(x=stratified_random_sample['satisfaction_level'])
plt.title('Outliers in Satisfaction Score')
plt.show()

z_scores = np.abs(stats.zscore(stratified_random_sample['satisfaction_level']))

outliers = stratified_random_sample[z_scores > 3]
print("Total outliers found:", len(outliers))
print(outliers)

print(df.columns)

print(df['salary'].unique())
print(df['Department'].unique())

low = stratified_random_sample[
        stratified_random_sample['salary']=='low']['satisfaction_level']

medium = stratified_random_sample[
          stratified_random_sample['salary']=='medium']['satisfaction_level']

high = stratified_random_sample[
        stratified_random_sample['salary']=='high']['satisfaction_level']

f_stat, p_value = stats.f_oneway(low, medium, high)

print("F-statistic:", f_stat)
print("P-value:", p_value)

if p_value < 0.05:
    print("Result: Reject H0")
    print("Salary DOES affect satisfaction level")
else:
    print("Result: Accept H0")
    print("Salary does NOT affect satisfaction level")

import statsmodels.api as sm
from statsmodels.formula.api import ols

model = ols('satisfaction_level ~ C(salary) + C(Department) + C(salary):C(Department)', 
             data=stratified_random_sample).fit()

anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)
