from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
print(os.path.exists("Titanic-Dataset.csv"))


df = pd.read_csv(
    "C:\\Users\\khann\\Desktop\\New folder\\DAY 1\\Titanic-Dataset.csv")

df.head()
df.info()
df.describe()
df.isnull().sum()

df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])


sns.boxplot(x=df['Age'])
plt.title('Boxplot of Age')
plt.show()

sns.boxplot(x=df['Fare'])
plt.title('Boxplot of Fare')
plt.show()

Q1 = df['Fare'].quantile(0.25)
q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Fare'] >= Q1 - 1.5 * IQR) & (df['Fare'] <= Q3 + 1.5 * IQR)]


df = remove_outliers(df, 'Age')
df = remove_outliers(df, 'Fare')

# Final check
print("\nCleaned dataset shape:", df.shape)
print(df.head())
