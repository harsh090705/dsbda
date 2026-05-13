import pandas as pd
import numpy as py
df=pd.read_csv("titanic.csv / path of titanic.csv after downloading from kaggle")
print(df.head())
print(df.isnull().sum())
print(df.describe())
print(df.info())
print(df.shape)
print(df.dtypes)
df['Age']=df['Age'].astype(float)
df['Survived']=df['Survived'].astype(int)
df['Sex']=df['Sex'].map({'male':0 , 'female':1})
df=pd.get_dummies(df,columns['Embarked'])
print(df.head())
