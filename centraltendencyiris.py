import pandas as pd 
import numpy as py
df = pd.read_csv("iris.csv")

print(df.head())
print(df.info())
print(df['species'].unique())
setosa = df[df['species'] == 'Iris-setosa']

print(setosa.head())
print(setosa.describe())
versicolor = df[df['species'] == 'Iris-versicolor']

print(versicolor.head())
print(versicolor.describe())
virginica = df[df['species'] == 'Iris-virginica']

print(virginica.head())
print(virginica.describe())
print(df.groupby('species').mean())
print(df.groupby('species').std())
print(df.groupby('species').quantile([0.25,0.50,0.75]))
print(df)
