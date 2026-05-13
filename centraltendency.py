import pandas as pd
import numpy as np
data = {
    'Name': ['Amit','Rahul','Sneha','Pooja','Rohan','Neha','Karan','Anjali','Vikas','Priya'],
    'Age_Group': ['18-25','18-25','26-35','26-35','18-25','36-45','36-45','26-35','18-25','36-45'],
    'Income': [25000,30000,40000,42000,28000,50000,52000,41000,27000,48000]
}

df = pd.DataFrame(data)

print(df)
print(df.head())
print(df.dtypes)
summary = df.groupby('Age_Group')['Income'].agg(['mean','median','min','max','std'])

print(summary)
income_list = df.groupby('Age_Group')['Income'].apply(list)

print(income_list)
print(df.info())
print(df)


