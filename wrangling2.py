import pandas as pd
import numpy as np

data = {
    'Student_ID': [1,2,3,4,5,6,7,8,9,10],
    'Math_Marks': [85,90,np.nan,95,88,300,76,84,91,89],
    'Science_Marks': [78,85,82,80,np.nan,79,300,88,84,81],
    'English_Marks': [74,79,85,90,87,92,76,np.nan,80,78],
    'Attendance': [85,90,88,92,95,20,87,89,np.nan,91]
}

df = pd.DataFrame(data)

print(df)
print(df.isnull().sum())
print(df.info())
print(df.describe())
df['Math_Marks'].fillna(df['Math_Marks'].mean(), inplace=True)
df['Science_Marks'].fillna(df['Science_Marks'].mean(), inplace=True)
df['English_Marks'].fillna(df['English_Marks'].mean(), inplace=True)
df['Attendance'].fillna(df['Attendance'].mean(), inplace=True)
print(df.isnull().sum())
Q1 = df[['Math_Marks','Science_Marks','English_Marks','Attendance']].quantile(0.25)
Q3 = df[['Math_Marks','Science_Marks','English_Marks','Attendance']].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

outliers = ((df[['Math_Marks','Science_Marks','English_Marks','Attendance']] < lower_limit) | 
            (df[['Math_Marks','Science_Marks','English_Marks','Attendance']] > upper_limit))

print(outliers)
for column in ['Math_Marks','Science_Marks','English_Marks','Attendance']:
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    median = df[column].median()

    df[column] = np.where(df[column] > upper_limit, median, df[column])
    df[column] = np.where(df[column] < lower_limit, median, df[column])

print(df)
df['Attendance_Log'] = np.log(df['Attendance'])
print(df)
