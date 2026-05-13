import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
df = pd.read_csv("Social_Network_Ads.csv")

print(df.head())
print(df.info())
X = df[['Age', 'EstimatedSalary']]

y = df['Purchased']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)
sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(y_pred)
cm = confusion_matrix(y_test, y_pred)

print(cm)
TN = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TP = cm[1][1]

print("True Negative:", TN)
print("False Positive:", FP)
print("False Negative:", FN)
print("True Positive:", TP)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
error_rate = 1 - accuracy

print("Error Rate:", error_rate)
precision = TP / (TP + FP)

print("Precision:", precision)
recall = TP / (TP + FN)

print("Recall:", recall)
print(df.head())
