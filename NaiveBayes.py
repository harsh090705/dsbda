import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
df = pd.read_csv("iris.csv")

print(df.head())
print(df.info())
X = df.iloc[:, 0:4]

y = df['species']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(y_pred)
cm = confusion_matrix(y_test, y_pred)

print(cm)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
y_test_binary = np.where(y_test == 'Iris-setosa', 1, 0)

y_pred_binary = np.where(y_pred == 'Iris-setosa', 1, 0)
cm_binary = confusion_matrix(y_test_binary, y_pred_binary)

print(cm_binary)
TN = cm_binary[0][0]
FP = cm_binary[0][1]
FN = cm_binary[1][0]
TP = cm_binary[1][1]

print("True Negative:", TN)
print("False Positive:", FP)
print("False Negative:", FN)
print("True Positive:", TP)
error_rate = 1 - accuracy

print("Error Rate:", error_rate)
precision = TP / (TP + FP)

print("Precision:", precision)
recall = TP / (TP + FN)

print("Recall:", recall)
print(df.head())

