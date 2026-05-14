
===============================
DATA WRANGLING
===============================

import pandas as pd
import numpy as np

df = pd.read_csv("data.csv")

print(df.head())

print(df.isnull().sum())

print(df.describe())

print(df.info())

print(df.shape)

print(df.dtypes)

df['Age'] = df['Age'].astype(float)
df['Survived'] = df['Survived'].astype(int)

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

df = pd.get_dummies(df, columns=['Embarked'])

print(df.head())
   


===============================
DATA WRANGLING II
===============================

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

df['Math_Marks'].fillna(df['Math_Marks'].mean(), inplace=True)
df['Science_Marks'].fillna(df['Science_Marks'].mean(), inplace=True)
df['English_Marks'].fillna(df['English_Marks'].mean(), inplace=True)
df['Attendance'].fillna(df['Attendance'].mean(), inplace=True)

print(df.isnull().sum())

print(df.describe())

Q1 = df[['Math_Marks','Science_Marks','English_Marks','Attendance']].quantile(0.25)
Q3 = df[['Math_Marks','Science_Marks','English_Marks','Attendance']].quantile(0.75)

outliers = ((df[['Math_Marks','Science_Marks','English_Marks','Attendance']] < lower_limit) | 
            (df[['Math_Marks','Science_Marks','English_Marks','Attendance']] > upper_limit))

print(outliers)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

for column in ['Math_Marks','Science_Marks','English_Marks','Attendance']:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    median = df[column].median()

    df[column] = np.where(df[column] > upper_limit, median, df[column])
    df[column] = np.where(df[column] < lower_limit, median, df[column])

df['Attendance_Log'] = np.log(df['Attendance'])

print(df)



=============================== 
DESCRIPTIVE STATISTICS
===============================

import pandas as pd
import numpy as np

data = {
    'Name': ['Amit','Rahul','Sneha','Pooja','Rohan','Neha','Karan','Anjali','Vikas','Priya'],
    'Age_Group': ['18-25','18-25','26-35','26-35','18-25','36-45','36-45','26-35','18-25','36-45'],
    'Income': [25000,30000,40000,42000,28000,50000,52000,41000,27000,48000]
}

df = pd.DataFrame(data)

print(df)

print(df.dtypes)

summary = df.groupby('Age_Group')['Income'].agg(['mean','median','min','max','std'])

print(summary)

income_list = df.groupby('Age_Group')['Income'].apply(list)

print(income_list)



===============================
IRIS DESCRIPTIVE STATISTICS
===============================

import pandas as pd
import numpy as np

df = pd.read_csv("iris.csv")

print(df.head())

print(df.info())

print(df['species'].unique())

setosa = df[df['species'] == 'Iris-setosa']
print(setosa.describe())

versicolor = df[df['species'] == 'Iris-versicolor']
print(versicolor.describe())

virginica = df[df['species'] == 'Iris-virginica']
print(virginica.describe())

print(df.groupby('species').mean())

print(df.groupby('species').std())

print(df.groupby('species').quantile([0.25,0.50,0.75]))
print(df)



===============================
LINEAR REGRESSION
===============================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("HousingData.csv")

print(df.head())

print(df.info())

print(df.isnull().sum())

df = df.fillna(df.mean())

print(df.describe())

X = df.drop('MEDV', axis=1)
y = df['MEDV']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(y_pred)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2)

print("Model Coefficients:")

print(model.coef_)

print("Intercept:")

print(model.intercept_)
print(df.head())



===============================
LOGISTIC REGRESSION
===============================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

df = pd.read_csv("Social_Network_Ads.csv")

print(df.head())

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

cm = confusion_matrix(y_test, y_pred)

print(cm)

TN = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TP = cm[1][1]

accuracy = accuracy_score(y_test, y_pred)

error_rate = 1 - accuracy

precision = TP / (TP + FP)

recall = TP / (TP + FN)

print("Accuracy:", accuracy)
print("Error Rate:", error_rate)
print("Precision:", precision)
print("Recall:", recall)



===============================
NAIVE BAYES
===============================

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



===============================
NLP PREPROCESSING
===============================

import nltk
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

document = """
Natural Language Processing is a branch of Artificial Intelligence.
It helps computers understand human language.
Machine learning techniques are widely used in NLP applications.
"""

tokens = word_tokenize(document)

print(tokens)

pos_tags = pos_tag(tokens)

print(pos_tags)

stop_words = set(stopwords.words('english'))

filtered_words = []

for word in tokens:
    if word.lower() not in stop_words:
        filtered_words.append(word)

print(filtered_words)

stemmer = PorterStemmer()

stemmed_words = []

for word in filtered_words:
    stemmed_words.append(stemmer.stem(word))

print(stemmed_words)

lemmatizer = WordNetLemmatizer()

lemmatized_words = []

for word in filtered_words:
    lemmatized_words.append(lemmatizer.lemmatize(word))

print(lemmatized_words)

documents = [
    "Natural language processing helps computers understand language",
    "Machine learning is used in artificial intelligence",
    "NLP techniques are widely used in text processing"
]
cv = CountVectorizer()

tf_matrix = cv.fit_transform(documents)

tf_df = pd.DataFrame(
    tf_matrix.toarray(),
    columns=cv.get_feature_names_out()
)

print(tf_df)

tfidf = TfidfVectorizer()

tfidf_matrix = tfidf.fit_transform(documents)

tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=tfidf.get_feature_names_out()
)

print(tfidf_df)

print("Tokenized Words:", tokens)

print("Filtered Words:", filtered_words)

print("Stemmed Words:", stemmed_words)

print("Lemmatized Words:", lemmatized_words)


===============================
TITANIC HISTOGRAM
===============================
import seaborn as sns
import matplotlib.pyplot as plt

titanic = sns.load_dataset('titanic')

print(titanic.head())

sns.countplot(x='survived', data=titanic)
plt.show()

sns.countplot(x='sex', hue='survived', data=titanic)
plt.show()

sns.countplot(x='pclass', hue='survived', data=titanic)
plt.show()

sns.heatmap(titanic.isnull(), yticklabels=False, cbar=False)
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

titanic = sns.load_dataset('titanic')

plt.hist(titanic['fare'], bins=30)

plt.xlabel('Fare')

plt.ylabel('Number of Passengers')

plt.title('Distribution of Ticket Fare')

plt.show()



===============================
TITANIC BOXPLOT
===============================
import seaborn as sns
import matplotlib.pyplot as plt

titanic = sns.load_dataset('titanic')

sns.boxplot(x='sex', y='age', hue='survived', data=titanic)

plt.xlabel('Gender')

plt.ylabel('Age')

plt.title('Age Distribution by Gender and Survival')

plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

titanic = sns.load_dataset('titanic')

sns.boxplot(x='sex', y='age', hue='survived', data=titanic)

plt.xlabel('Gender')

plt.ylabel('Age')

plt.title('Age Distribution by Gender and Survival')

plt.show()

2. Observations and Inference
Female passengers had a higher survival rate compared to male passengers.
Most surviving females were between age group 20 to 40.
Male passengers show more spread in age distribution.
Many younger passengers survived compared to older passengers.
The box plot shows presence of some outliers in age data.
Median age of passengers is around 30 years.



===============================
IRIS HISTOGRAM AND BOXPLOT
===============================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

iris = sns.load_dataset('iris')
print(df.head())

print(iris.info())

iris.hist(figsize=(10,8))

plt.show()

plt.figure(figsize=(10,6))

sns.boxplot(data=iris)

plt.show()

print(iris.describe())

sns.boxplot(x=iris['sepal_width'])

plt.show()

Inference
Most features are normally distributed.
Petal length and petal width show clear variation among species.
Sepal width contains some outliers visible in the box plot.
Median values are approximately centered in most distributions.
Petal measurements have less spread compared to sepal measurements.



===============================
SCALA APACHE SPARK
===============================

import org.apache.spark.sql.SparkSession

object SimpleSparkApp {

    def main(args: Array[String]): Unit = {

        val spark = SparkSession.builder()
            .appName("Simple Spark App")
            .master("local[*]")
            .getOrCreate()

        val data = Seq(
            ("Amit", 85),
            ("Rahul", 90),
            ("Sneha", 88)
        )

        import spark.implicits._

        val df = data.toDF("Name", "Marks")

        df.show()

        spark.stop()
    }
}
