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

print(document)
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
