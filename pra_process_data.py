import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

data_clean = pd.read_csv('data_export_processed.csv')
data_clean = data_clean[data_clean['Label'] != 'Neutral']

def pra_proses(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)()(?:-)?(?:\)|\(|D|P)',
                           text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text


X_train, X_test, y_train, y_test = train_test_split(data_clean['content'], data_clean['Label'], test_size = 0.20, random_state = 0)

tfidf_vectorizer = TfidfVectorizer()
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)            

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)     


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
vectorizer.fit(X_train)       
X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)                            

from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()
nb.fit(tfidf_train, y_train)

X_train.toarray()
y_pred = nb.predict(tfidf_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)

print("MultinomialNB Accuracy:", accuracy_score(y_test,predicted))
# print("MultinomialNB Precision:", precision_score(y_test,predicted, average="micro", zero_division=0))
# print("MultinomialNB Recall:", recall_score(y_test,predicted, average="micro", zero_division=0))
# print("MultinomialNB f1_score:", f1_score(y_test,predicted, average="micro", zero_division=0))
print("MultinomialNB Precision:", precision_score(y_test,predicted, average="binary", pos_label="Negative"))
print("MultinomialNB Recall:", recall_score(y_test,predicted, average="binary", pos_label="Negative"))
print("MultinomialNB f1_score:", f1_score(y_test,predicted, average="binary", pos_label="Negative"))

print(f'confusion_matrix:\n {confusion_matrix(y_test, predicted)}')
print('=====================================================\n')
print(classification_report(y_test, predicted, zero_division=0))
print('=====================================================\n')

# Load dataset
# data_clean = pd.read_csv('hasil_TextPreProcessing_shopee.csv')