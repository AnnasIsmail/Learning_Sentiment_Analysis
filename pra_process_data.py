import re
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud

# Load the preprocessed data from 'data_export_processed.csv'
data_clean = pd.read_csv('data_export_processed.csv')

# Remove rows with 'Neutral' label
print('Total Data Sebelum Filter Neutral: '+str(data_clean.shape[0]))
data_clean = data_clean[data_clean['sentiment'] != 'Neutral']
print('Total Data Setelah Filter Neutral: '+str(data_clean.shape[0]))

# Data Preprocessing Function (pra_proses)


def preprocess_text(text):
    # Remove HTML tags
    text = re.sub('<[^>]*>', '', text)

    # Find and handle emoticons
    emoticons = re.findall('(?::|;|=)()(?:-)?(?:\)|\(|D|P)', text)

    # Remove non-word characters, convert to lowercase, and join emoticons
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))

    return text


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data_clean['content'], data_clean['sentiment'], test_size=0.20, random_state=0)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

# Count Vectorization
vectorizer = CountVectorizer()
vectorizer.fit(X_train)
X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)

# Train a Multinomial Naive Bayes model on TF-IDF vectors
nb = MultinomialNB()
nb.fit(tfidf_train, y_train)
y_pred = nb.predict(tfidf_test)

# Calculate accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy (TF-IDF): {accuracy}")

# Train another Multinomial Naive Bayes model on Count Vectors
clf = MultinomialNB()
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)

# Evaluate the Count Vectorization model
print("MultinomialNB Accuracy (Count Vectorization):",
      accuracy_score(y_test, predicted))
print("MultinomialNB Precision (Count Vectorization):", precision_score(
    y_test, predicted, average="binary", pos_label="Positive"))
print("MultinomialNB Recall (Count Vectorization):", recall_score(
    y_test, predicted, average="binary", pos_label="Positive"))
print("MultinomialNB F1-score (Count Vectorization):",
      f1_score(y_test, predicted, average="binary", pos_label="Positive"))

# Print the confusion matrix
print(f'Confusion Matrix:\n{confusion_matrix(y_test, predicted)}')
print('=====================================================\n')

# Generate and print a classification report
print(classification_report(y_test, predicted, zero_division=0))
print('=====================================================\n')

# DAILY TREND
# data_clean['at'] = pd.to_datetime(data_clean['at'])
# daily_reviews = data_clean.groupby(data_clean['at'].dt.date).size()
# plt.figure(figsize=(12, 6))
# plt.plot(daily_reviews.index, daily_reviews.values)
# plt.title('Daily Review Trend')
# plt.xlabel('Date')
# plt.ylabel('Count Review')
# plt.grid(True)
# plt.show()

# WORLD CLOUD
# text = ' '.join(data_clean['text_tokens'])

# Membuat objek Word Cloud
# wordcloud = WordCloud(width=800, height=400,
#                       background_color='white').generate(text)

# Menampilkan Word Cloud
# plt.figure(figsize=(10, 5))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.show()

# Menggabungkan semua teks menjadi satu teks panjang
text = ' '.join(data_clean['text_tokens'])

# Membagi teks menjadi kata-kata
words = text.split()

# Menghitung frekuensi setiap kata
word_counts = Counter(words)

# Menampilkan 10 kata teratas yang paling sering muncul
top_words = word_counts.most_common(10)

for word, count in top_words:
    print(f'Kata: {word}, Kemunculan: {count}')

# show diagram
# Hitung jumlah data positif dan negatif
positive_count = len(data_clean[data_clean['sentiment'] == 'Positive'])
negative_count = len(data_clean[data_clean['sentiment'] == 'Negative'])

# Data untuk diagram pie
labels = [f'Positive ({positive_count})', f'Negative ({negative_count})']
sizes = [positive_count, negative_count]
colors = ['green', 'red']

# Atur eksplodasi jika Anda ingin 'highlight' sebagian data
explode = (0.1, 0)

# Buat diagram pie
plt.figure(figsize=(6, 6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')  # Agar diagram terlihat seperti lingkaran
plt.title('Comparison of Positive and Negative Data')
plt.show()

data_clean.to_csv('data_has_analyzed.csv', index=False)
data_clean.to_excel('data_has_analyzed.xlsx', index=False)
