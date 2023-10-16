import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from textblob import TextBlob

my_df = pd.read_csv('data_export.csv')
print(my_df.shape[0])
print(my_df.shape[1])


def clean_text(df, text_field, new_text_field_name):
    df[new_text_field_name] = df[text_field].str.lower()
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(
        r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))
    df[new_text_field_name] = df[new_text_field_name].apply(
        lambda elem: re.sub(r"\d+", "", elem))
    return my_df


my_df['text_clean'] = my_df['content'].str.lower()
my_df['text_clean']
data_clean = clean_text(my_df, 'content', 'text_clean')


def sentiment(polarity):
    if polarity > 0.0:
        return "Positive"
    elif polarity == 0.0:
        return "Neutral"
    elif polarity < 0.0:
        return "Negative"


data_clean['polarity'] = data_clean['text_clean'].apply(
    lambda x: TextBlob(x).sentiment.polarity)
data_clean['sentiment'] = data_clean['polarity'].apply(sentiment)
print(data_clean.head(10))
sentiment_counts = data_clean['sentiment'].value_counts()
print(sentiment_counts)
sentiment_counts = data_clean['Label'].value_counts()
print(sentiment_counts)

# Menggunakan operator & (dan) untuk menggabungkan dua kondisi
count = len(data_clean[(data_clean['Label'] == 'Neutral')
            & ((data_clean['sentiment'] == 'Negative'))])

# Mencetak jumlah baris yang memenuhi kondisi
print(
    f"Jumlah baris dengan Label 'Neutral' dan sentiment 'Positive' atau 'Negative': {count}")

# nltk.download('stopwords')

stop = stopwords.words('english')
data_clean['text_StopWord'] = data_clean['text_clean'].apply(
    lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
# print(data_clean.head(10))

# nltk.download('punkt')

data_clean['text_tokens'] = data_clean['text_StopWord'].apply(
    lambda x: word_tokenize(x))
print(data_clean.head())


factory = StemmerFactory()
stemmer = factory.create_stemmer()


def stemmed_wrapper(term): return stemmer.stem(term)


term_dict = {}
# hitung=0

for document in data_clean['text_tokens']:
    for term in document:
        if term not in term_dict:
            term_dict[term] = ' '

# print(len(term_dict))
# print("-----------")

for term in term_dict:
    term_dict[term] = stemmed_wrapper(term)
    # hitung+=1
    # print(hitung, ":", term, ":",term_dict[term])

# print(term_dict)
# print("-----------")


def get_stemmed_term(document):
    return [term_dict[term] for term in document]


data_clean['text_steamindo'] = data_clean['text_tokens'].apply(
    lambda x: ' '.join(get_stemmed_term(x)))
print(data_clean.head(10))

data_clean.to_csv('data_export_processed.csv', index=False)
data_clean.to_excel('data_export_processed.xlsx', index=False)
