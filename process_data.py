import re

import nltk
import pandas as pd

my_df = pd.read_csv('data_export.csv')
print(my_df.shape[0])
print(my_df.shape[1])

def clean_text(df, text_field, new_text_field_name):
    df[new_text_field_name] = df[text_field].str.lower()
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?","",elem))
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"\d+","",elem))  
    return my_df

my_df['text_clean'] = my_df['content'].str.lower()
my_df['text_clean']
data_clean = clean_text(my_df, 'content', 'text_clean')
# print(data_clean.head(10))

# nltk.download('stopwords')
from nltk.corpus import stopwords

stop = stopwords.words('indonesian')
data_clean['text_StopWord'] = data_clean['text_clean'].apply(lambda x:' '.join([word for word in x.split() if word not in (stop)]))
# print(data_clean.head(10))

# nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize

data_clean['text_tokens'] = data_clean['text_StopWord'].apply(lambda x: word_tokenize(x)) 
print(data_clean.head())

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stemmed_wrapper (term): return stemmer.stem(term)

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

data_clean['text_steamindo'] = data_clean['text_tokens'].apply(lambda x:' '.join(get_stemmed_term(x))) 
print(data_clean.head (10))

data_clean.to_csv('data_export_processed.csv', index=False)
data_clean.to_excel('data_export_processed.xlsx', index=False)
