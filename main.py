import os.path
import string

import nltk
import pandas as pd
import re
import numpy as np

from keras.src.utils import pad_sequences

import contraction_dict as ctd

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import TweetTokenizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import tensorflow as tf
from tensorflow import keras

from gensim.models import Word2Vec

from spellchecker import SpellChecker

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
tokenizer = TweetTokenizer()
vectorizer = TfidfVectorizer()


def data_remove(data, percentage=0.2):
    grouped_data = data.groupby('class')

    less_data = pd.DataFrame(columns=data.columns)
    for i, group in grouped_data:
        less_data_group = group.sample(frac=percentage)
        less_data = pd.concat([less_data, less_data_group], ignore_index=True)

    less_data = less_data.reset_index(drop=True)
    return less_data


def replace_contractions(text, contraction_dict):
    words = text.split()
    return ' '.join([contraction_dict.get(word, word) for word in words])


def remove_mentions(text):
    # Define a regular expression pattern to match words starting with '@'
    mention_pattern = r'@\w+'
    # Remove mentions from the text
    return re.sub(mention_pattern, '', text)


def to_lower(text):
    return text.lower()


def autocorrect_text(text):
    spell = SpellChecker()

    words = text.split()

    corrected_text = []
    for word in words:
        corrected_word = spell.correction(word)
        if corrected_word != word:
            corrected_text.append(corrected_word)
        else:
            pass

    corrected_sentence = " ".join(f"{corrected_text}")

    return corrected_sentence



def remove_nulls(text):
    if len(text) > 0:
        return text


def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)


def remove_html_tags(text):
    regular_expression = r'<.*?>'
    return re.sub(regular_expression, '', text)


def remove_punctuation(text):
    return re.sub('[%s]' % re.escape(string.punctuation), '', text)


def remove_digits(text):
    return re.sub('W*dw*', '', text)


def remove_stopwords(text):
    return ' '.join([word for word in str(text).split() if word not in stop_words])


def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])


def lemmatize_words(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])


def remove_extra_spaces(text):
    return re.sub(' +', ' ', text)


def tokenize(text):
    return tokenizer.tokenize(text)


path_to_csv = 'archive/tweet_emotions.csv'
raw_data = pd.read_csv(path_to_csv, usecols=['sentiment', 'content'])
raw_data.rename(columns={'sentiment': 'class'}, inplace=True)
class_labels = {cls: label for label, cls in enumerate(raw_data['class'].unique())}
raw_data['labels'] = raw_data['class'].map(class_labels)
raw_data = data_remove(raw_data)

raw_data['content'] = raw_data['content'].apply(replace_contractions, contraction_dict=ctd.contraction_dict)
raw_data['content'] = raw_data['content'].apply(remove_mentions)
raw_data['content'] = raw_data['content'].apply(to_lower)
raw_data['content'] = raw_data['content'].apply(autocorrect_text)
raw_data['content'] = raw_data['content'].apply(remove_nulls)
raw_data.dropna(subset=['content'], inplace=True)
raw_data['content'] = raw_data['content'].apply(remove_urls)
raw_data['content'] = raw_data['content'].apply(remove_html_tags)
raw_data['content'] = raw_data['content'].apply(remove_punctuation)
raw_data['content'] = raw_data['content'].apply(remove_digits)

stop_words = set(stopwords.words('english'))
stop_words.add('subject')
stop_words.add('http')

raw_data['content'] = raw_data['content'].apply(remove_stopwords)
raw_data['content'] = raw_data['content'].apply(stem_words)
raw_data['content'] = raw_data['content'].apply(lemmatize_words)
raw_data['content'] = raw_data['content'].apply(remove_extra_spaces)
#raw_data['content'] = raw_data['content'].apply(tokenize)
raw_data.to_csv('archive/cleaned_data.csv')
content = raw_data['content'].values
y = raw_data['labels'].values

content_train, content_test, y_train, y_test = train_test_split(content, y, test_size=0.25, random_state=42)

'''
#Linear Regression Approach with tfidf
X_train = vectorizer.fit_transform(content_train)
X_test = vectorizer.transform(content_test)

classifier = LogisticRegression(max_iter=10000)
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)

print(f"Accuracy: {score}")
#Accuracy = 33%
#Accuracy = 0.33860949709477056
'''



#DNN approach
X_train_tokenized = [tokenizer.tokenize(sentence) for sentence in content_train]
X_test_tokenized = [tokenizer.tokenize(sentence) for sentence in content_test]

word2vec_model = Word2Vec(sentences=X_train_tokenized, vector_size=100, window=5, min_count=1, workers=4)

X_train_word2vec = [[word2vec_model.wv[word] for word in sentence if word in word2vec_model.wv] for sentence in X_train_tokenized]
X_test_word2vec = [[word2vec_model.wv[word] for word in sentence if word in word2vec_model.wv] for sentence in X_test_tokenized]

max_seq_length = max(max(len(seq) for seq in X_train_word2vec), max(len(seq) for seq in X_test_word2vec))
X_train_word2vec_padded = pad_sequences(X_train_word2vec, maxlen=max_seq_length, padding='post', dtype='float32')
X_test_word2vec_padded = pad_sequences(X_test_word2vec, maxlen=max_seq_length, padding='post', dtype='float32')

model = keras.Sequential([
    keras.layers.LSTM(512, return_sequences=True),
    keras.layers.Dropout(0.2),
    keras.layers.LSTM(256, return_sequences=True),
    keras.layers.Dropout(0.2),
    keras.layers.LSTM(128, return_sequences=True),
    keras.layers.Dropout(0.2),
    keras.layers.LSTM(64, return_sequences=True),
    keras.layers.Dropout(0.2),
    keras.layers.LSTM(32, return_sequences=False),
    keras.layers.Dense(13, activation='softmax')
])

weights_path = 'weights/w.weights.h5'
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

if os.path.exists(weights_path) and 1<0:
    model.load_weights(weights_path)
else:
    y_train = np.array(y_train, dtype=np.int32)
    history = model.fit(X_train_word2vec_padded, y_train, batch_size=32, epochs=100)
    model.save_weights(weights_path)

loss, accuracy = model.evaluate(X_test_word2vec_padded, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')