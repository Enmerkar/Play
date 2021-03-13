#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 13:37:26 2021

Test various NLP processes.
Use RNN trained on Tolkien corpus to generate new story.

Could even try blends: e.g. Tolkien + Dan Brown.

@author: JKN08.
"""

# Clean the hobbit
with open('/home/justin/Play/data.txt') as hobbit:
    hobbit_lines = hobbit.readlines()

hobbit_lines_filtered = list(filter(lambda x: x!='\n', hobbit_lines))
hobbit_lines_filtered = hobbit_lines_filtered[59:1937]
chapter_index = [i for i, x in enumerate(hobbit_lines_filtered) if x.startswith('Chapter ')]
chapter_head = chapter_index + [i+1 for i in chapter_index]
keep_lines = set([*range(len(hobbit_lines_filtered))])
keep_lines -= set(chapter_head)
hobbit_clean = [line for i, line in enumerate(hobbit_lines_filtered) if i in keep_lines]
hobbit_string = ' '.join(hobbit_clean)

import spacy
nlp_en = spacy.load('en_core_web_sm')

hobbit_doc = nlp_en(hobbit_string)

doc_en = nlp_en(u'My name is Justin and I am a human who likes eating various foodstuffs.')

for token in doc_en:
    print(f'{token.text:<10}', f'{token.lemma_:<10}', f'{token.pos_:<10}', f'{token.dep_:<10}')
    
nlp_en.pipeline

hobbit_vocab = [token.lemma_.lower() for token in hobbit_doc if not token.is_stop and token.is_alpha]
hobbit_vocab_df = pd.DataFrame(hobbit_vocab)
hobbit_vocab_df.columns = ['Word']
hobbit_vocab_count = hobbit_vocab_df['Word'].value_counts()
hobbit_vocab_count.head()

# TF-IDF and Text Classification

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split

vect = TfidfVectorizer()
dtm = vect.fit_transform(hobbit_clean)
print(vect.get_feature_names())

hobbit_df = pd.DataFrame(hobbit_lines_filtered)
hobbit_df.columns = ['Text']
chapter_df = pd.DataFrame(chapter_index)
chapter_df.columns = ['Line']
chapter_df['Chapter'] = chapter_df.index + 1
hobbit_df['Chapter'] = 0
for i, row in chapter_df.iterrows():
    hobbit_df['Chapter'][row[0]] = row[1]
chapter = 0
for i, row in hobbit_df.iterrows():
    if row[1] > 0: chapter = row[1]
    hobbit_df['Chapter'][i] = chapter

X_train, X_test, y_train, y_test = train_test_split(hobbit_df['Text'],
                                                    hobbit_df['Chapter'],
                                                    test_size=0.3,
                                                    random_state=42)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)

X_train.shape
X_train_counts.shape

from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

text_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC())])

text_clf.fit(X_train, y_train)

predictions = text_clf.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

print(confusion_matrix(y_test, predictions))

print(classification_report(y_test, predictions))

accuracy_score(y_test, predictions)

text_clf.predict(['Bywater','Troll','Mirkwood','Smaug','Zelda','Horse'])

# Semantics

nlp_en_lg = spacy.load('en_core_web_lg')

nlp_en_lg.vocab.vectors.shape

tokens = nlp_en_lg(u'lion cat pet')
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))
        
tokens = nlp_en_lg(u'Gandalf is a powerful wizard from Hobbiton in the Shire.')
for token in tokens:
    print(f'{token.text:<10}', f'{token.has_vector:<10}', f'{token.vector_norm:<20}', f'{token.is_oov:<10}')

from scipy import spatial

cosine_similarity = lambda vec1, vec2: 1 - spatial.distance.cosine(vec1, vec2)

wizard = nlp_en_lg.vocab['wizard'].vector
Tolkien = nlp_en_lg.vocab['Tolkien'].vector
new_word = wizard + Tolkien

matches = []
for word in nlp_en_lg.vocab:
    if word.has_vector:
        matches.append((word, cosine_similarity(new_word, word.vector)))

matches_sorted = sorted(matches, key=lambda item:-item[1])
print([t[0].text for t in matches_sorted[:10]])

# Sentiment Analysis

import nltk
nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

a = 'This is a good movie.'
sid.polarity_scores(a)

b = 'This movie fucking SUCKS!!'
sid.polarity_scores(b)

hobbit_df['Sentiment'] = hobbit_df['Text'].apply(lambda text: sid.polarity_scores(text)['compound'])

hobbit_df.groupby('Chapter').sum('Sentiment')

# Topic Modelling

cv = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
hobbit_dtm = cv.fit_transform(hobbit_clean)

vocab = cv.get_feature_names()

from sklearn.decomposition import LatentDirichletAllocation

LDA = LatentDirichletAllocation(n_components=7, random_state=42)
LDA.fit(hobbit_dtm)

LDA.components_.shape

first_topic = LDA.components_[0]

# list top probabilities for each topic
for i, topic in enumerate(LDA.components_):
    print([vocab[j] for j in topic.argsort()[-10:]])

topic_results = LDA.transform(hobbit_dtm)

topic_results.shape

# Non-Negative Matrix Factorisation

hobbit_1_df = hobbit_df[hobbit_df['Chapter']==1]['Text'][2:]

cv = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
hobbit_1_dtm = cv.fit_transform(hobbit_1_df)
vocab = cv.get_feature_names()

from sklearn.decomposition import NMF

nmf_model = NMF(n_components=5, random_state=42)
nmf_model.fit(hobbit_1_dtm)

# list top coefficients for each topic
for i, topic in enumerate(nmf_model.components_):
    print([vocab[j] for j in topic.argsort()[-10:]])

topic_results = nmf_model.transform(hobbit_1_dtm)
hobbit_1_df['Topic'] = topic_results.argmax(axis=1)

# Text Generation

hobbit_str = ''
for line in hobbit_clean:
    hobbit_str += line.replace('\n',' ')

nlp = spacy.load('en_core_web_lg', disable=['parser','tagger', 'ner'])
nlp.max_length = 300000

def separate_punct(doc):
    return [token.text.lower() for token in nlp(doc) if token.text not in '\\n\\n \\n\\n\\n!\"-#$%&()--.*+,-/:;<=>?@[\\\\]^_`{|}~\\t\\n ']

tokens = separate_punct(hobbit_str[0:nlp.max_length])

train_len = 30 + 1
text_sequences = []

for i in range(train_len, len(tokens)):
    seq = tokens[i-train_len:i]
    text_sequences.append(seq)

import numpy as np
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_sequences)
sequences = np.array(tokenizer.texts_to_sequences(text_sequences))

vocab_size = len(tokenizer.index_word)
tokenizer.word_counts

from keras.utils import to_categorical

X = sequences[:,:-1]
y = to_categorical(sequences[:,-1], num_classes=vocab_size+1)

seq_len = X.shape[1]

from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding

def create_model(vocab_size, seq_len):
    model = Sequential()
    model.add(Embedding(vocab_size, seq_len, input_length=seq_len))
    model.add(LSTM(seq_len*2, return_sequences=True))
    model.add(LSTM(seq_len*3))
    model.add(Dense(seq_len*2, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

model = create_model(vocab_size+1, seq_len)
model.fit(X, y, batch_size=128, epochs=50, verbose=2)

from pickle import dump

model.save('hobbit_a.h5')
dump(tokenizer, open('hobbit_tokenizer_a', 'wb'))

from keras.preprocessing.sequence import pad_sequences

def generate_text(model, tokenizer, seq_len, seed_text, num_gen_words):
    output_text = []
    input_text = seed_text
    for i in range(num_gen_words):
        encoded_text = tokenizer.texts_to_sequences([input_text])[0]
        pad_encoded = pad_sequences([encoded_text], maxlen=seq_len, truncating='pre')
        pred_word_index = model.predict_classes(pad_encoded, verbose=0)[0]
        pred_word = tokenizer.index_word[pred_word_index]
        input_text += ' ' + pred_word
        output_text.append(pred_word)
    return ' '.join(output_text)

seed_text = 'The lonely band marched towards the distant hills stopping once to take a drink from the crystal river the July sun was'
hobbit_gen_text = generate_text(model, tokenizer, 20, seed_text, 50)

from kera.models import load_model

model = load_model('hobbit.h5')
tokenizer = load(open('hobbit_tokenizer', 'rb'))






















