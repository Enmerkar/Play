#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 13:37:26 2021

Test various NLP processes.
Use RNN trained on Tolkien corpus to generate new story.

Could even try blends: e.g. Tolkien + Dan Brown.

@author: JKN08.
"""

# Read Ubuntu
hobbit = open('data.txt')
fellowship_ring = open('/home/justin/Downloads/fellowship_ring.txt')
two_towers = open('/home/justin/Downloads/two_towers.txt')
return_king = open('/home/justin/Downloads/return_king.txt')

# Read Mac
hobbit = open('/Users/justinknife@qantas.com.au/Documents/hobbit.txt')
fellowship_ring = open('/Users/justinknife@qantas.com.au/Documents/fellowship_ring.txt')
two_towers = open('/Users/justinknife@qantas.com.au/Documents/two_towers.txt')
return_king = open('/Users/justinknife@qantas.com.au/Documents/return_king.txt')

# Clean the hobbit
# Or
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

hobbit_df['Chapter'].value_counts()

hobbit_df['Sentiment'] = hobbit_df['Text'].apply(lambda text: sid.polarity_scores(text)['compound'])

hobbit_df.groupby('Chapter').sum('Sentiment')






