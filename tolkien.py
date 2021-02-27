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
hobbit = open('/home/justin/Downloads/hobbit.txt')
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
with open('/home/justin/Downloads/hobbit.txt') as hobbit:
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

#hobbit_utf = hobbit_string.encode(encoding='UTF-8', errors='strict')

hobbit_doc = nlp_en(hobbit_string)

doc_en = nlp_en(u'My name is Justin and I am a human who likes eating various foodstuffs.')

for token in doc_en:
    print(f'{token.text:<10}', f'{token.lemma_:<10}', f'{token.pos_:<10}', f'{token.dep_:<10}')
    
nlp_en.pipeline

hobbit_doc

# TF-IDF

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

vect = TfidfVectorizer()
dtm = vect.fit_transform(hobbit_clean)
print(vect.get_feature_names())

c_vect = CountVectorizer()






