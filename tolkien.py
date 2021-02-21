#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 13:37:26 2021

Test various NLP processes.
Use RNN trained on Tolkien corpus to generate new story.

Could even try blends: e.g. Tolkien + Dan Brown.

@author: JKN08.
"""

# Read all books
hobbit = open('/home/justin/Documents/Github/Play/data/hobbit.txt')
fellowship_ring = open('/home/justin/Documents/Github/Play/data/fellowship_ring.txt')
two_towers = open('/home/justin/Documents/Github/Play/data/two_towers.txt')
return_king = open('/home/justin/Documents/Github/Play/data/return_king.txt')

hobbit_lines = hobbit.readlines()

# Close all books
hobbit.close()
fellowship_ring.close()
two_towers.close()
return_king.close()


# Or
with open('/home/justin/Documents/Github/Play/data/hobbit.txt') as hobbit:
    hobbit_lines = hobbit.readlines()

# Remove blank lines.
# Delete top matter.
# Delete "Chapter" and succeeding rows.

import spacy

nlp_en = spacy.load('en_core_web_sm')

doc_en = nlp_en(u'My name is Justin and I am a human who likes eating various foodstuffs.')

for token in doc_en:
    print(f'{token.text:<10}', f'{token.lemma_:<10}', f'{token.pos_:<10}', f'{token.dep_:<10}')
    
nlp_en.pipeline



