# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 11:39:34 2016

@author: Chris
"""

# Explore Google's huge Word2Vec model.

import gensim
import logging

# Logging code taken from http://rare-technologies.com/word2vec-tutorial/
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Load Google's pre-trained Word2Vec model.
model = gensim.models.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)

# Does the model include stop words?
print("Does it include the stop words like \'a\', \'and\', \'the\'? %d %d %d" % ('a' in model.vocab, 'and' in model.vocab, 'the' in model.vocab))

# Retrieve the entire list of "words" from the Google Word2Vec model, and write
# these out to text files so we can peruse them.
vocab = list(model.vocab.keys())

fileNum = 1

wordsInVocab = len(vocab)
wordsPerFile = int(100E3)

# Write out the words in 100k chunks.
for wordIndex in range(0, wordsInVocab, wordsPerFile):
    # Write out the chunk to a numbered text file.    
    with open("vocabulary/vocabulary_%.2d.txt" % fileNum, 'w',encoding='utf-8') as f:
        # For each word in the current chunk...        
        for i in range(wordIndex, wordIndex + wordsPerFile):
            f.write(vocab[i]+'\n')

    
    fileNum += 1