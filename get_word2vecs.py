"""

Created on July 28, 2019
Author: Ashkan Bashiri
Coordinated Systems Lab
University of Virginia
"""

import gensim
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Load Google's pre-trained Word2Vec model.
model = gensim.models.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)

vocab = list(model.vocab.keys())

"""
# This code only prints the first 5 elements in each word vector
selected_words = ['code','king','queen','man','woman']
for word in selected_words:
    vec = model.wv[word]
    print("Word: {s_word}\nVector = {vector}\n".format(s_word=word,vector=vec[0:5]))
"""
cntr = 0
for fileNum in range(1,31):
    with open("vocabulary/vocabulary_%.2d.txt" %fileNum, 'r',encoding='utf-8') as f:
        word = f.readline().rstrip('\n')
        while word:
            cntr+= 1
            print(cntr)
            vec = model.wv[word]
            print("Word: {s_word} \nVector = {vector} \n".format(s_word=word, vector=vec[0:5]))
            word = f.readline().rstrip('\n')

