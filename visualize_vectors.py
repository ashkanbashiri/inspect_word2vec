"""

Created on July 28, 2019
Author: Ashkan Bashiri
Coordinated Systems Lab
University of Virginia
"""

import gensim
import logging
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import re
import nltk

from gensim.models import word2vec

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []
    words= ['king', 'queen','man','woman','robot','tiger','jungle','cheetah','boy' \
        ,'girl','human','desert','fight','movie','tv','show','computer','notebook','python']

    '''
    # OR: visualize words from the model
    
    vocab = list(model.wv.vocab)
    words = vocab[1:100]
    '''

    for word in words:
        tokens.append(model[word])
        labels.append(word)

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Load Google's pre-trained Word2Vec model.
model = gensim.models.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)

tsne_plot(model)
