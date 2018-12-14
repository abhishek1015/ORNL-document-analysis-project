import numpy as np
import read_reports_and_label
from itertools import groupby
import scipy as sp
from matplotlib import pyplot as plt
import math
from scipy.stats import logistic
import sklearn

# --------------------- helper functions -----------------------



def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def sorted_by_idx(l, idx):
    l = [b for a,b in sorted((tup[idx], tup) for tup in l)]
    return l
    
def sorted_distinct_labels(l, idx):
    l =list(set(x[idx] for x in sorted_by_idx(l, idx)))
    return sorted(l)

def plot_fig(img, withcolorbar=True):
    plt.figure
    img=plt.imshow(img, interpolation='none', cmap='gray', aspect='auto');
    if withcolorbar:
        plt.colorbar()
    plt.show()
    
def mask(img, mask):
    return np.multiply(img, mask);

def classembedding(word_document_embedding, reports, classidx):
    labellist = sorted_distinct_labels(reports, classidx);
    num_of_labels = len(labellist);
    num_of_words = word_document_embedding.shape[0];
    num_of_documents = word_document_embedding.shape[1];
    word_label_matrix = np.zeros((num_of_words, num_of_labels), int);
   # pdb.set_trace();
    for lidx in range(num_of_labels):
        labelpos = [report[classidx] == labellist[lidx] for report in reports];
        word_label_matrix[:, lidx] = word_document_embedding[:, labelpos].sum(axis=1);      
    return word_label_matrix
                   
# ------------------ script ------------------------------------

dir                 = '/Users/8kd/workdir/data/cancer-reports/';
filename            = 'matched_fd.json';

taskid = 2; # icd classification

reports_with_any_labels = read_reports_and_label.parse(dir+filename);

reports = [x for x in reports_with_any_labels if x[taskid] != ''];

from gensim.models import Phrases
documents = ["the mayor of new york was there", "machine learning can be useful sometimes","new york mayor was present"]

sentence_stream = [doc.split(" ") for doc in documents]
bigram = Phrases(sentence_stream, min_count=1, threshold=2)
sent = [u'the', u'mayor', u'of', u'new', u'york', u'was', u'there']
print(bigram[sent])


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 15:58:47 2018

@author: 8kd
"""

