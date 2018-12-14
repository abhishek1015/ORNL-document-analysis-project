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

#construct sorted word list
word_list = [];
for doc in reports:
    word_list = word_list + list(set(doc[0]));

word_list = list(set(word_list));
word_list = sorted(word_list);  

# give an index to each word 
wordidx = dict();
i=0;
for x in word_list:
    wordidx[x] = i;
    i=i+1;

num_of_words = len(word_list);
num_of_columns = len(reports);

# generate word-document matrix
# row: one word each row
# col: one document each column
word_document_matrix = np.zeros((num_of_words, num_of_columns), int);

for idoc in range(len(reports)):
    temp_wl = reports[idoc][0];
    temp_wl = sorted(temp_wl);
    word_counts = [(key, len(list(group))) for key, group 
                   in groupby(temp_wl)];
    for word_count in word_counts:                
        word_document_matrix[wordidx[word_count[0]], idoc] = word_count[1]; 
        
#word_document_matrix_mask = word_document_matrix.sum(axis=1) > 3 ;
                   
#word_document_matrix = \
#    mask(word_document_matrix, word_document_matrix_mask);
    
plot_fig(np.log2(word_document_matrix + 1))

# generate word-class matrix
word_class_matrix =  \
    classembedding(word_document_matrix, reports, taskid);
labellist = sorted_distinct_labels(reports, taskid);
word_class_matrix = \
sklearn.preprocessing.normalize(word_class_matrix, norm='l1', axis=1);
plot_fig(word_class_matrix, True)

# document based filtering
valid_word = word_document_matrix.sum(axis=1) > 5

# class based filtering
valid_word= np.logical_and(((word_class_matrix>0.5).sum(axis=1) != 0),
                           valid_word);


word_unique_per_class = [x for x, v in zip(word_list, valid_word) 
    if v == 1]
# word-word corelation
#C = np.matmul(word_class_matrix, np.transpose(word_class_matrix));
#C_CSC = sp.sparse.csc_matrix(C);
#perm = sp.sparse.csgraph.reverse_cuthill_mckee(C_CSC);
#C_reordered = np.copy(C);
#C_reordered[np.arange(num_of_words), :] = C_reordered[perm, :];
#C_reordered[:, np.arange(num_of_words)] = C_reordered[:, perm];
#plot_fig(C)
#plot_fig(C_reordered)
#plot_fig(logistic.cdf(C_reordered))
#P, L, U = sp.linalg.lu(C);
#plot_fig(abs(L))


    

# manifold learning
#import sklearn.manifold

#Y = sklearn.manifold.locally_linear_embedding(
#       word_document_matrix, n_neighbors=25, n_components=10);
        
#fig = plt.figure;
#img=plt.imshow(Y, interpolation='none', cmap='gray', aspect='auto');
#plt.colorbar()
#plt.show()

        
#[U, S, V] = sp.linalg.svd(word_document_matrix);

# Created by Abhishek K Dubey
# Date: Dec 13, 2018



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 15:58:47 2018

@author: 8kd
"""

