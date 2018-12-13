import numpy as np
import read_reports_and_label
from itertools import groupby
import scipy as sp
from matplotlib import pyplot as plt
# --------------------- helper functions -----------------------
def sorted_by_idx(l, idx):
    l = [b for a,b in sorted((tup[idx], tup) for tup in l)]
    return l
    
def sorted_distinct_labels(l, idx):
    return list(set(x[idx] for x in sorted_by_idx(l, idx)))

# ------------------ script ------------------------------------

dir                 = '/Users/8kd/workdir/data/cancer-reports/';
filename            = 'matched_fd.json';

reports_with_any_labels = read_reports_and_label.parse(dir+filename);
reports_with_organ_labels = [x for x in reports_with_any_labels if x[1] != ''];


reports = reports_with_organ_labels;

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
        
word_document_matrix_mask = np.logical_and(word_document_matrix > 2, word_document_matrix < 25);         
word_document_matrix = np.multiply(word_document_matrix_mask, word_document_matrix);
fig = plt.figure;
img=plt.imshow(np.log10(word_document_matrix +1), 
               interpolation='none', cmap='gray', aspect='auto');
plt.colorbar()
plt.show()


        
#[U, S, V] = sp.linalg.svd(word_document_matrix);

# Created by Abhishek K Dubey
# Date: Dec 13, 2018



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 15:58:47 2018

@author: 8kd
"""

