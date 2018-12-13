#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 17:10:13 2018

@author: 8kd
"""
import json
import re
import read_reports_and_label
import itertools
import operator
from itertools import groupby
import numpy as np
import math
from matplotlib import pyplot as plt
import scipy.linalg


def sorted_distinct_labels(l, idx):
    return list(set(x[idx] for x in sorted_by_idx(l, idx)))

def sorted_by_idx(l, idx):
    l = [b for a,b in sorted((tup[idx], tup) for tup in l)]
    return l

def groupby_count(l, idx):
    l = sorted_by_idx(l, idx);
    label_counts =  [ (key, len(list(group))) for key, group in 
                 groupby([x[idx] for x in l])];
    return label_counts

def label2vec(l, idx, labelstr):
    vec = [x[idx]==labelstr for x in l];
    vec = np.multiply(vec, 1/math.sqrt(sum(vec)))
    return vec

dir                 = '/Users/8kd/workdir/data/cancer-reports/';
filename            = 'matched_fd.json';

# reading pathology reports as tuple
# (token list, organ, idc code, laterality, behavior, histology)
reports_with_any_labels = read_reports_and_label.parse(dir+filename);

# reports with organ labels only
reports_with_organ_labels = [x for x in reports_with_any_labels if x[1] != ''];
reports_with_icd_labels = [x for x in reports_with_any_labels if x[2] != ''];
reports_with_laterality_labels = [x for x in reports_with_any_labels if x[3] != ''];
reports_with_behavior_labels = [x for x in reports_with_any_labels if x[4] != ''];
reports_with_histology_labels = [x for x in reports_with_any_labels if x[5] != ''];
reports_with_all_labels =  [x for x in reports_with_any_labels if x[1] != '' and 
                        x[2]!='' and x[3]!='' and x[4]!='' and x[5]!=''];

# per class labels counts
organ_labels =  groupby_count(reports_with_organ_labels, 1);
icd_labels =  groupby_count(reports_with_icd_labels, 2);
laterality_labels =  groupby_count(reports_with_laterality_labels, 3);
behavior_labels =  groupby_count(reports_with_behavior_labels, 4);
histology_labels =  groupby_count(reports_with_histology_labels, 5);

# constructing co-relation matrix
num_of_labels = sum([len(sorted_distinct_labels(reports_with_all_labels, i)) 
                       for i in range(1, 6)]);
num_of_document = len(reports_with_all_labels);

label_nested_list = [sorted_distinct_labels(reports_with_all_labels, i)
                       for i in range(1,6)];

labellist = []; labelidxlist = [];
for i in range(len(label_nested_list)):
    labelidxlist = labelidxlist + [(i+1)]*len(label_nested_list[i]);
    labellist = labellist + label_nested_list[i]

l2vec = np.zeros((num_of_labels, num_of_document), float);
for i in range(len(labellist)):
    l2vec[i] = label2vec(reports_with_all_labels, labelidxlist[i], labellist[i])

label_correction_matrix = np.zeros((num_of_labels, num_of_labels), float);

# displaying co-relation matrix
fig = plt.figure;        
img=plt.imshow(label_correction_matrix, interpolation='none', cmap='gray');    
plt.colorbar()
plt.show() 

# QR decomposition
for i in range(len(labellist)):
    for j in range(len(labellist)):
        label_correction_matrix[i][j] = np.dot(l2vec[i], l2vec[j]);
[q, r] = scipy.linalg.qr(label_correction_matrix);   

# displaying r
fig = plt.figure;        
img=plt.imshow(np.abs(r), interpolation='none', cmap='gray');    
plt.colorbar()
plt.show() 