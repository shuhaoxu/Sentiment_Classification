#!/usr/bin/python
# author : Shuhao Xu, BUPT

from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf

# Fix working path
pth = os.getcwd() + '/preprocess/'
os.chdir(pth)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('mode', 'train', 'mode of current running style')
flags.DEFINE_integer('cut_length', 9, 'restricted length of sentence')
flags.DEFINE_string('train_file', '../data/raw_data/sentiment_XS_30k.txt', 'raw training file')
flags.DEFINE_string('test_file', '../data/raw_data/sentiment_XS_test.txt', 'raw test file')
flags.DEFINE_string('train_embed_file', '../data/vec/train_wvecs.txt', 'train embedding file')
flags.DEFINE_string('out_prefix', '../data/idxs/', 'write idx to file')

mode       = FLAGS.mode
cut_length = FLAGS.cut_length
out_prefix = FLAGS.out_prefix
train_embed_file = FLAGS.train_embed_file

# Read raw data
def read_corpus_file():
    if mode == 'train':
        filename = FLAGS.train_file
    else:
        filename = FLAGS.test_file
    records = [line.strip() for line in open(filename)]
    records = records[1:] # drop 1st line : 'labels, text'
    labels  = [line.split(',')[0] for line in records]
    sents   = [(line.split(',')[1]).split() for line in records]
    return sents,labels

# Read training word-vector text file
def read_embed_file(filename):
    records = [line.strip().split() for line in open(filename)]
    corps   = [records[i][0] for i in range(len(records)-1)]
    return corps
    
# Transform words to index 
def word2idx(sents, corps):
    idxs = []
    num  = len(corps)
    for line in sents:
        idx = []
        for i in range(min(cut_length, len(line))):
            word = line[i]
            if word in corps:
                idx.append(corps.index(word))
            else:
                idx.append(num)
        i += 1
        if i < cut_length:
            tmp = [num for _ in range(cut_length-i)]
            idx += tmp
        idxs.append(idx)
    return idxs

# Numerize labels
def convert2number(labels):
    uniql = np.unique(labels)
    l_map = {l:i for i,l in enumerate(uniql)}
    num_l = [l_map[l] for l in labels]
    return num_l

# Write to text file
def write2file(sents, label, sents_file, label_file):
    num = len(label)
    sfp = open(sents_file, 'w')
    lfp = open(label_file, 'w')
    for i in range(num):
        lfp.write(str(label[i]) + '\n')
        idxs = sents[i]
        for idx in idxs:
            sfp.write(str(idx) + ' ')
        sfp.write('\n')
    sfp.close()
    lfp.close()

if __name__ == '__main__':
    # read raw data
    sents, labels = read_corpus_file()
    label = convert2number(labels)
    label = np.asarray(label).astype(int)

    # get word vectors
    embed_corp = read_embed_file(train_embed_file)

    # transform & write out
    idxs  = word2idx(sents, embed_corp)
    sents_out = out_prefix + mode + '_sents.txt'
    label_out = out_prefix + mode + '_label.txt'
    write2file(idxs, label, sents_out, label_out)
