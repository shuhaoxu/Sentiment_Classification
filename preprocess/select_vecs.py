#!/usr/bin/python
# author : Shuhao Xu

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

flags.DEFINE_string('wordvec_file', None, 'pretrained word vector')
flags.DEFINE_string('train_corpus', None, 'training data corpus')
flags.DEFINE_string('out_prefix', '../data/vec/train_wvecs', 'prefix of selected vector')

# Load training corpus
def load_train_corps(filename):
    corps = [line.strip().split() for line in open(filename)]
    return corps[0]

# Load pre-trained word vetors
def load_raw_wordvecs(filename):
    records = [line.strip().split() for line in open(filename)]
    line    = records[0]
    records = records[1:]
    num, dim = int(line[0]), int(line[1])
    corps   = []
    wvecs   = np.empty((num, dim))
    for i in range(num):
        line = records[i]
        corps.append(line[0])
        vec  = np.asarray(line[1:]).astype(np.float32)
        wvecs[i] = vec
    return corps, wvecs

# Select word vectors according to training corpus
def select_train_wvecs(train_corp, embed_corp, wvecs):
    inter_corp  = list(set(train_corp).intersection(set(embed_corp)))
    num, dim    = len(inter_corp), 300
    train_wvecs = np.empty((num+1, dim))
    for i, word in enumerate(inter_corp):
        idx = embed_corp.index(word)
        train_wvecs[i] = wvecs[idx]
    # add default word vector
    mean = np.mean(train_wvecs[:-1,:], 0)
    train_wvecs[-1,:] = mean.copy()
    return inter_corp, train_wvecs

if __name__ == '__main__':
    # settings
    train_corpus = FLAGS.train_corpus
    wordvec_file = FLAGS.wordvec_file
    out_prefix   = FLAGS.out_prefix

    # load init data
    train_corp = load_train_corps(train_corpus)
    embed_corp, wvecs = load_raw_wordvecs(wordvec_file)

    inter_corp, train_wvecs = select_train_wvecs(train_corp, embed_corp, wvecs)
   
    # save selected word vectors
    np.save(out_prefix, train_wvecs)    
    with open(out_prefix+'.txt', 'w') as fp:
        for i in range(len(inter_corp)):
            word = inter_corp[i]
            wvec = train_wvecs[i].copy()
            fp.write(str(word) + ' ')
            for v in wvec:
                fp.write(str(v) + ' ')
            fp.write('\n')
        word = '<default>'
        wvec = train_wvecs[-1,:].copy()
        for v in wvec:
            fp.write(str(v) + ' ')
