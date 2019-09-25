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

flags.DEFINE_string('pos_file', '../data/lexicon/ntusd-positive.txt', 'positive sentiment lexicon')
flags.DEFINE_string('neg_file', '../data/lexicon/ntusd-negative.txt', 'negative sentiment lexicon')
flags.DEFINE_string('embed_file', '../data/vec/train_wvecs.txt', 'training word vector text')
flags.DEFINE_string('vec_file', '../data/vec/train_wvecs.npy', 'training word vector npy')
flags.DEFINE_string('word_senti_embed', '../data/vec/train_wevecs', 'sentiment embedding')
flags.DEFINE_string('word_senti_pos_embed', '../data/vec/train_wepvecs', 'word, sentiment and pos embedding')

# Load training word-vector
def read_wordvec():
    embed_file = FLAGS.embed_file
    vec_file   = FLAGS.vec_file
    corp = [line.split()[0] for line in open(embed_file)]
    wvec = np.load(vec_file)
    return corp, wvec

# Get sentiment lexicon
def read_lexicon(pos_file, neg_file):
    pos_emotion = [line.strip() for line in open(pos_file)]
    neg_emotion = [line.strip() for line in open(neg_file)]
    return pos_emotion, neg_emotion

# Embed sentiment words
def emotion_plority_embed(corps, poss, negs):
    corp_emotion = {}
    for word in corps:
        if word in poss:
            corp_emotion[word] = 2
        elif word in negs:
            corp_emotion[word] = 1
        else:
            corp_emotion[word] = 0
    emotion_embed = np.eye(3) # others, negative, positive
    num, dim = len(corps), 3
    corp_emotion_embed = np.empty((num, dim), dtype=int)
    for i,w in enumerate(corps):
        eidx = corp_emotion[w]
        corp_emotion_embed[i] = emotion_embed[eidx]
    return corp_emotion_embed

if __name__ == '__main__':
    # get filenames
    pos_file = FLAGS.pos_file
    neg_file = FLAGS.neg_file
    word_senti_file = FLAGS.word_senti_embed
    word_senti_pos_file = FLAGS.word_senti_pos_embed
    
    # compute sentiment embedding
    corps, wvecs = read_wordvec()
    pos_corp, neg_corp = read_lexicon(pos_file, neg_file)
    emotion_embed = emotion_plority_embed(corps, pos_corp, neg_corp)

    # save word_sentiment embedding
    wevecs = np.hstack((wvecs, emotion_embed)).astype(float)
    np.save(word_senti_file, wevecs)
