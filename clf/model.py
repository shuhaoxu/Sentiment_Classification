#!/usr/bin/python
# author : Shuhao Xu

from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf

from clf.modules import ln, get_token_embeddings, positional_encoding, noam_scheme, scaled_dot_product_attention
from clf.modules import multihead_attention, ff

# Fix working path
pth = os.getcwd() + '/clf/'
os.chdir(pth)

class MITE(object):
    def __init__(self, lr, keep_prob, epochs, 
                 hidden_size, d_model, classes,
                 num_blocks, num_heads):
        self.lr          = lr
        self.drop_rate   = 1 - keep_prob
        self.epochs      = epochs
        self.hidden_size = hidden_size
        self.d_model     = d_model 
        self.num_blocks  = num_blocks
        self.num_heads   = num_heads
        self.classes     = classes

    def build(self, X, embed_mat, training=True):
        # Lookup word vectors
        X_embed = tf.nn.embedding_lookup(embed_mat, X)
        X_embed = tf.layers.dropout(X_embed, (self.drop_rate), training=training)
        # Full Connected Layer
        enc = tf.layers.dense(X_embed, self.d_model, activation=tf.nn.relu)
        enc *= self.d_model ** 0.5
        # Transformer Block
        for i in range(self.num_blocks):
            with tf.variable_scope('num_blocks_{}'.format(i)):
                # self-attention
                enc = multihead_attention(queries=enc,
                                          keys=enc,
                                          values=enc,
                                          num_heads=self.num_heads,
                                          dropout_rate=self.drop_rate,
                                          training=training,
                                          causality=False)
                # feed forward
                enc = ff(enc, num_units=[self.hidden_size, self.d_model])
        # Decoder for classify
        dec = tf.contrib.layers.flatten(enc)
        dec = tf.layers.dropout(dec, (self.drop_rate), training=training)
        dec = tf.layers.dense(dec, self.hidden_size, activation=tf.nn.relu)
        # Classify Layer
        outputs = tf.layers.dense(dec, self.classes, activation=tf.nn.relu)
        return outputs
