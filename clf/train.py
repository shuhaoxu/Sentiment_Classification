#!/usr/bin/python
# author : Shuhao Xu

from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

from clf.utils import load_data, generate_batch, split_train_dev
from clf.model import MITE

''' Fix working path '''
pth = os.getcwd()
os.chdir(pth)

''' Settings '''
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('embed_prefix', '../data/vec/', 'embedding path')
flags.DEFINE_string('data_prefix', '../data/idxs/', 'index path')
flags.DEFINE_string('word_vec', 'wevecs', 'option of word vector: wvecs, wevecs')
flags.DEFINE_string('mode', 'train', 'mode of running')
flags.DEFINE_integer('gpu', 1, 'appointed gpu')
flags.DEFINE_integer('epochs', 100, 'epoch of training model')
flags.DEFINE_float('lr', 1e-3, 'learning rate for optimizer')
flags.DEFINE_integer('batch_size', 128, 'mini-batch for each epoch')
flags.DEFINE_integer('hidden_size', 128, 'number of hidden layer')
flags.DEFINE_integer('d_model', 256, 'size of encoder output')
flags.DEFINE_integer('num_classes', 2, 'label category')
flags.DEFINE_integer('sents_len', 9, 'length of input sentence')
flags.DEFINE_integer('num_blocks', 1, 'number of encoder')
flags.DEFINE_integer('num_heads', 4, 'number of multi-heads')
flags.DEFINE_float('keep_prob', 0.5, 'keep prob')

''' GPU Setting '''
os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8

''' Hyper Parameters '''
embed_prefix = FLAGS.embed_prefix
data_prefix  = FLAGS.data_prefix
mode         = FLAGS.mode
lr           = FLAGS.lr
keep_prob    = FLAGS.keep_prob
epochs       = FLAGS.epochs
batch_size   = FLAGS.batch_size
hidden_size  = FLAGS.hidden_size
d_model      = FLAGS.d_model
num_blocks   = FLAGS.num_blocks
num_heads    = FLAGS.num_heads
sents_len    = FLAGS.sents_len
classes      = FLAGS.num_classes

''' File Name '''
word_vec_file  = embed_prefix + 'train_' + FLAGS.word_vec + '.npy'
train_sents_file = data_prefix + 'train_sents.txt'
train_label_file = data_prefix + 'train_label.txt'

if __name__ == '__main__':
    # embedding table 
    embed_matrix = np.load(word_vec_file).astype(np.float32)

    # load data
    train_Xs, train_Ys, train_Ys_dense = load_data(train_sents_file, train_label_file)

    # input placeholder
    with tf.name_scope("input"):
        X = tf.placeholder(tf.int32, [None, sents_len], name="X")
        Y = tf.placeholder(tf.int32, [None, classes],   name="Y")
        keep_prob = tf.placeholder(tf.float32)

    # load model
    model  = MITE(lr, keep_prob, epochs,
                  hidden_size, d_model, classes,
                  num_blocks, num_heads)

    # Operators
    logits  = model.build(X, embed_matrix)
    predict = tf.nn.softmax(logits, name='pred')
    ## loss
    loss_func = tf.nn.softmax_cross_entropy_with_logits(logits=predict,
                                                        labels=Y)
    loss_op   = tf.reduce_mean(loss_func, name='loss')
    ## optimizer
    global_step = tf.train.get_or_create_global_step()
    lr_op       = tf.Variable(lr, dtype=tf.float32)
    optimizer   = tf.train.AdamOptimizer(lr_op)
    add_global  = global_step.assign_add(1)
    ## accuracy
    correct_pred = tf.equal(tf.argmax(predict, 1),
                            tf.argmax(Y, 1),
                            name='correct_pred')
    pred_class   = tf.argmax(predict, 1, name='pred_class')
    acc_op       = tf.reduce_mean(tf.cast(correct_pred, tf.float32),
                                  name='acc')
    ## train operator
    train_op = optimizer.minimize(loss_op, global_step=global_step)

    # initialize
    init = tf.global_variables_initializer()

    # running
    if mode == 'train':
        max_F1  = 0.0
        with tf.Session(config=config) as sess:
            sess.run(init)
            all_f1  = []
            for step in range(epochs):
                # generate batch
                batch_x, batch_y, batch_y_dense = generate_batch(train_Xs, train_Ys, train_Ys_dense, batch_size)
                # feed_dict
                feed_dict = {X:batch_x, Y:batch_y, keep_prob:0.5}
                # training
                train_loss, train_acc, train_preds, _ = sess.run([loss_op, acc_op, pred_class, train_op],
                                                                 feed_dict=feed_dict)
                f1 = f1_score(batch_y_dense, train_preds)
                all_f1.append(f1)
                print ('Step {}, '.format(step) +
                       'train loss:{:4f}, '.format(train_loss) +
                       'train accuracy:{:4f}, '.format(train_acc) +
                       'train F1 score:{:4f}, '.format(f1),
                       end='\r')
                if f1 > max_F1:
                    max_F1 = f1
        print ('Optimization Finished.')
        print ('Max F1: {:4f}'.format(max_F1))
        print ('Epoch : {}'.format(all_f1.index(max_F1)))
