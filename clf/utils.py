#!/usr/bin/python
# author : Shuhao Xu

from __future__ import division
from __future__ import print_function

import random
import numpy as np

def load_data(sents_file, label_file):
    # sents id, labels
    sents_ids = [line.strip().split() for line in open(sents_file)]
    #import IPython; IPython.embed()
    sents_ids = np.asarray(sents_ids).astype(np.int32)
    label     = [int(line.strip()) for line in open(label_file)]
    label_onehot = np.eye(2)[label]
    return sents_ids, label_onehot, label

def generate_batch(sents, label, label_dense, batch_size):
    smp_idxs  = random.sample(range(0, sents.shape[0]), batch_size)
    label_dense = np.asarray(label_dense).astype(int)
    sents_smp = sents[smp_idxs]
    #import IPython; IPython.embed()
    label_smp = label[smp_idxs]
    label_dense_smp = label_dense[smp_idxs]
    return sents_smp, label_smp, label_dense_smp

def split_train_dev(data, label, label_dense, dev_num):
    num       = data.shape[0]
    idxs      = np.arange(0, num)
    random.shuffle(idxs)
    label_dense = np.asarray(label_dense).astype(int)
    # generate index
    dev_ids   = list(idxs[:dev_num])
    train_ids = list(idxs[dev_num:])
    #import IPython; IPython.embed()
    # get data
    trainx    = data[train_ids]
    devx      = data[dev_ids]
    # get onehot label
    trainy    = label[train_ids]
    devy      = label[dev_ids]
    # get normal label
    trainy_dense = label_dense[train_ids]
    devy_dense   = label_dense[dev_ids]
    return trainx, trainy, trainy_dense, devx, devy, devy_dense
