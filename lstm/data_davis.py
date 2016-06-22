import cPickle
import gzip
import os
import json

import numpy
import scipy.io
import theano

object_name = 'stone'

def prepare_data(seqs, labels, maxlen=None):
    """Create the matrices from the datasets.

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.

    This swap the axis!
    """
    # x: a list of sentences
    lengths = [s.shape[0] for s in seqs]
    feat_dim = seqs[0].shape[1]

    if maxlen is not None:
        new_seqs = []
        new_labels = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, labels):
            if l < maxlen:
                new_seqs.append(s)
                new_labels.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        labels = new_labels
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None, None

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)

    x = numpy.zeros((maxlen, n_samples, feat_dim)).astype(theano.config.floatX)
    x_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    for idx, s in enumerate(seqs):
        x[:lengths[idx], idx, :] = s
        x_mask[:lengths[idx], idx] = 1.

    return x, x_mask, labels


def load_data():
    '''Loads the dataset

    :type path: String
    :param path: The path to the dataset (here IMDB)
    '''

    #############
    # LOAD DATA #
    #############

    train_set_x = []
    train_set_y = []
    test_set_x = []
    test_set_y = []

    data_file = 'features/features_%s.mat' % object_name
    data_mat = scipy.io.loadmat(data_file)

    dataset = data_mat['list_features'][0]
    action_vec = data_mat['Y']['action'][0,0][0]
    actor_vec = data_mat['Y']['actor'][0,0][0]

    for i, feats in enumerate(dataset):
        train_set_x.append(feats.transpose())
        train_set_y.append(action_vec[i]-1)


    # split training set into validation set
    n_samples = len(train_set_x)
    sidx = numpy.random.permutation(n_samples)
    # n_train = n_samples
    test_portion = 0.2  # 1 for each 5 action
    valid_portion = 0.1  # 10% of training data
    n_train_total = int(numpy.round(n_samples * (1. - test_portion)))
    n_valid = int(numpy.round(n_train_total * valid_portion))
    n_train = n_train_total - n_valid

    print 'Training Set :', sidx[:n_train]
    print 'Valid Set :', sidx[n_train:n_train_total]
    print 'Testing Set  :', sidx[n_train_total:]

    test_set_x  = [train_set_x[s] for s in sidx[n_train_total:]]
    test_set_y  = [train_set_y[s] for s in sidx[n_train_total:]]
    valid_set_x  = [train_set_x[s] for s in sidx[n_train:n_train_total]]
    valid_set_y  = [train_set_y[s] for s in sidx[n_train:n_train_total]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    train = (train_set_x, train_set_y, [])
    valid = (valid_set_x, valid_set_y, [])
    test = (test_set_x, test_set_y, [])

    return train, valid, test

