import cPickle
import gzip
import os
import json

import numpy
import scipy.io
import theano

# object_name = 'all';
# object_list = ['cup', 'stone', 'sponge', 'spoon', 'knife', 'spatula'];
object_list = ['al0'];
#subject_list = ['and', 'fer', 'gui', 'mic', 'kos']
# test_subject_list = ['and']
maxlen=None

def prepare_data(seqs, labels, maxlen=maxlen):
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


def load_data(test_subject_list, valid_portion=0.0):
    '''Loads the dataset

    :type path: String
    :param path: The path to the dataset (here IMDB)
    :type valid_portion: float
    :param valid_portion: The proportion of the full train set used for
        the validation set.
    '''

    #############
    # LOAD DATA #
    #############
    
    # if object_name == 'all':
    #     object_list = ['cup', 'stone', 'sponge', 'spoon', 'knife', 'spatula'];
    # else:
    #     object_list = [object_name];

    max_y = 0

    data_points = []
    data_labels = []
    data_meta = []

    feat_count = 0
    feat_mean = 0

    #print '\nTest_subject_list:', test_subject_list

    for obj in object_list:
        dataset_path = os.path.join('data', 'al', '%s_wo5_labels.json' % (obj))
        print 'loading data: %s' % (dataset_path, )
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)

        print '  dataset length: ', len(dataset)

        # load the image features into memory
        features_path = os.path.join('data', 'al', '%s.npy' % (obj))
        print 'loading features: %s' % (features_path, )
        features_struct = numpy.load(features_path)
        features = features_struct
        #features = numpy.concatenate((features_struct, features_struct), axis=1)
        print features.shape

        for d in dataset:
            #feats = numpy.transpose(features[:, d['start_fid']:d['end_fid']+1])
            feats = features[d['s_fid']:d['e_fid']+1, : ]

            feat_count += features.shape[0]
            feat_mean += numpy.sum(numpy.mean(features, axis=1))

            #if not d['subject'] in subject_list:
            #    continue

            data_y = d['label'] - 1

            # if d['test_flag'] == 1:
            #if d['subject'] in test_subject_list:

                # print d['subject'], test_subject_list

             #   test_set_x.append(feats)
             #   test_set_y.append(data_y + max_y)
             #   test_set_meta.append(d)

                # import pdb; pdb.set_trace()

            #else:
            #    if (d['attention_type'] + max_y) > 0:
            data_points.append(feats)
            data_labels.append(data_y + max_y)
            data_meta.append(d)

        y = [d['label'] for d in dataset]
        max_y += max(y)


    # feat_mean = feat_mean/feat_count
    # print 'Substract feature mean: ', feat_mean

    # for f in train_set_x:
    #     f -= feat_mean
    # for f in test_set_x:
    #     f -= feat_mean

    # split training set into validation set
    n_samples = len(data_points)
    sidx = numpy.random.permutation(n_samples) #not required
    #sidx = range(n_samples)
    test_ratio = 0.3
    train_ratio = 0.4
    test_count = int(numpy.round(n_samples*test_ratio))
    train_count = int(numpy.round(n_samples*train_ratio))
    start_train = test_count
    end_train = test_count + train_count
    start_valid = end_train

    test_set_x = [data_points[s] for s in sidx[0:test_count]]
    test_set_y = [data_labels[s] for s in sidx[0:test_count]]
    test_set_meta = [data_meta[s] for s in sidx[0:test_count]]

    train_set_x = [data_points[s] for s in sidx[start_train:end_train]]
    train_set_y = [data_labels[s] for s in sidx[start_train:end_train]]
    train_set_meta = [data_meta[s] for s in sidx[start_train:end_train]]

    valid_set_x = [data_points[s] for s in sidx[start_valid:]]
    valid_set_y = [data_labels[s] for s in sidx[start_valid:]]
    valid_set_meta = [data_meta[s] for s in sidx[start_valid:]]

    train = (train_set_x, train_set_y, train_set_meta)
    valid = (valid_set_x, valid_set_y, valid_set_meta)
    test = (test_set_x, test_set_y, test_set_meta)

    return train, valid, test
    