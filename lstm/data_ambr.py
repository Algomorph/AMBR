import os
import json

import numpy
import theano


object_list = ['al0'];
maxlen = None


def prepare_data(batch_x, batch_y, maxlen=maxlen):
    """Create the matrices from the datasets.

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.

    This swap the axis!
    """
    # x: a list of sentences
    lengths = [s.shape[0] for s in batch_x]
    feat_dim = batch_x[0].shape[1]

    if maxlen is not None:
        new_seqs = []
        new_labels = []
        new_lengths = []
        for l, s, y in zip(lengths, batch_x, batch_y):
            if l < maxlen:
                new_seqs.append(s)
                new_labels.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        batch_y = new_labels
        batch_x = new_seqs

        if len(lengths) < 1:
            return None, None, None

    n_samples = len(batch_x)
    maxlen = numpy.max(lengths)

    x = numpy.zeros((maxlen, n_samples, feat_dim)).astype(theano.config.floatX)
    x_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    for idx, s in enumerate(batch_x):
        x[:lengths[idx], idx, :] = s
        x_mask[:lengths[idx], idx] = 1.

    return x, x_mask, batch_y


def load_data():
    """Loads the dataset

    :type path: String
    :param path: The path to the dataset (here IMDB)
    :type valid_portion: float
    :param valid_portion: The proportion of the full train set used for
        the validation set.
    """

    #############
    # LOAD DATA #
    #############

    max_y = 0

    features_by_sample = []
    labels_by_sample = []
    meta_by_sample = []

    feat_count = 0
    feat_mean = 0

    for obj in object_list:
        base_data_path = os.path.join('/media/algomorph/Data/AMBR_data/train', 'data', 'al')
        dataset_path = os.path.join(base_data_path, '{:s}_wo5_labels.json'.format(obj))
        print('loading data: %s' % (dataset_path,))
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)

        print('  dataset length: ', len(dataset))

        # load the image features into memory
        features_path = os.path.join(base_data_path, "{:s}.npy".format(obj))
        print('loading features: %s' % (features_path,))
        all_features = numpy.load(features_path)

        for sample_data in dataset:
            sample_features = all_features[sample_data['s_fid']:sample_data['e_fid'] + 1, :]

            feat_count += all_features.shape[0]
            feat_mean += numpy.sum(numpy.mean(all_features, axis=1))

            sample_label = sample_data['label'] - 1

            features_by_sample.append(sample_features)
            labels_by_sample.append(sample_label + max_y)
            meta_by_sample.append(sample_data)

        y = [d['label'] for d in dataset]
        max_y += max(y)

    # split features into test, training, and validation set
    n_samples = len(features_by_sample)
    sidx = numpy.random.permutation(n_samples)
    test_ratio = 0.3
    train_ratio = 0.4
    test_count = int(numpy.round(n_samples * test_ratio))
    train_count = int(numpy.round(n_samples * train_ratio))
    start_train = test_count
    end_train = test_count + train_count
    start_valid = end_train

    test_set_x = [features_by_sample[s] for s in sidx[0:test_count]]
    test_set_y = [labels_by_sample[s] for s in sidx[0:test_count]]
    test_set_meta = [meta_by_sample[s] for s in sidx[0:test_count]]

    train_set_x = [features_by_sample[s] for s in sidx[start_train:end_train]]
    train_set_y = [labels_by_sample[s] for s in sidx[start_train:end_train]]
    train_set_meta = [meta_by_sample[s] for s in sidx[start_train:end_train]]

    validation_set_x = [features_by_sample[s] for s in sidx[start_valid:]]
    validation_set_y = [labels_by_sample[s] for s in sidx[start_valid:]]
    validation_set_meta = [meta_by_sample[s] for s in sidx[start_valid:]]

    train = (train_set_x, train_set_y, train_set_meta)
    valid = (validation_set_x, validation_set_y, validation_set_meta)
    test = (test_set_x, test_set_y, test_set_meta)

    return train, valid, test
