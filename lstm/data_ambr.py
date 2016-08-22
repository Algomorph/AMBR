import os
import json

import numpy as np
from numpy import inf
import theano

object_list = ['al0'];
maxlen = None


def prepare_data(batch_x, batch_y, maxlen=maxlen):
    """Create the matrices from the datasets.

    This pad each sequence to the same length: the length of the
    longuest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    length.

    This swaps the axes!
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
    maxlen = np.max(lengths)

    x = np.zeros((maxlen, n_samples, feat_dim)).astype(theano.config.floatX)
    x_mask = np.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    for idx, s in enumerate(batch_x):
        x[:lengths[idx], idx, :] = s
        x_mask[:lengths[idx], idx] = 1.

    return x, x_mask, batch_y


def load_data(datasets, base_work_folder):
    """Loads the dataset

    :type path: String
    :param path: The path to the dataset (here IMDB)
    :type valid_portion: float
    :param valid_portion: The proportion of the full train set used for
        the validation set.
    """

    features_by_sample = []
    labels_by_sample = []
    meta_by_sample = []

    feat_count = 0
    feat_mean = 0

    for obj in datasets:
        base_data_path = os.path.join(base_work_folder, 'data')
        dataset_path = os.path.join(base_data_path, '{:s}_labels.json'.format(obj))
        print('  loading labels from: %s' % (dataset_path,))
        with open(dataset_path, 'r') as f:
            label_entries = json.load(f)

        print('  dataset length: ', len(label_entries))

        # load the image features into memory
        features_path = os.path.join(base_data_path, "{:s}_vgg.npz".format(obj))
        print('  loading features from: %s' % (features_path,))
        archive = np.load(features_path)
        all_features = archive[archive.files[0]]

        for entry in label_entries:
            sample_features = all_features[entry['beginning']:entry['end'] + 1, :]
            if len(sample_features) == 0:
                raise ValueError("Got input sequence length 0: {:s}".format(str(entry)))

            feat_count += all_features.shape[0]
            feat_mean += np.sum(np.mean(all_features, axis=1))

            sample_label = entry['label']

            features_by_sample.append(sample_features)
            labels_by_sample.append(sample_label)
            meta_by_sample.append(entry)

    features_by_sample = np.array(features_by_sample)
    labels_by_sample = np.array(labels_by_sample)

    unique_labels = np.unique(labels_by_sample)
    n_categories = len(unique_labels)

    # split features into test, training, and validation set
    n_samples = len(features_by_sample)
    sidx = np.random.permutation(n_samples)
    test_ratio = 0.2
    train_ratio = 0.6
    test_count = int(np.round(n_samples * test_ratio))
    train_count = int(np.round(n_samples * train_ratio))
    start_train = test_count
    end_train = test_count + train_count
    start_valid = end_train

    train_set_x = [features_by_sample[s] for s in sidx[start_train:end_train]]
    train_set_y = [labels_by_sample[s] for s in sidx[start_train:end_train]]
    train_set_meta = [meta_by_sample[s] for s in sidx[start_train:end_train]]

    # compute weights
    sample_counts_per_label = np.zeros(n_categories, np.int32)
    for label in train_set_y:
        sample_counts_per_label[label] += 1

    weights_by_label = len(train_set_x) / (np.count_nonzero(sample_counts_per_label) * sample_counts_per_label)
    weights_by_label[weights_by_label == inf] = 0
    print(weights_by_label)
    train_set_weights = [weights_by_label[y] for y in train_set_y]

    test_set_x = [features_by_sample[s] for s in sidx[0:test_count]]
    test_set_y = [labels_by_sample[s] for s in sidx[0:test_count]]
    test_set_meta = [meta_by_sample[s] for s in sidx[0:test_count]]

    validation_set_x = [features_by_sample[s] for s in sidx[start_valid:]]
    validation_set_y = [labels_by_sample[s] for s in sidx[start_valid:]]
    validation_set_meta = [meta_by_sample[s] for s in sidx[start_valid:]]

    train_set = (train_set_x, train_set_y, train_set_meta, train_set_weights)
    validation_set = (validation_set_x, validation_set_y, validation_set_meta)
    test_set = (test_set_x, test_set_y, test_set_meta)

    return train_set, validation_set, test_set, n_categories
