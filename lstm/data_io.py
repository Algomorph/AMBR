import os
import json

import numpy as np
from numpy import inf
import theano


def prepare_data(batch_features, batch_labels, maximum_sequence_length=None):
    """Create a matrix from multiple sequences.
    This pads/cuts each sequence to the same length, which is the length of the longest sequence or
    maximum_sequence_length. Note: the the axes are swapped!
    """
    # x: a list of sentences
    lengths = [s.shape[0] for s in batch_features]
    feat_dim = batch_features[0].shape[1]

    if maximum_sequence_length is not None:
        new_seqs = []
        new_labels = []
        new_lengths = []
        for l, s, y in zip(lengths, batch_features, batch_labels):
            if l < maximum_sequence_length:
                new_seqs.append(s)
                new_labels.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        batch_labels = new_labels
        batch_features = new_seqs

        if len(lengths) < 1:
            return None, None, None

    n_samples = len(batch_features)
    maximum_sequence_length = np.max(lengths)

    x = np.zeros((maximum_sequence_length, n_samples, feat_dim)).astype(theano.config.floatX)
    x_mask = np.zeros((maximum_sequence_length, n_samples)).astype(theano.config.floatX)
    for idx, s in enumerate(batch_features):
        x[:lengths[idx], idx, :] = s
        x_mask[:lengths[idx], idx] = 1.

    return x, x_mask, batch_labels


class SequenceDataset(object):
    """
    Represents a set of sequences, where each sequence time step is represented by a vector of features.
    Each sequences has a label associated with it.
    Meta information is just the way the sequence was represented in the file.
    """
    def __init__(self, sequence_features, sequence_labels, meta_information, weights=None):
        self.features = sequence_features
        self.labels = sequence_labels
        self.meta_information = meta_information
        if len(self.features) != len(self.labels) or len(self.features) != len(self.meta_information):
            raise ValueError("Expecting same numbers of features, labels, and meta data. " +
                             "Got: {:d}, {:d}, and {:d}, respectively."
                             .format(len(self.features), len(self.labels), len(self.meta_information)))
        self.weights = weights
        if weights is not None:
            if len(self.features) != len(weights):
                raise ValueError("Expecting the same number of weights as the number of samples. " +
                                 "Got {:d} and {:d}, respectively.".format(len(self.weights), len(self.features)))

    def __len__(self):
        return len(self.features)


def load_data(datasets, base_work_folder):
    """Loads the dataset
    :type path: String
    :param path: The path to the dataset
    :type valid_portion: float
    :param valid_portion: The proportion of the full train set used for
        the validation set.
    :rtype: (lstm.data_io.SequenceDataset, lstm.data_io.SequenceDataset, lstm.data_io.SequenceDataset, int, int)
    :return training, validation, and testing dataset, the category and the feature counts
    """

    features_by_sample = []
    labels_by_sample = []
    meta_by_sample = []

    for obj in datasets:
        base_data_path = os.path.join(base_work_folder, 'data')
        dataset_path = os.path.join(base_data_path, '{:s}_labels.json'.format(obj))
        print('  loading labels from: %s' % (dataset_path,))
        with open(dataset_path, 'r') as f:
            label_entries = json.load(f)

        print('  dataset length: ', len(label_entries))

        # load the image features into memory
        features_path = os.path.join(base_data_path, "{:s}_features.npz".format(obj))
        print('  loading features from: %s' % (features_path,))
        archive = np.load(features_path)
        all_features = archive["features"]

        for entry in label_entries:
            sample_features = all_features[entry['start']:entry['end'] + 1, :]
            if len(sample_features) == 0:
                raise ValueError("Got input sequence length 0: {:s}".format(str(entry)))

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
    randomized_index = np.random.permutation(n_samples)
    test_ratio = 0.2
    train_ratio = 0.6
    test_count = int(np.round(n_samples * test_ratio))
    train_count = int(np.round(n_samples * train_ratio))
    start_train = test_count
    end_train = test_count + train_count
    start_valid = end_train

    train_set_x = [features_by_sample[s] for s in randomized_index[start_train:end_train]]
    train_set_y = [labels_by_sample[s] for s in randomized_index[start_train:end_train]]
    train_set_meta = [meta_by_sample[s] for s in randomized_index[start_train:end_train]]

    # compute weights
    sample_counts_per_label = np.zeros(n_categories, np.int32)
    for label in train_set_y:
        sample_counts_per_label[label] += 1

    weights_by_label = len(train_set_x) / (np.count_nonzero(sample_counts_per_label) * sample_counts_per_label)
    weights_by_label[weights_by_label == inf] = 0
    train_set_weights = [weights_by_label[y] for y in train_set_y]

    test_set_x = [features_by_sample[s] for s in randomized_index[0:test_count]]
    test_set_y = [labels_by_sample[s] for s in randomized_index[0:test_count]]
    test_set_meta = [meta_by_sample[s] for s in randomized_index[0:test_count]]

    validation_set_x = [features_by_sample[s] for s in randomized_index[start_valid:]]
    validation_set_y = [labels_by_sample[s] for s in randomized_index[start_valid:]]
    validation_set_meta = [meta_by_sample[s] for s in randomized_index[start_valid:]]

    training_set = SequenceDataset(train_set_x, train_set_y, train_set_meta, train_set_weights)
    validation_set = SequenceDataset(validation_set_x, validation_set_y, validation_set_meta)
    test_set = SequenceDataset(test_set_x, test_set_y, test_set_meta)

    n_features = len(test_set_x[0])

    return training_set, validation_set, test_set, n_categories, n_features
