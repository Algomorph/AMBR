#!/usr/bin/python3
"""
Build a tweet sentiment analyzer
"""

# stdlib
import os
import sys

# charts
import matplotlib as mpl

import numpy as np

from matplotlib import pyplot as plt
from lstm.data_io import load_data, load_multiview_data, load_test_data, load_multiview_test_data
from ext_argparse.argproc import process_arguments
from lstm.arguments import Arguments
from lstm.params import Parameters
from lstm.network import Network

mpl.rcParams['image.interpolation'] = 'nearest'


def main():
    args = process_arguments(Arguments, "Train & test an LSTM model on the given input.")

    verbose = True

    model_path = os.path.join(args.folder, args.model_file)
    model_dir = os.path.dirname(model_path)

    # make subfolders for model file (if there are some specified in the model_file but they don't yet exist)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if args.test_entire_data:
        if args.multiview_labels is None:
            test_data, n_categories, n_features = \
                load_test_data(args.datasets, args.folder)
        else:
            test_data, test_groups, n_categories, n_features = \
                load_multiview_test_data(args.datasets, args.folder, args.multiview_labels)
    else:
        if args.multiview_labels is None:
            training_data, validation_data, test_data, n_categories, n_features = \
                load_data(args.datasets, args.folder, args.validation_ratio, args.test_ratio,
                          randomization_seed=args.random_seed)
        else:
            training_data, validation_data, test_data, test_groups, n_categories, n_features = \
                load_multiview_data(args.datasets, args.folder, args.multiview_labels,
                                    args.validation_ratio, args.test_ratio, randomization_seed=args.random_seed)

    # This create the initial parameters as numpy arrays.
    # This will create Theano Shared Variables from the model parameters.
    if os.path.exists(model_path) and not args.overwrite_model:
        parameters = Parameters(archive=np.load(model_path))
    else:
        parameters = Parameters(n_features, args.hidden_unit_count, n_categories)

    network = Network(optimizer_name=args.optimizer, model_output_path=model_path, parameters=parameters,
                      random_seed=args.random_seed, decay_constant=args.decay_constant)

    # Turn interactive plotting off
    plt.ioff()

    if not args.test_entire_data:
        if verbose:
            print("--------------------")
            print("SAMPLE COUNTS")
            print("Training: %d " % len(training_data))
            print("Validation: %d" % len(validation_data))
            print("Test: %d" % len(test_data))
            print("--------------------")

        if args.overwrite_model or not os.path.isfile(model_path):
            network.train(training_data, validation_data, test_data, args.batch_size, args.validation_batch_size,
                          args.report_interval, args.validation_interval, args.save_interval, args.patience,
                          args.max_epochs, learning_rate=args.learning_rate)

    if args.multiview_labels:
        network.test(test_data)
        network.multiview_test(test_groups)
    else:
        network.test(test_data)
    return 0


if __name__ == '__main__':
    plt.ion()
    sys.exit(main())
