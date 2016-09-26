#!/usr/bin/python3
"""
Build a tweet sentiment analyzer
"""

# stdlib
from collections import OrderedDict
import pickle as pkl
import time
import os
import sys
from contextlib import contextmanager

# scipy stack
import numpy as np
import scipy.io as sio

# theano
import theano
from theano import config
import theano.tensor as tensor

# charts
import matplotlib as mpl

from matplotlib import pyplot as plt
from lstm.data_io import load_data
from ext_argparse.argproc import process_arguments
from lstm.arguments import Arguments
from lstm.params import Parameters
from lstm.model import Model

mpl.rcParams['image.interpolation'] = 'nearest'


def main():
    args = process_arguments(Arguments, "Train & test an LSTM model on the given input.")
    # TODO: make model path an argument
    model_subdir = 'monocular_model'
    model_dir = os.path.join(args.folder, model_subdir)

    verbose = True
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    model_path = os.path.join(model_dir, 'lstm_model.npz')

    training_data, validation_data, test_data, n_categories, n_features = load_data(args.datasets, args.folder)
    # This create the initial parameters as np ndarrays.
    # This will create Theano Shared Variables from the model parameters.
    if args.reload_model:
        parameters = Parameters(archive=model_path)
    else:
        parameters = Parameters(n_features, args.hidden_unit_count, n_categories)

    model = Model(optimizer_name=args.optimizer, model_output_path=model_path, parameters=parameters,
                  random_seed=args.random_seed, decay_constant=args.decay_constant)

    # Turn interactive plotting off
    plt.ioff()

    if verbose:
        print("Sample counts:")
        print("%d Training sample count:" % len(training_data))
        print("%d Validation sample count:" % len(validation_data))
        print("%d Testing sample count:" % len(test_data))
        print('Training the model...')

    if args.overwrite_model or not os.path.isfile(model_path):
        model.train_on(training_data, validation_data, test_data, args.batch_size, args.validation_batch_size,
                       args.report_interval, args.validation_interval, args.save_interval, args.patience,
                       args.max_epochs, learning_rate=args.learning_rate)

    model.test_on(test_data)
    return 0


if __name__ == '__main__':
    plt.ion()
    sys.exit(main())
