#  ================================================================
#  Created by Gregory Kramida on 8/11/16.
#  Copyright (c) 2016 Gregory Kramida
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ================================================================

import numpy as np

import sys
import os
from contextlib import contextmanager
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
with suppress_stdout():
    import theano
    from theano import config


def generate_random_orthogonal_vectors(size):
    """
    Generates a square matrix of [size x size] elements, where each row is orthogonal to every other row, but the
    vector directions are random
    :param size: size of the square output matrix
    :return: a square matrix of [size x size] elements, where each row is orthogonal to every other row, but the
    vector directions are random
    """
    initial_random_weights = np.random.randn(size, size)
    u, s, v = np.linalg.svd(initial_random_weights)
    return u.astype(config.floatX)


class Model(object):
    class Globals(object):
        embedded_weights_literal = "embedding_layer_weights"
        classifier_weights_literal = "classifier_weights"
        classifier_bias_literal = "classifier_bias"

        def __init__(self, feature_count=None, hidden_unit_count=None, category_count=None, archive=None):
            if archive is None and (feature_count is None or hidden_unit_count is None or category_count is None):
                raise ValueError(
                    "If archive is not passed in, an " + Model.Globals.__name__ +
                    " object needs all other constructor arguments to be integers.")
            if archive is None:
                self.embedding_weights = \
                    (0.01 * np.random.rand(feature_count, hidden_unit_count)).astype(config.floatX)  # formerly Wemb
                self.classifier_weights = \
                    0.01 * np.random.randn(hidden_unit_count, category_count).astype(config.floatX)  # formerly U
                self.classifier_bias = np.zeros((category_count,)).astype(config.floatX)  # formerly b
            else:
                self.embedding_weights = archive[self.embedded_weights_literal]
                self.classifier_weights = archive[self.classifier_weights_literal]
                self.classifier_bias = archive[self.classifier_bias_literal]

        def write_to_dict(self, archive_dict):
            archive_dict[self.embedded_weights_literal] = self.embedding_weights
            archive_dict[self.classifier_weights_literal] = self.classifier_weights
            archive_dict[self.classifier_bias_literal] = self.classifier_bias

    class LSTM(object):
        input_weights_literal = "lstm_input_weights"
        hidden_weights_literal = "lstm_hidden_weights"
        bias_literal = "lstm_bias"

        def __init__(self, hidden_unit_count=None, archive=None):
            if archive is None and hidden_unit_count is None:
                raise ValueError(
                    "If archive is not passed in, an " + Model.LSTM.__name__ +
                    " object needs hidden_unit_count argument to be an integer.")
            if archive is None:
                gen__r_o_v = generate_random_orthogonal_vectors
                self.input_weights = np.concatenate([gen__r_o_v(hidden_unit_count),
                                                     gen__r_o_v(hidden_unit_count),
                                                     gen__r_o_v(hidden_unit_count),
                                                     gen__r_o_v(hidden_unit_count)], axis=1)  # formerly lstm_W
                self.hidden_weights = np.concatenate([gen__r_o_v(hidden_unit_count),
                                                      gen__r_o_v(hidden_unit_count),
                                                      gen__r_o_v(hidden_unit_count),
                                                      gen__r_o_v(hidden_unit_count)], axis=1)  # formerly lstm_U

                self.bias = np.zeros((4 * hidden_unit_count,)).astype(config.floatX)  # formerly lstm_b
            else:
                self.input_weights = archive[self.input_weights_literal]
                self.hidden_weights = archive[self.hidden_weights_literal]
                self.bias = archive[self.bias_literal]

        def write_to_dict(self, archive_dict):
            archive_dict[self.input_weights_literal] = self.input_weights
            archive_dict[self.hidden_weights_literal] = self.hidden_weights
            archive_dict[self.bias_literal] = self.bias

    def __init__(self, feature_count=None, hidden_unit_count=None, category_count=None, archive=None):
        if archive is None and (feature_count is None or hidden_unit_count is None or category_count is None):
            raise ValueError(
                "If archive is not passed in, an " + Model.__name__ +
                " object needs all other constructor arguments to be integers.")
        if archive is None:
            self.feature_count = feature_count
            self.hidden_unit_count = hidden_unit_count
            self.category_count = category_count
            self.globals = Model.Globals(feature_count, hidden_unit_count, category_count)
            self.lstm = Model.LSTM(hidden_unit_count)
        else:
            self.globals = Model.Globals(archive=archive)
            self.lstm = Model.Globals(archive=archive)
            # infer basic settings from globals
            self.hidden_unit_count = self.globals.classifier_bias.shape[0]
            self.feature_count = self.globals.embedding_weights.shape[0]
            self.category_count = self.globals.classifier_weights.shape[1]

    def save_to_archive(self, path):
        archive_dict = {}
        self.globals.write_to_dict(archive_dict)
        self.lstm.write_to_dict(archive_dict)
        np.savez(path, **archive_dict)


# Avoid this function, instead make everything explicit
def convert_to_theano(input_dict, output_dict):
    """
    :param output_dict: dictionary of theano shared (host/device) arrays
    :type input_dict: dict
    :param input_dict: dictionary of numpy arrays
    """
    for key, value in input_dict.items():
        output_dict[key] = theano.shared(value, name=key)


class TheanoModel(object):
    """
    A Theano representation of LSTM parameters
    """

    class Globals(object):
        def __init__(self, globals_):
            """
            :type globals_: Model.Globals
            :param globals_: non-theano global parameters (using numpy arrays)
            """
            self.embedding_weights = theano.shared(globals_.embedding_weights,
                                                   globals_.embedded_weights_literal)
            self.classifier_weights = theano.shared(globals_.classifier_weights,
                                                    globals_.classifier_weights_literal)
            self.classifier_bias = theano.shared(globals_.classifier_bias, globals_.classifier_bias_literal)

    class LSTM(object):
        def __init__(self, lstm_inputs):
            """
            :type lstm_inputs: Model.LSTM
            :param lstm_inputs: non-theano LSTM input layer (using numpy arrays)
            """
            self.input_weights = theano.shared(lstm_inputs.input_weights, lstm_inputs.input_weights_literal)
            self.hidden_weights = theano.shared(lstm_inputs.hidden_weights, lstm_inputs.hidden_weights_literal)
            self.bias = theano.shared(lstm_inputs.bias, lstm_inputs.input_weights_literal)

    def __init__(self, model):
        """
        :type model: Model
        :param model: source LSTM parameters (using numpy arrays)
        """
        self.feature_count = model.feature_count
        self.hidden_unit_count = model.hidden_unit_count
        self.category_count = model.category_count
        self.globals = TheanoModel.Globals(model.globals)
        self.lstm = TheanoModel.LSTM(model.lstm)
        self.parameter_dict = {}
        self.parameter_dict.update(self.globals.__dict__)
        self.parameter_dict.update(self.lstm.__dict__)

    def values(self):
        return self.parameter_dict.values()

    def items(self):
        return self.parameter_dict.items()

