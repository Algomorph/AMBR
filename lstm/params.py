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


class Parameters(object):
    class Globals(object):
        embedding_weights_literal = "embedding_layer_weights"
        classifier_weights_literal = "classifier_weights"
        classifier_bias_literal = "classifier_bias"

        def __init__(self, feature_count=None, hidden_unit_count=None, category_count=None, archive=None):
            if archive is None and (feature_count is None or hidden_unit_count is None or category_count is None):
                raise ValueError(
                    "If archive is not passed in, an " + Parameters.Globals.__name__ +
                    " object needs all other constructor arguments to be integers.")
            if archive is None:
                self.embedding_weights = theano.shared(
                    (0.01 * np.random.rand(feature_count, hidden_unit_count)).astype(config.floatX),  # formerly Wemb
                    self.embedding_weights_literal)
                self.classifier_weights = theano.shared(
                    0.01 * np.random.randn(hidden_unit_count, category_count).astype(config.floatX),
                    self.classifier_weights_literal)  # formerly U
                self.classifier_bias = theano.shared(np.zeros((category_count,)).astype(config.floatX),
                                                     self.classifier_bias_literal)  # formerly b
            else:
                self.read_values_from_dict(archive)

        def read_values_from_dict(self, archive):
            self.embedding_weights = theano.shared(archive[self.embedding_weights_literal],
                                                   self.embedding_weights_literal)
            self.classifier_weights = theano.shared(archive[self.classifier_weights_literal],
                                                    self.classifier_weights_literal)
            self.classifier_bias = theano.shared(archive[self.classifier_bias_literal], self.classifier_bias_literal)

    class LSTM(object):
        input_weights_literal = "lstm_input_weights"
        hidden_weights_literal = "lstm_hidden_weights"
        bias_literal = "lstm_bias"

        def __init__(self, hidden_unit_count=None, archive=None):
            if archive is None and hidden_unit_count is None:
                raise ValueError(
                    "If archive is not passed in, an " + Parameters.LSTM.__name__ +
                    " object needs hidden_unit_count argument to be an integer.")
            if archive is None:
                gen__r_o_v = generate_random_orthogonal_vectors
                self.input_weights = theano.shared(np.concatenate([gen__r_o_v(hidden_unit_count),
                                                                   gen__r_o_v(hidden_unit_count),
                                                                   gen__r_o_v(hidden_unit_count),
                                                                   gen__r_o_v(hidden_unit_count)], axis=1),
                                                   self.input_weights_literal)  # formerly lstm_W
                self.hidden_weights = theano.shared(np.concatenate([gen__r_o_v(hidden_unit_count),
                                                                    gen__r_o_v(hidden_unit_count),
                                                                    gen__r_o_v(hidden_unit_count),
                                                                    gen__r_o_v(hidden_unit_count)], axis=1),
                                                    self.hidden_weights_literal)  # formerly lstm_U

                self.bias = theano.shared(np.zeros((4 * hidden_unit_count,)).astype(config.floatX),
                                          self.bias_literal)  # formerly lstm_b
            else:
                self.load_values_from_dict(archive)

        def load_values_from_dict(self, archive):
            self.input_weights = theano.shared(archive[self.input_weights_literal], self.input_weights_literal)
            self.hidden_weights = theano.shared(archive[self.hidden_weights_literal], self.hidden_weights_literal)
            self.bias = theano.shared(archive[self.bias_literal], self.bias_literal)

    def __init__(self, feature_count=None, hidden_unit_count=None, category_count=None, archive=None):
        if archive is None and (feature_count is None or hidden_unit_count is None or category_count is None):
            raise ValueError(
                "If archive is not passed in, an " + Parameters.__name__ +
                " object needs all other constructor arguments to be integers.")
        if archive is None:
            self.feature_count = feature_count
            self.hidden_unit_count = hidden_unit_count
            self.category_count = category_count
            self.globals = Parameters.Globals(feature_count, hidden_unit_count, category_count)
            self.lstm = Parameters.LSTM(hidden_unit_count)
        else:
            self.globals = Parameters.Globals(archive=archive)
            self.lstm = Parameters.Globals(archive=archive)
            # infer basic settings from globals
            self.hidden_unit_count = self.globals.classifier_bias.shape[0]
            self.feature_count = self.globals.embedding_weights.shape[0]
            self.category_count = self.globals.classifier_weights.shape[1]
        self.parameter_dict = {}
        self.__update_dict()

    def __update_dict(self):
        self.parameter_dict.update(self.globals.__dict__)
        self.parameter_dict.update(self.lstm.__dict__)

    def write_to_dict(self, archive_dict):
        for key, value in self.parameter_dict.items():
            archive_dict[key] = value.get_value()

    def as_dict(self):
        archive_dict = {}
        self.write_to_dict(archive_dict)
        return archive_dict

    def read_from_dict(self, archive_dict):
        self.globals.read_values_from_dict(archive_dict)
        self.lstm.read_values_from_dict(archive_dict)
        self.__update_dict()

    def save_to_numpy_archive(self, path):
        archive_dict = {}
        self.write_to_dict(archive_dict)
        np.savez(path, **archive_dict)

    def items(self):
        return self.parameter_dict.items()

    def values(self):
        return self.parameter_dict.values()
