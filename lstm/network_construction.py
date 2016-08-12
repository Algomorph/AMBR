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
from lstm.model import TheanoModel
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
    # theano
    import theano
    from theano import config, tensor
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


def numpy_float_x(data):
    return np.asarray(data, dtype=config.floatX)


def build_lstm_layer(model, lstm_input, masks=None):
    """
    :param lstm_input: input batch
    :type model: TheanoModel
    :param model: embedding & lstm input layers
    :param masks: masks that cancel out the effect of the "empty" tail of each sequence (due to samples being of different lengths)
    :return:
    """
    n_frames_in_sample = lstm_input.shape[0]
    if lstm_input.ndim == 3:
        n_samples_in_batch = lstm_input.shape[1]
    else:
        n_samples_in_batch = 1

    lstm_input_weighted = (
        tensor.dot(lstm_input, model.lstm.input_weights)
        + model.lstm.bias)

    h_u_c = model.hidden_unit_count

    assert masks is not None

    def quarter_slice(_x, ix_quarter):
        if _x.ndim == 3:
            return _x[:, :, ix_quarter * h_u_c:(ix_quarter + 1) * h_u_c]
        return _x[:, ix_quarter * h_u_c:(ix_quarter + 1) * h_u_c]

    def lstm_step(input_mask, layer_input, previous_hidden_unit, previous_cell, previous_input_gate,
                  previous_forget_gate, previous_output_gate):
        network_after_update = tensor.dot(previous_hidden_unit, model.lstm.hidden_weights)
        network_after_update += layer_input

        # squash inputs from embedding layer
        new_input_gate = tensor.nnet.sigmoid(quarter_slice(network_after_update, 0))
        new_forget_gate = tensor.nnet.sigmoid(quarter_slice(network_after_update, 1))
        new_output_gate = tensor.nnet.sigmoid(quarter_slice(network_after_update, 2))
        new_cell = tensor.tanh(quarter_slice(network_after_update, 3))

        # new forget gate state tells us how much influence to recall (let through) from previous cell state,
        # new input gate sate tells us how much influence to propagate (let through) from the new cell state
        new_cell = new_forget_gate * previous_cell + new_input_gate * new_cell
        new_cell = input_mask[:, None] * new_cell + (1. - input_mask)[:, None] * previous_cell

        new_hidden_unit = new_output_gate * tensor.tanh(new_cell)
        new_hidden_unit = input_mask[:, None] * new_hidden_unit + (1. - input_mask)[:,
                                                                  None] * previous_hidden_unit

        return new_hidden_unit, new_cell, new_input_gate, new_forget_gate, new_output_gate

    init_hidden = tensor.alloc(numpy_float_x(0.), n_samples_in_batch, model.hidden_unit_count)
    init_cell = tensor.alloc(numpy_float_x(0.), n_samples_in_batch, model.hidden_unit_count)
    init_input = tensor.alloc(numpy_float_x(0.), n_samples_in_batch, model.hidden_unit_count)
    init_forget = tensor.alloc(numpy_float_x(0.), n_samples_in_batch, model.hidden_unit_count)
    init_output = tensor.alloc(numpy_float_x(0.), n_samples_in_batch, model.hidden_unit_count)

    (hidden_unit, cell, input_gate, forget_gate, output_gate), updates = theano.scan(lstm_step,
                                                                                     sequences=[masks,
                                                                                                lstm_input_weighted],
                                                                                     # inputs
                                                                                     outputs_info=[init_hidden,
                                                                                                   init_cell,
                                                                                                   init_input,
                                                                                                   init_forget,
                                                                                                   init_output],
                                                                                     name='_layers',
                                                                                     n_steps=n_frames_in_sample)

    return hidden_unit, cell, input_gate, forget_gate, output_gate


def build_dropout_layer(input_projection, noise_bool_flag, random_seed):
    """
    A way to prevent overfitting by introducing random noise to hidden units.
    Reference:
    Srivastava, Nitish, Geoffrey E. Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov.
    "Dropout: a simple way to prevent neural networks from overfitting." Journal of Machine Learning Research 15, no. 1
    (2014): 1929-1958.
    :param input_projection: state of the hidden units
    :param noise_bool_flag: input flag that specifies whether noise should be used
    :param random_seed: random seed to use for introducing noise in the hidden units
    :return: theano tensor for projection after introduction of dropout
    """
    output_projection = tensor.switch(noise_bool_flag, (input_projection *
                                                        RandomStreams(random_seed).binomial(
                                                            input_projection.shape, p=0.5, n=1,
                                                            dtype=input_projection.dtype)),
                                      input_projection * 0.5)
    return output_projection


def build_network(model, use_dropout=True, weighted_cost=False, random_seed=2016):
    """
    :type weighted_cost: bool
    :param weighted_cost: whether to weigh the output costs
    :param random_seed: random seed to use for dropout noise
    :see build_dropout_layer
    :type model: TheanoModel
    :param model: parameters of the LSTM network
    :param use_dropout: whether or not to use dropout
    :return: all network inputs and tensor functions
    """
    # x = batch of samples, each with a vector of features for each timestep
    # y = batch of labels corresponding to samples in x
    # mask = batch of masks corresponding to samples in x,
    # where each sample's mask is a row of ones and zeros

    # size of x:
    # [n_timesteps_in_sample * n_samples_in_batch * n_features]
    x = tensor.tensor3('x', dtype=config.floatX)
    # size of mask:
    # [n_timesteps_in_sample * n_samples_in_batch]
    y = tensor.vector('y', dtype='int64')
    masks = tensor.matrix('mask', dtype=config.floatX)

    n_timesteps_in_sample = x.shape[0]
    n_samples_in_batch = x.shape[1]

    embedding_output = theano.dot(x, model.globals.embedding_weights)

    timestep_projections_unmasked, cell_state, input_gate_sate, forget_gate_state, output_gate_state \
        = build_lstm_layer(model, embedding_output, masks=masks)

    # mean pooling
    # weight_gradient = [index of time step] / [total number of unmasked time steps]
    weight_gradient = tensor.arange(n_timesteps_in_sample).astype(config.floatX)
    weight_gradient = weight_gradient[:, None] / masks.sum(axis=0)

    # apply mask to results
    timestep_projections_masked = timestep_projections_unmasked * masks[:, :, None]

    # apply linear weight gradient
    # latter time steps influence the overall sample decision more heavily?
    timestep_projections_weighted = timestep_projections_masked * weight_gradient[:, :, None]
    sample_projection = timestep_projections_weighted.sum(axis=0)
    sample_projection = sample_projection / masks.sum(axis=0)[:, None]

    if use_dropout:
        noise_bool_flag = theano.shared(numpy_float_x(0.))
        sample_projection = build_dropout_layer(sample_projection, noise_bool_flag, random_seed)

    # formerly "pred"
    sample_prediction = tensor.nnet.softmax(
        tensor.dot(sample_projection,
                   model.globals.classifier_weights) + model.globals.classifier_bias)

    # formerly "f_pred_prob" and "f_pred"
    sample_prediction_probability_function = theano.function([x, masks], sample_prediction,
                                                             name='prediction_probability_function')
    sample_prediction_function = theano.function([x, masks], sample_prediction.argmax(axis=1),
                                                 name='prediction_function')

    # formerly "out_proj_all"
    timestep_embedding_outputs = \
        tensor.dot(timestep_projections_masked, model.globals.classifier_weights) + \
        model.globals.classifier_bias

    # formerly "pred_all"
    timestep_predictions, updates = theano.scan(lambda proj: tensor.nnet.softmax(proj),
                                                sequences=[timestep_embedding_outputs],
                                                non_sequences=None,
                                                n_steps=n_timesteps_in_sample)

    # formerly "f_pred_prob_all"
    timestep_prediction_function = theano.function([x, masks], timestep_predictions, name='f_pred_prob_all')

    # hidden_all
    network_state = [timestep_projections_unmasked,
                     cell_state,
                     input_gate_sate,
                     forget_gate_state,
                     output_gate_state,
                     model.lstm.input_weights,
                     model.lstm.hidden_weights,
                     model.lstm.bias,
                     model.globals.classifier_weights,
                     model.globals.classifier_bias,
                     model.globals.embedding_weights]  # 10 in total

    network_state_function = theano.function([x, masks], network_state, name='hidden_status')

    off = 1e-8
    if sample_prediction.dtype == 'float16':
        off = 1e-6

    if weighted_cost:
        cost_weights = tensor.vector('cost_weights', dtype=config.floatX)
        cost = -tensor.log(
            tensor.dot(sample_prediction[tensor.arange(n_samples_in_batch), y], cost_weights) + off).mean()
    else:
        cost_weights = None
        cost = -tensor.log(sample_prediction[tensor.arange(n_samples_in_batch), y] + off).mean()

    return noise_bool_flag, x, masks, cost_weights, y, sample_prediction_probability_function, \
           sample_prediction_function, cost, timestep_prediction_function, network_state_function
