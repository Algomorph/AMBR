#  ================================================================
#  Created by Gregory Kramida on 8/9/16.
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
from theano import config, tensor


# TODO: revise docstrings

def rmsprop(learning_rate, parameters, grads, x, mask, y, cost, w=None):
    """
    A variant of  SGD that scales the step size by running average of the
    recent step norms.

    Parameters
    ----------
    learning_rate : Theano SharedVariable
        Initial learning rate
    :type parameters: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [Hint2014]_.

    .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
       lecture 6a,
       http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """

    zipped_grads = [theano.shared(p.get_value() * np.asarray(0., dtype=config.floatX),
                                  name='%s_grad' % k)
                    for k, p in parameters.items()]
    running_grads = [theano.shared(p.get_value() * np.asarray(0., dtype=config.floatX),
                                   name='%s_rgrad' % k)
                     for k, p in parameters.items()]
    running_grads2 = [theano.shared(p.get_value() * np.asarray(0., dtype=config.floatX),
                                    name='%s_rgrad2' % k)
                      for k, p in parameters.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]
    if w is not None:
        inputs = [x, mask, y, w]
    else:
        inputs = [x, mask, y]
    f_grad_shared = theano.function(inputs, cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * np.asarray(0., dtype=config.floatX),
                           name='%s_updir' % k)
             for k, p in parameters.items()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(list(parameters.values()), updir_new)]
    f_update = theano.function([learning_rate], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update


def sgd(learning_rate, model, grads, x, mask, y, cost, w=None):
    """ Stochastic Gradient Descent

    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.

    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in model.items()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    if w is not None:
        inputs = [x, mask, y, w]
    else:
        inputs = [x, mask, y]
    f_grad_shared = theano.function(inputs, cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - learning_rate * g) for p, g in zip(list(model.values()), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([learning_rate], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update


def adadelta(learning_rate, model, gradients, x, mask, y, cost, w=None):
    """
    An adaptive learning rate optimizer.
    For more information, see [ADADELTA]_.

    [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
    Rate Method*, arXiv:1212.5701.
    :type learning_rate: theano.tensor.sharedvar.TensorSharedVariable
    :param learning_rate: Initial learning rate
    :type model: lstm.params.Parameters
    :param model: Model parameters
    :param gradients:  Gradients of cost w.r.t to parameters
    :type x: theano.tensor.sharedvar.TensorSharedVariable
    :param x: Model inputs / samples [sequences of feature vectors] or batches of such samples
    :type mask: theano.tensor.sharedvar.TensorSharedVariable
    :param mask: Masks for samples/batches (x)
    :type y: theano.tensor.sharedvar.TensorSharedVariable
    :param y: Targets / Labels
    :param cost: objective function to minimize
    :type w: theano.tensor.sharedvar.TensorSharedVariable
    :param w: per-target weights (optional)
    :return:
    """
    zipped_grads = [theano.shared(parameter.get_value() * np.asarray(0., dtype=config.floatX),
                                  name='%s_grad' % name)
                    for name, parameter in model.items()]
    running_updates2 = [theano.shared(parameter.get_value() * np.asarray(0., dtype=config.floatX),
                                 name='%s_rup2' % name)
                   for name, parameter in model.items()]
    running_grads2 = [theano.shared(parameter.get_value() * np.asarray(0., dtype=config.floatX),
                                    name='%s_rgrad2' % name)
                      for name, parameter in model.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, gradients)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, gradients)]

    if w is not None:
        inputs = [x, mask, y, w]
    else:
        inputs = [x, mask, y]
    f_grad_shared = theano.function(inputs, cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_updates2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_updates2, updir)]
    parameter_updates = [(p, p + ud) for p, ud in zip(list(model.values()), updir)]

    update_weights = theano.function([learning_rate], [], updates=ru2up + parameter_updates,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, update_weights


optimizer_dict = {
    'adadelta': adadelta,
    'sgd': sgd,
    'rmsprop': rmsprop
}


def get_optimizer_names():
    return optimizer_dict.keys()


def get_optimizer_constructor(name):
    if name in optimizer_dict:
        return optimizer_dict[name]
    else:
        raise ValueError("Optimizer {:s} not supported. " +
                         "Supported optimizer names: {:s}".format(str(get_optimizer_names())))
