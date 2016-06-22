'''
Build a tweet sentiment analyzer
'''
from collections import OrderedDict
import cPickle as pkl
import os
import sys
import time

import numpy as np
import scipy.io as sio

import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

# from util import *
import data

import pylab
import matplotlib as mpl
mpl.rcParams['image.interpolation'] = 'nearest'


# Set the random number generators' seeds for consistency
SEED = 123
np.random.seed(SEED)


def numpy_floatX(data):
    return np.asarray(data, dtype=config.floatX)


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params



def grad_array(tgrad):
    return [np.asarray(g) for g in tgrad]

def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.5, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 0.5)
    return proj


def _p(pp, name):
    return '%s_%s' % (pp, name)


def init_params(options):
    """
    Global (not LSTM) parameter. For the embeding and the classifier.
    """
    params = OrderedDict()
    # embedding
    randn = np.random.rand(options['feat_dim'],
                              options['dim_proj'])
    params['Wemb'] = (0.01 * randn).astype(config.floatX)
    params = get_layer(options['encoder'])[0](options,
                                              params,
                                              prefix=options['encoder'])
    # classifier
    params['U'] = 0.01 * np.random.randn(options['dim_proj'],
                                            options['ydim']).astype(config.floatX)
    params['b'] = np.zeros((options['ydim'],)).astype(config.floatX)

    return params


def load_params(path, params):
    pp = np.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def get_layer(name):
    fns = layers[name]
    return fns


def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(config.floatX)


def param_init_lstm(options, params, prefix='lstm'):
    """
    Init the LSTM parameter:

    :see: init_params
    """
    W = np.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'W')] = W
    U = np.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'U')] = U
    b = np.zeros((4 * options['dim_proj'],))
    params[_p(prefix, 'b')] = b.astype(config.floatX)

    return params


def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None, 
               init_h=None, init_c=None):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_, c_):
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
        c = tensor.tanh(_slice(preact, 3, options['dim_proj']))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    dim_proj = options['dim_proj']

    if not init_h:
        init_h = tensor.alloc(numpy_floatX(0.), n_samples, dim_proj)
        init_c = tensor.alloc(numpy_floatX(0.), n_samples, dim_proj)

    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[init_h, init_c],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)

    return rval[0], rval[1]


# ff: Feed Forward (normal neural net), only useful to put after lstm
#     before the classifier.
layers = {'lstm': (param_init_lstm, lstm_layer)}


def sgd(lr, tparams, grads, x, mask, y, cost):
    """ Stochastic Gradient Descent

    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.

    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function([x, mask, y], cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update


def adadelta(lr, tparams, grads, x, mask, y, cost):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
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
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, x, mask, y, cost):
    """
    A variant of  SGD that scales the step size by running average of the
    recent step norms.

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
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

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * numpy_floatX(0.),
                           name='%s_updir' % k)
             for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update


def build_model(tparams, options):
    trng = RandomStreams(SEED)

    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))

    x = tensor.tensor3('x', dtype=config.floatX)
    mask = tensor.matrix('mask', dtype=config.floatX)
    y = tensor.vector('y', dtype='int64')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

#    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps,
#                                                n_samples,
#                                                options['dim_proj']])

#    randn = np.random.rand(options['feat_dim'],
#                              options['dim_proj'])
#    fixed_wemb = (0.01 * randn).astype(config.floatX)
#    fixed_wemb_shared = theano.shared(fixed_wemb, "fixed_wemb")
#    emb = theano.dot(x, fixed_wemb_shared)

    emb = theano.dot(x, tparams['Wemb'])
    proj_all_raw, one_step_func = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix=options['encoder'],
                                            mask=mask)
    if options['encoder'] == 'lstm':
        # mean pooling
        wg = tensor.arange(n_timesteps).astype(config.floatX)
        wg = wg[:, None] / mask.sum(axis=0)
        proj_all_raw = proj_all_raw * mask[:, :, None] 
        proj_all = proj_all_raw * wg[:, :, None]
        proj = proj_all.sum(axis=0)
        proj = proj / mask.sum(axis=0)[:, None]
    if options['use_dropout']:
        proj = dropout_layer(proj, use_noise, trng)

    pred = tensor.nnet.softmax(tensor.dot(proj, tparams['U']) + tparams['b'])

    f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')
    f_pred = theano.function([x, mask], pred.argmax(axis=1), name='f_pred')


    def onestep_softmax(proj):
        return tensor.nnet.softmax(proj)

    out_proj_all = tensor.dot(proj_all, tparams['U']) + tparams['b']
    pred_all, updates = theano.scan(onestep_softmax, 
                                sequences=[out_proj_all],
                                non_sequences=None,
                                n_steps=n_timesteps
                                )
    
    # min_val = out_proj_all.min(axis=2)
    # max_val = out_proj_all.max(axis=2)
    # out_proj_all = (out_proj_all + min_val[:,:,None])/(max_val-min_val)[:,:,None]

    # pred_all = out_proj_all / out_proj_all.sum(axis=2)[:,:,None]

    f_pred_prob_all = theano.function([x, mask], pred_all, name='f_pred_prob_all')

    hidden_status = theano.function([x, mask], proj_all_raw, name='hidden_status')


    off = 1e-8
    if pred.dtype == 'float16':
        off = 1e-6

    cost = -tensor.log(pred[tensor.arange(n_samples), y] + off).mean()

    return use_noise, x, mask, y, f_pred_prob, f_pred, cost, f_pred_prob_all, hidden_status


def pred_error(f_pred, prepare_data, data, iterator, verbose=False, show_hs=False, hs_func=None):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    valid_err = 0
    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  np.array(data[1])[valid_index])
        preds = f_pred(x, mask)
        targets = np.array(data[1])[valid_index]
        valid_err += (preds == targets).sum()

    valid_err = 1. - numpy_floatX(valid_err) / len(data[0])


    if show_hs and not hs_func is None: 
        x = data[0][0][:,None,:].astype('float32')
        mask = np.ones((x.shape[0], 1), dtype='float32')
        hs = hs_func(x, mask)
        pylab.clf()
        pylab.imshow(np.squeeze(hs[:,0,:]))
        pylab.colorbar()
        pylab.savefig("hs_test_tmp.png")
        time.sleep(0.1)


    return valid_err


def pred_avg_PrRc(f_pred_prob, prepare_data, data, iterator, nCls, verbose=False):

    n_samples = len(data[0])
    feat_dim = data[0][0].shape[1]
    probs = np.zeros((n_samples, nCls)).astype(config.floatX)
    gts = np.zeros((n_samples, )).astype('int32')

    n_done = 0

    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  np.array(data[1])[valid_index])
        pred_probs = f_pred_prob(x, mask)
        probs[valid_index, :] = pred_probs
        gts[valid_index] = np.array(data[1])[valid_index]

        n_done += len(valid_index)

    preds = np.argmax(probs, axis=1)
    cm = confusion_matrix(gts, preds, nCls)
    tp = np.diagonal(cm)
    cls_count = np.sum(cm, axis=0)
    fp = np.sum(cm, axis=1)-tp
    fn = cls_count-tp

    prectmp = tp / (tp + fp)
    prectmp[np.where(tp==0)[0]] = 0
    prectmp[np.where(cls_count==0)[0]] = float('nan')
    prec = np.nanmean(prectmp)

    rectmp  = tp / (tp + fn)
    rectmp[np.where(tp==0)[0]] = 0
    rectmp[np.where(cls_count==0)[0]] = float('nan')
    rec = np.nanmean(rectmp)

    return probs, gts, prec, rec


def test_lstm(
    test_subject, 
    model_file,   # the file to save the model
    prepare_data,
    options,
    metadata,
    result_dir=None
):
    print "model options", options
    print 'Loading test data'
    train, valid, test = data.load_data()

    if not result_dir:
        result_dir = 'test_results_%s' % (metadata['suffix'])
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)


    options['valid_batch_size'] = 1
    params = init_params(options)
    load_params(model_file, params)
    tparams = init_tparams(params)
    (use_noise, x, mask,
     y, f_pred_prob, f_pred, cost, f_pred_prob_all, hidden_status) = build_model(tparams, options)

    kf_test = get_minibatches_idx(len(test[0]), options['valid_batch_size'])
    print "%d test examples" % len(test[0])

    test_err = 0
#        for _, test_index in kf_test:
#            x, mask, y = prepare_data([test[0][t] for t in test_index],
#                                  np.array(test[1])[test_index])

    probs, gts, prec, rec = pred_avg_PrRc(f_pred_prob, prepare_data, test, kf_test, options['ydim'], verbose=False)
    preds_all = np.argmax(probs, axis=1)
    cm = confusion_matrix(gts, preds_all, nCls=options['ydim'])
    cm = np.asarray(cm, 'float32')
    cm = cm / np.sum(cm, axis=0)
    cm[np.where(np.isnan(cm))] = 0
    f = pylab.figure(2)
    f.clf()
    ax = f.add_axes([0.1, 0.1, 0.8, 0.8])
    im = ax.imshow(cm, interpolation='nearest')
    f.colorbar(im)
    pylab.savefig("%s/confusion_matrix_sub_%s.png" % (result_dir, test_subject))

    # import pdb; pdb.set_trace()

    results = {'scores': probs, 
               'gts': gts, 
               'prec': prec, 
               'rec': rec}
    result_file = '%s/%s_result.mat' % (result_dir, model_file.split('/')[-1].split('.')[0])
    sio.savemat(result_file, results)


    # # result_dir_stat = '%s/stat_sub%02d' % (result_dir, test_subject)
    # result_dir_figs = '%s/figs_sub_%s' % (result_dir, test_subject)
    # # if not os.path.exists(result_dir_stat):
    #     # os.mkdir(result_dir_stat)
    # if not os.path.exists(result_dir_figs):
    #     os.mkdir(result_dir_figs)

    # for t in xrange(len(test[0])):
    #     x, mask, y = prepare_data([test[0][t]], np.array(test[1])[t])
    #     preds = f_pred_prob_all(x, mask)

    #     fig = np.squeeze(preds[:,0,:])

    #     pylab.figure(1)
    #     pylab.clf()
    #     pylab.imshow(fig); pylab.colorbar()
    #     pylab.savefig("%s/pred_%d.png" % (result_dir_figs, t))
    #     # time.sleep(0.5)

    #     # import pdb; pdb.set_trace()

    return;


def train_lstm(
    model_file,   # the file to save the model
    load_data, 
    prepare_data, 
    options,
    metadata, 
    result_dir=None,
):
    print "model options", options
    saveto = model_file
    saveFreq = options['saveFreq']
    validFreq = options['validFreq']
    optimizer = options['optimizer']

    print 'Loading data'
    train, valid, test = load_data()

    print 'Building model'
    # This create the initial parameters as np ndarrays.
    # Dict name (string) -> np ndarray
    params = init_params(options)

    if options['reload_model']:
        load_params(model_file, params)

    # This create Theano Shared Variable from the parameters.
    # Dict name (string) -> Theano Tensor Shared Variable
    # params and tparams have different copy of the weights.
    tparams = init_tparams(params)

    # use_noise is for dropout
    (use_noise, x, mask,
     y, f_pred_prob, f_pred, cost, f_pred_prob_all, hidden_status) = build_model(tparams, options)

    if options['decay_c'] > 0.:
        decay_c = theano.shared(numpy_floatX(options['decay_c']), name='decay_c')
        weight_decay = 0.
        weight_decay += (tparams['U'] ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    f_cost = theano.function([x, mask, y], cost, name='f_cost')

    grads = tensor.grad(cost, wrt=tparams.values())
    f_grad = theano.function([x, mask, y], grads, name='f_grad')

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, tparams, grads,
                                        x, mask, y, cost)

    print 'Optimization'

    kf_valid = get_minibatches_idx(len(valid[0]), options['valid_batch_size'])
    kf_test = get_minibatches_idx(len(test[0]), options['valid_batch_size'])

    print "%d train examples" % len(train[0])
    print "%d valid examples" % len(valid[0])
    print "%d test examples" % len(test[0])

    history_errs = []
    best_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = len(train[0]) / options['batch_size']
    if saveFreq == -1:
        saveFreq = len(train[0]) / options['batch_size']

    uidx = 0  # the number of update done
    estop = False  # early stop
    start_time = time.time()
    try:
        for eidx in xrange(options['max_epochs']):
            n_samples = 0

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(len(train[0]), options['batch_size'], shuffle=True)

            for _, train_index in kf:
                uidx += 1
                use_noise.set_value(1.)

                # Select the random examples for this minibatch
                y = [train[1][t] for t in train_index]
                x = [train[0][t]for t in train_index]

                # Get the data in np.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)
                x, mask, y = prepare_data(x, y)
                n_samples += x.shape[1]

                # # Check gradients
                # grads = f_grad(x, mask, y)
                # grads_value = grad_array(grads)
                # # import pdb; pdb.set_trace()
                # print 'gradients :', [np.mean(g) for g in grads_value]
                # params = unzip(tparams)
                # print 'parameter :', [np.mean(vv) for kk, vv in params.iteritems()]

                cost = f_grad_shared(x, mask, y)
                f_update(options['lrate'])

                if np.isnan(cost) or np.isinf(cost):
                    print 'NaN detected'
                    return 1., 1., 1.

                if np.mod(uidx, options['dispFreq']) == 0:
                    print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost

                if saveto and np.mod(uidx, saveFreq) == 0:
                    print 'Saving...',

                    if best_p is not None:
                        params = best_p
                    else:
                        params = unzip(tparams)
                    np.savez(saveto, history_errs=history_errs, **params)
                    pkl.dump(options, open('%s.pkl' % saveto, 'wb'), -1)
                    print 'Done'


                if np.mod(uidx, options['validFreq']) == 0:
                    use_noise.set_value(0.)
                    train_err = pred_error(f_pred, prepare_data, train, kf)
                    valid_err = pred_error(f_pred, prepare_data, valid, kf_valid)
                    test_err = pred_error(f_pred, prepare_data, test, kf_test, show_hs=True, hs_func=hidden_status)

                    history_errs.append([train_err, valid_err, test_err])

                    # import pdb; pdb.set_trace()

                    pylab.figure(1); pylab.clf()
                    lines = pylab.plot(np.array(history_errs))
                    pylab.legend(lines, ['train', 'valid', 'test'])
                    pylab.savefig("err_%s.png" % (metadata['suffix']))
                    time.sleep(0.1)


                    if (uidx == 0 or
                        valid_err <= np.array(history_errs)[:,1].min()):

                        best_p = unzip(tparams)
                        bad_counter = 0
                        if valid_err < np.array(history_errs)[:,1].min():
                            print '  New best validation results.'

                    print 'TrainErr=%.06f  ValidErr=%.06f  TestErr=%.06f' % (train_err, valid_err, test_err)

                    if (len(history_errs) > options['patience'] and
                        valid_err >= np.array(history_errs)[:-options['patience'],
                                                               1].min()):
                        bad_counter += 1
                        if bad_counter > options['patience']:
                            print 'Early Stop!'
                            estop = True
                            break


            print 'Seen %d samples' % n_samples

            if estop:
                break

    except KeyboardInterrupt:
        print "Training interupted"

    end_time = time.time()
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)

    use_noise.set_value(0.)
    kf_train_sorted = get_minibatches_idx(len(train[0]), options['batch_size'])
    train_err = pred_error(f_pred, prepare_data, train, kf_train_sorted)
    valid_err = pred_error(f_pred, prepare_data, valid, kf_valid)
    test_err = pred_error(f_pred, prepare_data, test, kf_test)

    print 'TrainErr=%.06f  ValidErr=%.06f  TestErr=%.06f' % (train_err, valid_err, test_err)
    if saveto:
        np.savez(saveto, train_err=train_err,
                    valid_err=valid_err, test_err=test_err,
                    history_errs=history_errs, **best_p)
    print 'The code run for %d epochs, with %f sec/epochs' % (
        (eidx + 1), (end_time - start_time) / (1. * (eidx + 1)))
    print >> sys.stderr, ('Training took %.1fs' %
                          (end_time - start_time))
    return train_err, valid_err, test_err    


def confusion_matrix(gt, pred, nCls):
    cm = np.zeros((nCls, nCls))
    for i in xrange(nCls):
        idxCls = np.where(gt==i)[0]
        if idxCls.size == 0:
            continue
        predCls = pred[idxCls]
        for j in xrange(nCls):
            cm[j, i] = np.where(predCls==j)[0].shape[0]

    return cm

def test_confusion_matrix():
    from random import randint
    nCls = 10
    gt = np.tile(np.arange(nCls), (7, 1))
    gt = np.reshape(gt.T, (gt.size, ))
    pred = np.asarray([randint(0, nCls-1) for x in xrange(gt.size)])
    cm = confusion_matrix(gt, pred, nCls)

    # print cm
    print np.sum(cm, axis=0)
    print np.sum(cm, axis=1)
    return cm


def get_default_options(nCls):
    dim_proj=128  # word embeding dimension and LSTM number of hidden units.
    patience=15  # Number of epoch to wait before early stop if no progress
    max_epochs=300  # The maximum number of epoch to run
    dispFreq=50  # Display to stdout the training progress every N updates
    decay_c=0.  # Weight decay for the classifier applied to the U weights.
    lrate=0.0001  # Learning rate for sgd (not used for adadelta and rmsprop)
    feat_dim=4096  # CNN feature dimension
    optimizer=adadelta  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
    encoder='lstm'  # TODO: can be removed must be lstm.
    validFreq=20  # Compute the validation error after this number of update.
    saveFreq=50  # Save the parameters after every saveFreq updates
    batch_size=10  # The batch size during training.
    valid_batch_size=5  # The batch size used for validation/test set.

    # Parameter for extra option
    noise_std=0.
    use_dropout=True  # if False slightly faster, but worst test error
                       # This frequently need a bigger model.
    reload_model=None  # Path to a saved model we want to start from.
    ydim = nCls  # number of class

    options = locals().copy()
    return options

def action_exp():
    dim_proj = 64
    feat_dim = 4096
    
    model_dir = 'action_model'  # reserved as the first version
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    nCls = 6*5
    options = get_default_options(nCls)
    options['dim_proj'] = dim_proj;
    options['feat_dim'] = feat_dim;

    metadata = {
        'exp_name': 'default',
        'suffix': 'default'
        }

    subject_list = ['and', 'mic', 'fer', 'kos', 'gui']
    subject_list = ['and']

    # training
    for test_subject in subject_list:

        metadata['suffix'] = '%s_%s' % (metadata['exp_name'], test_subject)

        model_file = '%s/lstm_%s_model.npz' % (model_dir, metadata['suffix'])

        data.test_subect_list = [test_subject]

        load_data_func = data.load_data
        prepare_data_func = data.prepare_data

        if not os.path.isfile(model_file):
            train_lstm(model_file, load_data_func, prepare_data_func, options, metadata)

    # testing
    # for test_subject in subject_list:
        test_lstm(test_subject, model_file, prepare_data_func, options, metadata)


if __name__ == '__main__':
    pylab.ion()
    action_exp()

# stdbuf -oL python lstm_exp.py  2>&1 | tee log/train_exp_ts`date +%s`.log
