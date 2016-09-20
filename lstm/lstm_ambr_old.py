#!/usr/bin/python3
"""
Build a tweet sentiment analyzer
"""
# stdlib
from collections import OrderedDict
import pickle as pkl
import os
import sys
import time

# scipy stack
import numpy as np
import scipy.io as sio

# theano
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

# charts
from matplotlib import pyplot as pylab
import matplotlib as mpl

# local
from lstm.optimizer import adadelta, sgd, rmsprop
from lstm.data_io import load_data, prepare_data
from lstm.arguments import Arguments
from ext_argparse.argproc import process_arguments

mpl.rcParams['image.interpolation'] = 'nearest'

# Set the random number generators' seeds for consistency
# SEED = 123
SEED = 2016
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

    return list(zip(list(range(len(minibatches))), minibatches))


def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.items():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.items():
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


def init_params(args):
    """
    Global (not LSTM) parameters. For the embedding and the classifier.
    """
    params = OrderedDict()
    # embedding
    randn = np.random.rand(args.feature_count,
                           args.hidden_unit_count)
    params['Wemb'] = (0.01 * randn).astype(config.floatX)
    params = param_init_lstm(args, params, prefix=args.encoder)
    # classifier
    params['U'] = 0.01 * np.random.randn(args.hidden_unit_count,
                                         args.category_count).astype(config.floatX)
    params['b'] = np.zeros((args.category_count,)).astype(config.floatX)

    return params


def load_params(path, params):
    pp = np.load(path)
    for kk, vv in params.items():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(config.floatX)


def param_init_lstm(args, params, prefix='lstm'):
    """
    Init the LSTM parameter:

    :see: init_params
    """
    W = np.concatenate([ortho_weight(args.hidden_unit_count),
                        ortho_weight(args.hidden_unit_count),
                        ortho_weight(args.hidden_unit_count),
                        ortho_weight(args.hidden_unit_count)], axis=1)
    params[_p(prefix, 'W')] = W
    U = np.concatenate([ortho_weight(args.hidden_unit_count),
                        ortho_weight(args.hidden_unit_count),
                        ortho_weight(args.hidden_unit_count),
                        ortho_weight(args.hidden_unit_count)], axis=1)
    params[_p(prefix, 'U')] = U
    b = np.zeros((4 * args.hidden_unit_count,))
    params[_p(prefix, 'b')] = b.astype(config.floatX)

    return params


def generate_lstm_layer(tparams, embedding_layer, args, prefix='lstm', mask=None,
                        init_h=None, init_c=None):
    n_frames_in_sample = embedding_layer.shape[0]
    if embedding_layer.ndim == 3:
        n_samples = embedding_layer.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_, c_, i_, f_, o_):
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_

        input = tensor.nnet.sigmoid(_slice(preact, 0, args.hidden_unit_count))
        forget = tensor.nnet.sigmoid(_slice(preact, 1, args.hidden_unit_count))
        output = tensor.nnet.sigmoid(_slice(preact, 2, args.hidden_unit_count))
        cell = tensor.tanh(_slice(preact, 3, args.hidden_unit_count))

        cell = forget * c_ + input * cell
        cell = m_[:, None] * cell + (1. - m_)[:, None] * c_

        hidden = output * tensor.tanh(cell)
        hidden = m_[:, None] * hidden + (1. - m_)[:, None] * h_

        return hidden, cell, input, forget, output

    embedding_layer = (tensor.dot(embedding_layer, tparams[_p(prefix, 'W')]) +
                       tparams[_p(prefix, 'b')])

    dim_proj = args.hidden_unit_count

    if not init_h:
        init_h = tensor.alloc(numpy_floatX(0.), n_samples, dim_proj)
        init_c = tensor.alloc(numpy_floatX(0.), n_samples, dim_proj)
        init_i = tensor.alloc(numpy_floatX(0.), n_samples, dim_proj)
        init_f = tensor.alloc(numpy_floatX(0.), n_samples, dim_proj)
        init_o = tensor.alloc(numpy_floatX(0.), n_samples, dim_proj)

    rval, updates = theano.scan(_step,
                                sequences=[mask, embedding_layer],
                                outputs_info=[init_h, init_c, init_i, init_f, init_o],
                                name=_p(prefix, '_layers'),
                                n_steps=n_frames_in_sample)

    return rval


def build_model(tparams, args):
    trng = RandomStreams(SEED)

    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))

    x = tensor.tensor3('x', dtype=config.floatX)
    mask = tensor.matrix('mask', dtype=config.floatX)
    y = tensor.vector('y', dtype='int64')
    w = tensor.vector('w', dtype=config.floatX)

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    emb = theano.dot(x, tparams['Wemb'])
    rval = generate_lstm_layer(tparams, emb, args,
                               prefix=args.encoder,
                               mask=mask)

    proj_all_raw, c, i, f, o = rval

    if args.encoder == 'lstm':
        # mean pooling
        wg = tensor.arange(n_timesteps).astype(config.floatX)
        wg = wg[:, None] / mask.sum(axis=0)
        proj_all_raw = proj_all_raw * mask[:, :, None]
        proj_all = proj_all_raw * wg[:, :, None]
        proj = proj_all.sum(axis=0)
        proj = proj / mask.sum(axis=0)[:, None]
    if args.use_dropout:
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
                                    n_steps=n_timesteps)

    f_pred_prob_all = theano.function([x, mask], pred_all, name='f_pred_prob_all')

    hidden_all = [proj_all_raw, c, i, f, o,
                  tparams[_p(args.encoder, 'W')],
                  tparams[_p(args.encoder, 'U')],
                  tparams[_p(args.encoder, 'b')],
                  tparams['U'], tparams['b'], tparams['Wemb']]  # 10 in total

    hidden_status = theano.function([x, mask], hidden_all, name='hidden_status')

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
        # x = data[0][0][:,None,:].astype('float32')
        # mask = np.ones((x.shape[0], 1), dtype='float32')
        # hs = hs_func(x, mask)
        # pylab.clf()
        # pylab.imshow(np.squeeze(hs[:,0,:]))
        # pylab.colorbar()

        x = data[0][0][:, None, :].astype('float32')
        mask = np.ones((x.shape[0], 1), dtype='float32')
        # h, c, i, f, o
        hs = hs_func(x, mask)
        # hs_all = np.concatenate(hspylab.clf(), axis=2)
        pylab.figure(1)
        pylab.clf()
        for s in range(5):
            pylab.subplot(1, 5, s + 1)
            pylab.imshow(np.squeeze(hs[s][:, 0, :]), interpolation='nearest')
            pylab.colorbar()

        pylab.savefig("hs_test_tmp.png")

        pylab.figure(2);
        pylab.clf()
        pylab.subplot(3, 1, 1)
        pylab.imshow(hs[5], interpolation='nearest')
        pylab.colorbar()
        pylab.title("hs_Wmatrix_lstm")

        pylab.subplot(3, 1, 2)
        pylab.imshow(hs[6], interpolation='nearest')
        pylab.colorbar()
        pylab.title("hs_Umatrix_lstm")

        pylab.subplot(3, 1, 3)
        pylab.imshow(hs[8], interpolation='nearest')
        pylab.colorbar()
        pylab.title("hs_Umatrix")
        pylab.savefig("hs_matrix.png")

        pylab.figure(3);
        pylab.clf()
        pylab.subplot(2, 1, 1)
        pylab.plot(hs[7])
        pylab.title("hs_Bvec_lstm")
        pylab.subplot(2, 1, 2)
        pylab.plot(hs[9])
        pylab.title("hs_Bvec")
        pylab.savefig("hs_vector.png")

        time.sleep(0.1)

    return valid_err


def pred_avg_PrRc(f_pred_prob, prepare_data, data, iterator, category_count, verbose=False):
    n_samples = len(data[0])
    probabilities = np.zeros((n_samples, category_count)).astype(config.floatX)
    gts = np.zeros((n_samples,)).astype('int32')

    n_done = 0

    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  np.array(data[1])[valid_index])
        pred_probs = f_pred_prob(x, mask)
        probabilities[valid_index, :] = pred_probs
        gts[valid_index] = np.array(data[1])[valid_index]

        n_done += len(valid_index)

    preds = np.argmax(probabilities, axis=1)
    cm = confusion_matrix(gts, preds, category_count)
    tp = np.diagonal(cm)
    cls_count = np.sum(cm, axis=0)
    fp = np.sum(cm, axis=1) - tp
    fn = cls_count - tp

    prectmp = tp / (tp + fp)
    prectmp[np.where(tp == 0)[0]] = 0
    prectmp[np.where(cls_count == 0)[0]] = float('nan')
    prec = np.nanmean(prectmp)

    rectmp = tp / (tp + fn)
    rectmp[np.where(tp == 0)[0]] = 0
    rectmp[np.where(cls_count == 0)[0]] = float('nan')
    rec = np.nanmean(rectmp)

    return probabilities, gts, prec, rec


def test_lstm(model_file_path, args, result_dir=None):
    args.model_file = model_file_path
    print("model options", args)
    print('Loading test data')
    train, valid, test, n_categories = load_data()

    print("%d train examples" % len(train[0]))
    print("%d valid examples" % len(valid[0]))
    print("%d test examples" % len(test[0]))

    if not result_dir:
        result_dir = 'test_results9'

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    args.validation_batch_size = 1
    params = init_params(args)
    load_params(model_file_path, params)
    tparams = init_tparams(params)
    (use_noise, x, mask,
     y, f_pred_prob, f_pred, cost, f_pred_prob_all, hidden_status) = build_model(tparams, args)

    kf_test = get_minibatches_idx(len(test[0]), args.validation_batch_size)
    print("%d test examples" % len(test[0]))

    probs, gts, prec, rec = pred_avg_PrRc(f_pred_prob, prepare_data, test, kf_test, args.category_count,
                                          verbose=False)
    preds_all = np.argmax(probs, axis=1)
    cm = confusion_matrix(gts, preds_all, category_count=args.category_count)
    cm = np.asarray(cm, 'float32')
    cm = cm / np.sum(cm, axis=0)
    cm[np.where(np.isnan(cm))] = 0
    f = pylab.figure(2)
    f.clf()
    ax = f.add_axes([0.1, 0.1, 0.8, 0.8])
    im = ax.imshow(cm, interpolation='nearest')
    f.colorbar(im)
    pylab.savefig("%s/confusion_matrix_sub.png" % result_dir)

    results = {'scores': probs,
               'gts': gts,
               'prec': prec,
               'rec': rec}
    result_file = '%s/%s_result.mat' % (result_dir, model_file_path.split('/')[-1].split('.')[0])
    sio.savemat(result_file, results)

    preds_all = []
    for t in range(len(test[0])):
        x, mask, y = prepare_data([test[0][t]], np.array(test[1])[t])
        preds_all.append(f_pred_prob_all(x, mask))

    results_all = {'preds_all': preds_all,
                   'gts': gts,
                   'start_frame': [d['s_fid'] for d in test[2]],
                   'end_frame': [d['e_fid'] for d in test[2]],
                   'label': [d['label'] for d in test[2]]}

    results_all_file = '%s/%s_result_all.mat' % (result_dir, model_file_path.split('/')[-1].split('.')[0])
    sio.savemat(results_all_file, results_all)
    return


def get_optimizer_constructor(name):
    if name == 'adadelta':
        return adadelta
    elif name == 'sgd':
        return sgd
    elif name == 'rmsprop':
        return rmsprop
    else:
        raise ValueError("Optimizer {:s} not supported")


def train_lstm(model_output_path, args, check_gradients=False):
    args.model_file = model_output_path
    print("model options", args)
    save_interval = args.save_interval
    optimizer = get_optimizer_constructor(args.optimizer)

    print('Loading data')
    train, valid, test, n_categories = load_data(args.datasets, args.folder)
    print("%d train examples" % len(train[0]))
    print("%d valid examples" % len(valid[0]))
    print("%d test examples" % len(test[0]))

    print('Building model')
    # This create the initial parameters as np ndarrays.
    # Dict name (string) -> np ndarray
    params = init_params(args)

    if args.reload_model:
        load_params(model_output_path, params)

    # This create Theano Shared Variable from the parameters.
    # Dict name (string) -> Theano Tensor Shared Variable
    # params and tparams have different copy of the weights.
    tparams = init_tparams(params)

    # use_noise is for dropout
    (use_noise, x, mask,
     y, f_pred_prob, f_pred, cost, f_pred_prob_all, hidden_status) = build_model(tparams, args)

    if args.decay_c > 0.:
        decay_c = theano.shared(numpy_floatX(args.decay_c), name='decay_c')
        weight_decay = 0.
        weight_decay += (tparams['U'] ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    f_cost = theano.function([x, mask, y], cost, name='f_cost')

    grads = tensor.grad(cost, wrt=list(tparams.values()))
    f_grad = theano.function([x, mask, y], grads, name='f_grad')

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, tparams, grads,
                                        x, mask, y, cost)

    print('Optimization')

    kf_valid = get_minibatches_idx(len(valid[0]), args.validation_batch_size)
    kf_test = get_minibatches_idx(len(test[0]), args.validation_batch_size)

    history_errs = []
    eidx_a = []
    best_p = None

    if save_interval == -1:
        save_interval = len(train[0]) / args.batch_size

    uidx = 0  # the number of update done
    estop = False  # early stop
    start_time = time.time()
    try:
        for eidx in range(args.max_epochs):
            n_samples = 0

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(len(train[0]), args.batch_size, shuffle=True)

            for _, train_index in kf:
                uidx += 1
                use_noise.set_value(1.)

                # Select the random examples for this minibatch
                y = [train[1][t] for t in train_index]
                x = [train[0][t] for t in train_index]
                w = [train[3][t] for t in train_index]

                # Get the data in np.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)
                x, mask, y = prepare_data(x, y)
                n_samples += x.shape[1]

                # # Check gradients
                if check_gradients:
                    grads = f_grad(x, mask, y)
                    grads_value = grad_array(grads)
                    print('gradients :', [np.mean(g) for g in grads_value])
                    params = unzip(tparams)
                    print('parameter :', [np.mean(vv) for kk, vv in params.iteritems()])

                cost = f_grad_shared(x, mask, y)
                f_update(args.learning_rate)

                if np.isnan(cost) or np.isinf(cost):
                    print('NaN detected')
                    return 1., 1., 1.

                if np.mod(uidx, args.display_interval) == 0:
                    print('Epoch ', eidx, 'Update ', uidx, 'Cost ', cost)

                if model_output_path and uidx % save_interval == 0:
                    print('Saving...', end=' ')

                    if best_p is not None:
                        params = best_p
                    else:
                        params = unzip(tparams)
                    np.savez(model_output_path, history_errs=history_errs, **params)
                    pkl.dump(args.__dict__, open('%s.pkl' % model_output_path, 'wb'), -1)
                    print('Done')

                if uidx % args.validation_interval == 0:
                    use_noise.set_value(0.)
                    train_err = pred_error(f_pred, prepare_data, train, kf)
                    valid_err = pred_error(f_pred, prepare_data, valid, kf_valid)
                    # test_err = pred_error(f_pred, prepare_data, test, kf_test, show_hs=True, hs_func=hidden_status)
                    test_err = pred_error(f_pred, prepare_data, test, kf_test)

                    history_errs.append([train_err, valid_err, test_err])
                    eidx_a.append([eidx, eidx, eidx])

                    # import pdb; pdb.set_trace()

                    pylab.figure(1);
                    pylab.clf()
                    lines = pylab.plot(np.array(eidx_a), np.array(history_errs))
                    pylab.legend(lines, ['train', 'valid', 'test'])
                    pylab.savefig("err.png")
                    time.sleep(0.1)

                    if (uidx == 0 or
                                valid_err <= np.array(history_errs)[:, 1].min()):

                        best_p = unzip(tparams)
                        bad_counter = 0
                        if valid_err < np.array(history_errs)[:, 1].min():
                            print('  New best validation results.')

                    print('TrainErr=%.06f  ValidErr=%.06f  TestErr=%.06f' % (train_err, valid_err, test_err))

                    if (len(history_errs) > args.patience and
                                valid_err >= np.array(history_errs)[:-args.patience,
                                             1].min()):
                        bad_counter += 1
                        if bad_counter > args.patience:
                            print('Early Stop!')
                            estop = True
                            break

            print('Seen %d samples' % n_samples)

            if estop:
                break

    except KeyboardInterrupt:
        print("Training interupted")

    end_time = time.time()
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)

    use_noise.set_value(0.)
    kf_train_sorted = get_minibatches_idx(len(train[0]), args.batch_size)
    train_err = pred_error(f_pred, prepare_data, train, kf_train_sorted)
    valid_err = pred_error(f_pred, prepare_data, valid, kf_valid)
    test_err = pred_error(f_pred, prepare_data, test, kf_test)

    print('TrainErr=%.06f  ValidErr=%.06f  TestErr=%.06f' % (train_err, valid_err, test_err))
    if model_output_path:
        np.savez(model_output_path, train_err=train_err,
                 valid_err=valid_err, test_err=test_err,
                 history_errs=history_errs, **best_p)
    print('The code run for %d epochs, with %f sec/epochs' % (
        (eidx + 1), (end_time - start_time) / (1. * (eidx + 1))))
    print(('Training took %.1fs' %
           (end_time - start_time)), file=sys.stderr)
    return train_err, valid_err, test_err


def confusion_matrix(gt, pred, category_count):
    cm = np.zeros((category_count, category_count))
    for i in range(category_count):
        idx_category = np.where(gt == i)[0]
        if idx_category.size == 0:
            continue
        predicted_category = pred[idx_category]
        for j in range(category_count):
            cm[j, i] = np.where(predicted_category == j)[0].shape[0]

    return cm


def test_confusion_matrix():
    from random import randint
    nCls = 10
    gt = np.tile(np.arange(nCls), (7, 1))
    gt = np.reshape(gt.T, (gt.size,))
    pred = np.asarray([randint(0, nCls - 1) for x in range(gt.size)])
    cm = confusion_matrix(gt, pred, nCls)

    # print cm
    print(np.sum(cm, axis=0))
    print(np.sum(cm, axis=1))
    return cm


def main():
    args = process_arguments(Arguments, "Train & test an LSTM model on the given input.")

    model_subdir = 'monocular_model'
    model_dir = os.path.join(args.folder, model_subdir)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    model_path = os.path.join(model_dir, 'lstm_model.npz')

    if args.overwrite_model or not os.path.isfile(model_path):
        train_lstm(model_path, args)

    # testing
    test_lstm(model_path, args)
    return 0


if __name__ == '__main__':
    pylab.ion()
    sys.exit(main())
