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


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


# scipy stack
import numpy as np
import scipy.io as sio

# theano
with suppress_stdout():
    import theano
    from theano import config
    import theano.tensor as tensor

# charts
from matplotlib import pyplot as plt
import matplotlib as mpl

# local
from lstm.optimizer import get_optimizer_constructor
from lstm.arguments import Arguments
from lstm.model import Model, TheanoModel
from lstm.network_construction import build_network
from lstm.data_ambr import load_data, prepare_data
from ext_argparse.argproc import process_arguments

mpl.rcParams['image.interpolation'] = 'nearest'


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


def compute_prediction_error(f_pred, data, iterator, show_hs=False, hs_func=None):
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

        x = data[0][0][:, None, :].astype('float32')
        mask = np.ones((x.shape[0], 1), dtype='float32')
        # h, c, i, f, o
        hs = hs_func(x, mask)
        plt.figure(1)
        plt.clf()
        for s in range(5):
            plt.subplot(1, 5, s + 1)
            plt.imshow(np.squeeze(hs[s][:, 0, :]), interpolation='nearest')
            plt.colorbar()

        plt.savefig("hs_test_tmp.png")

        plt.figure(2)
        plt.clf()
        plt.subplot(3, 1, 1)
        plt.imshow(hs[5], interpolation='nearest')
        plt.colorbar()
        plt.title("hs_Wmatrix_lstm")

        plt.subplot(3, 1, 2)
        plt.imshow(hs[6], interpolation='nearest')
        plt.colorbar()
        plt.title("hs_Umatrix_lstm")

        plt.subplot(3, 1, 3)
        plt.imshow(hs[8], interpolation='nearest')
        plt.colorbar()
        plt.title("hs_Umatrix")
        plt.savefig("hs_matrix.png")

        plt.figure(3)
        plt.clf()
        plt.subplot(2, 1, 1)
        plt.plot(hs[7])
        plt.title("hs_Bvec_lstm")
        plt.subplot(2, 1, 2)
        plt.plot(hs[9])
        plt.title("hs_Bvec")
        plt.savefig("hs_vector.png")

        time.sleep(0.1)

    return valid_err


def pred_avg_PrRc(f_pred_prob, data, iterator, category_count, verbose=False):
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


def test_lstm(model_output_path, args, result_dir=None):
    args.model_file = model_output_path
    print("model options", args)
    print('Loading test data')
    train, valid, test, n_categories = load_data(args.datasets, args.folder)

    print("%d train examples" % len(train[0]))
    print("%d valid examples" % len(valid[0]))
    print("%d test examples" % len(test[0]))

    if not result_dir:
        result_dir = 'test_results9'

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    args.validation_batch_size = 1
    non_theano_model = Model(archive=model_output_path)
    model = TheanoModel(non_theano_model)
    (use_noise, x, mask, w, y, f_pred_prob,
     f_pred, cost, f_pred_prob_all, hidden_status) = build_network(model,
                                                                   hidden_unit_count=args.hidden_unit_count)

    kf_test = get_minibatches_idx(len(test[0]), args.validation_batch_size)
    print("%d test examples" % len(test[0]))

    probs, gts, prec, rec = pred_avg_PrRc(f_pred_prob, test, kf_test, args.category_count,
                                          verbose=False)
    preds_all = np.argmax(probs, axis=1)
    cm = confusion_matrix(gts, preds_all, category_count=args.category_count)
    cm = np.asarray(cm, 'float32')
    cm = cm / np.sum(cm, axis=0)
    cm[np.where(np.isnan(cm))] = 0
    f = plt.figure(2)
    f.clf()
    ax = f.add_axes([0.1, 0.1, 0.8, 0.8])
    im = ax.imshow(cm, interpolation='nearest')
    f.colorbar(im)
    plt.savefig("%s/confusion_matrix_sub.png" % result_dir)

    results = {'scores': probs,
               'gts': gts,
               'prec': prec,
               'rec': rec}
    result_file = '%s/%s_result.mat' % (result_dir, model_output_path.split('/')[-1].split('.')[0])
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

    results_all_file = '%s/%s_result_all.mat' % (result_dir, model_output_path.split('/')[-1].split('.')[0])
    sio.savemat(results_all_file, results_all)
    return


def train_lstm(model_output_path, args, check_gradients=False):
    random_seed = 2016
    np.random.seed(random_seed)
    args.model_file = model_output_path
    print("model options", args)
    save_interval = args.save_interval

    build_optimizer = get_optimizer_constructor(args.optimizer)

    print('Loading data')
    train, valid, test, n_categories = load_data(args.datasets, args.folder)
    print("%d train examples" % len(train[0]))
    print("%d valid examples" % len(valid[0]))
    print("%d test examples" % len(test[0]))

    print('Initializing the model...')

    # This create the initial parameters as np ndarrays.
    if args.reload_model:
        non_theano_model = Model(archive=model_output_path)
    else:
        non_theano_model = Model(args.feature_count, args.hidden_unit_count, args.category_count)

    # This will create Theano Shared Variables from the model parameters.
    model = TheanoModel(non_theano_model)

    print('Building the network...')
    # use_noise is for dropout
    (use_noise, x, mask, w,
     y, f_pred_prob, f_pred, cost, f_pred_prob_all, hidden_status) = build_network(model, use_dropout=args.use_dropout,
                                                                                   weighted_cost=args.weighted,
                                                                                   random_seed=random_seed)

    # TODO: figure out what is this weight decay, simply L2 regularization? Then decay_c is regularization constant?
    if args.decay_c > 0.:
        decay_c = theano.shared(numpy_floatX(args.decay_c), name='decay_c')
        weight_decay = 0.
        weight_decay += (model.globals.classifier_weights ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    if args.weighted:
        inputs = [x, mask, y, w]
    else:
        inputs = [x, mask, y]

    grads = tensor.grad(cost, wrt=list(model.values()))
    f_grad = theano.function(inputs, grads, name='f_grad')

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = build_optimizer(lr, model, grads,
                                              x, mask, y, cost, w)

    print('Training the model...')

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

                # Get the data in np.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)
                x, mask, y = prepare_data(x, y)
                if args.weighted:
                    w = [train[3][t] for t in train_index]
                    inputs = [x, mask, y, w]
                else:
                    inputs = [x, mask, y]

                n_samples += x.shape[1]

                # # Check gradients
                if check_gradients:
                    grads = f_grad(*inputs)
                    grads_value = grad_array(grads)
                    print('gradients :', [np.mean(g) for g in grads_value])
                    non_theano_model = unzip(model)
                    print('parameter :', [np.mean(vv) for kk, vv in non_theano_model.iteritems()])

                cost = f_grad_shared(*inputs)
                f_update(args.learning_rate)

                if np.isnan(cost) or np.isinf(cost):
                    if np.isinf(cost):
                        raise ValueError("Inf dectected in cost. Cost: {:s}".format(str(cost)))
                    else:
                        raise ValueError("NaN dectected in cost. Cost: {:s}".format(str(cost)))

                if uidx % args.display_interval == 0:
                    print('Epoch ', eidx, 'Update ', uidx, 'Cost ', cost)

                if model_output_path and uidx % save_interval == 0:
                    print('Saving...', end=' ')

                    if best_p is not None:
                        non_theano_model = best_p
                    else:
                        non_theano_model = unzip(model)
                    np.savez(model_output_path, history_errs=history_errs, **non_theano_model)
                    pkl.dump(args.__dict__, open('%s.pkl' % model_output_path, 'wb'), -1)
                    print('Done')

                if uidx % args.validation_interval == 0:
                    use_noise.set_value(0.)
                    train_err = compute_prediction_error(f_pred, train, kf)
                    valid_err = compute_prediction_error(f_pred, valid, kf_valid)
                    test_err = compute_prediction_error(f_pred, test, kf_test)

                    history_errs.append([train_err, valid_err, test_err])
                    eidx_a.append([eidx, eidx, eidx])

                    plt.figure(1)
                    plt.clf()
                    lines = plt.plot(np.array(eidx_a), np.array(history_errs))
                    plt.legend(lines, ['train', 'valid', 'test'])
                    plt.savefig("err.png")
                    time.sleep(0.1)

                    if uidx == 0 or valid_err <= np.array(history_errs)[:, 1].min():

                        best_p = unzip(model)
                        bad_counter = 0
                        if valid_err < np.array(history_errs)[:, 1].min():
                            print('  New best validation results.')

                    print('TrainErr=%.06f  ValidErr=%.06f  TestErr=%.06f' % (train_err, valid_err, test_err))

                    if (len(history_errs) > args.patience and valid_err >= np.array(history_errs)[:-args.patience,
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
        zipp(best_p, model.parameter_dict)
    else:
        best_p = unzip(model.parameter_dict)

    use_noise.set_value(0.)
    kf_train_sorted = get_minibatches_idx(len(train[0]), args.batch_size)
    train_err = compute_prediction_error(f_pred, train, kf_train_sorted)
    valid_err = compute_prediction_error(f_pred, valid, kf_valid)
    test_err = compute_prediction_error(f_pred, test, kf_test)

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
    # TODO: make model path an argument
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
    plt.ion()
    sys.exit(main())
