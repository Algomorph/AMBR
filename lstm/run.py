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

# local
from lstm.optimizer import get_optimizer_constructor
from lstm.arguments import Arguments
from lstm.params import Parameters, Parameters
from lstm.network_construction import build_network
from lstm.data_io import load_data, prepare_data
from ext_argparse.argproc import process_arguments

mpl.rcParams['image.interpolation'] = 'nearest'


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def to_numpy_theano_float(data):
    return np.asarray(data, dtype=config.floatX)


def get_minibatch_indices(dataset_length, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    index_list = np.arange(dataset_length, dtype="int32")

    if shuffle:
        np.random.shuffle(index_list)

    minibatch_index_sets = []
    for minibatch_start in range(0, dataset_length, minibatch_size):
        minibatch_index_sets.append(index_list[minibatch_start:minibatch_start + minibatch_size])

    if minibatch_start != dataset_length:
        # Make a minibatch out of what is left
        minibatch_index_sets.append(index_list[minibatch_start:])

    # enumerate minibatch_index_sets,
    # returns a list of tuples in the form (<index_of_minibatch>, <minibatch_index_set>)
    return minibatch_index_sets


def theano_to_numpy_grad_array(theano_gradients):
    return [np.asarray(g) for g in theano_gradients]


def compute_prediction_error(classify_sequence, data, index_sets, draw_histogram_function=None):
    """
    Compute the prediction error.
    :param draw_histogram_function:
    :param index_sets:
    :type classify_sequence: theano.compile.function_module.Function
    :param classify_sequence: Theano function for computing the prediction
    :type data: lstm.data_io.SequenceDataset
    :param data: the dataset for which to get classification error consistent of sequences and their labels
    prepare_data: usual prepare_data for that dataset.
    """
    error = 0
    for index_set in index_sets:
        batch_features, mask, batch_labels = prepare_data([data.features[t] for t in index_set],
                                                          np.array(data.labels)[index_set])
        sequence_classifications = classify_sequence(batch_features, mask)
        target_labels = np.array(data.labels)[index_set]
        error += (sequence_classifications == target_labels).sum()

    error = 1. - to_numpy_theano_float(error) / len(data)

    if draw_histogram_function is not None:
        batch_features = data.features[:, None, :].astype('float32')
        mask = np.ones((batch_features.shape[0], 1), dtype='float32')
        # h, c, i, f, o
        hs = draw_histogram_function(batch_features, mask)
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

    return error


def compute_prediction_precision_and_recall(compute_sequence_class_probabilities, sequence_dataset,
                                            batch_index_sets, category_count, verbose=False):
    n_samples = len(sequence_dataset)
    category_probabilities = np.zeros((n_samples, category_count)).astype(config.floatX)
    true_labels = np.zeros((n_samples,)).astype('int32')

    for batch_index_set in batch_index_sets:
        x, mask, y = prepare_data([sequence_dataset.features[t] for t in batch_index_set],
                                  np.array(sequence_dataset.labels)[batch_index_set])
        print(x.shape, mask.shape, y.shape)
        minibatch_category_probabilities = compute_sequence_class_probabilities(x, mask)
        category_probabilities[batch_index_set, :] = minibatch_category_probabilities
        true_labels[batch_index_set] = np.array(sequence_dataset.labels)[batch_index_set]

    predicted_labels = np.argmax(category_probabilities, axis=1)
    confusion_matrix = calculate_confusion_matrix(true_labels, predicted_labels, category_count)
    correct_predictions = np.diagonal(confusion_matrix)
    samples_per_class = np.sum(confusion_matrix, axis=0)
    false_positives = np.sum(confusion_matrix, axis=1) - correct_predictions
    false_negatives = samples_per_class - correct_predictions

    prectmp = correct_predictions / (correct_predictions + false_positives)
    prectmp[np.where(correct_predictions == 0)[0]] = 0
    prectmp[np.where(samples_per_class == 0)[0]] = float('nan')
    precision = np.nanmean(prectmp)

    rectmp = correct_predictions / (correct_predictions + false_negatives)
    rectmp[np.where(correct_predictions == 0)[0]] = 0
    rectmp[np.where(samples_per_class == 0)[0]] = float('nan')
    recall = np.nanmean(rectmp)

    return category_probabilities, true_labels, precision, recall


def test_lstm(model_output_path, args, result_dir=None):
    args.model_file = model_output_path
    print("Model options: ", args)
    print("Loading test data...")
    training_data, validation_data, test_data, n_categories = load_data(args.datasets, args.folder)

    print("Sample counts:")
    print("%d training samples" % len(training_data))
    print("%d validation samples" % len(validation_data))
    print("%d test samples" % len(test_data))

    if not result_dir:
        result_dir = 'test_results9'

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    args.validation_batch_size = 1
    parameters = Parameters(archive=np.load(model_output_path))
    (use_noise, x, mask, w, y, compute_sequence_class_probabilities,
     f_pred, cost, f_pred_prob_all, hidden_status) = build_network(parameters, use_dropout=args.use_dropout,
                                                                   weighted_cost=args.weighted)

    test_minibatch_index_sets = get_minibatch_indices(len(test_data), args.validation_batch_size)
    print("%d test examples" % len(test_data))

    class_probabilities, gts, prec, rec = compute_prediction_precision_and_recall(compute_sequence_class_probabilities,
                                                                                  test_data,
                                                                                  test_minibatch_index_sets,
                                                                                  args.category_count, verbose=False)
    predicted_labels = np.argmax(class_probabilities, axis=1)
    cm = calculate_confusion_matrix(gts, predicted_labels, category_count=args.category_count)
    cm = np.asarray(cm, 'float32')
    cm = cm / np.sum(cm, axis=0)
    cm[np.where(np.isnan(cm))] = 0
    f = plt.figure(2)
    f.clf()
    ax = f.add_axes([0.1, 0.1, 0.8, 0.8])
    im = ax.imshow(cm, interpolation='nearest')
    f.colorbar(im)
    plt.savefig("%s/confusion_matrix_sub.png" % result_dir)

    results = {'scores': class_probabilities,
               'gts': gts,
               'prec': prec,
               'rec': rec}
    result_file = '%s/%s_result.mat' % (result_dir, model_output_path.split('/')[-1].split('.')[0])
    sio.savemat(result_file, results)

    predicted_labels = []
    for t in range(len(test_data[0])):
        x, mask, y = prepare_data([test_data[0][t]], np.array(test_data[1])[t])
        predicted_labels.append(f_pred_prob_all(x, mask))

    results_all = {'preds_all': predicted_labels,
                   'gts': gts,
                   'start_frame': [d['s_fid'] for d in test_data[2]],
                   'end_frame': [d['e_fid'] for d in test_data[2]],
                   'label': [d['label'] for d in test_data[2]]}

    results_all_file = '%s/%s_result_all.mat' % (result_dir, model_output_path.split('/')[-1].split('.')[0])
    sio.savemat(results_all_file, results_all)
    return


def train_lstm(model_output_path, args, check_gradients=False):
    random_seed = 2016
    np.random.seed(random_seed)
    args.model_file = model_output_path
    print("Model options: ", args)

    save_interval = args.save_interval
    build_optimizer = get_optimizer_constructor(args.optimizer)

    print('Loading data...')
    print('Sample counts:')
    training_dataset, validation_dataset, test_dataset, n_categories = load_data(args.datasets, args.folder)
    print("%d training samples" % len(training_dataset))
    print("%d validation samples" % len(validation_dataset))
    print("%d testing samples" % len(test_dataset))

    print('Initializing the model...')

    # This create the initial parameters as np ndarrays.
    # This will create Theano Shared Variables from the model parameters.
    if args.reload_model:
        parameters = Parameters(archive=model_output_path)
    else:
        parameters = Parameters(args.feature_count, args.hidden_unit_count, args.category_count)

    print('Building the network...')
    # use_noise is for dropout
    (use_noise_flag, batch_features, mask, w, batch_labels, compute_sequence_class_probabilities, classify_sequence,
     compute_loss, f_pred_prob_all, hidden_status) = build_network(parameters, use_dropout=args.use_dropout,
                                                                   weighted_cost=args.weighted, random_seed=random_seed)

    # TODO: figure out what is this weight decay, simply L2 regularization? Then decay_c is regularization constant?
    if args.decay_constant > 0.:
        decay_constant = theano.shared(to_numpy_theano_float(args.decay_constant), name='decay_c')
        weight_decay = 0.
        weight_decay += (parameters.globals.classifier_weights ** 2).sum()
        weight_decay *= decay_constant
        compute_loss += weight_decay

    if args.weighted:
        inputs = [batch_features, mask, batch_labels, w]
    else:
        inputs = [batch_features, mask, batch_labels]

    grads = tensor.grad(compute_loss, wrt=list(parameters.values()))
    f_grad = theano.function(inputs, grads, name='f_grad')

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = build_optimizer(lr, parameters, grads,
                                              batch_features, mask, batch_labels, compute_loss, w)

    print('Training the model...')

    validation_minibatch_indices = get_minibatch_indices(len(validation_dataset), args.validation_batch_size)
    test_minibatch_indices = get_minibatch_indices(len(test_dataset), args.validation_batch_size)

    error_history = []
    epoch_index_aggregate = []

    if save_interval == -1:
        save_interval = len(training_dataset) / args.batch_size

    current_update_index = 0
    early_stop = False
    start_time = time.time()
    try:
        for epoch_index in range(args.max_epochs):
            n_samples = 0

            # Get new shuffled index for the training set.
            train_minibatch_indices = get_minibatch_indices(len(training_dataset), args.batch_size, shuffle=True)

            # traverse all the mini-batches
            for training_minibatch_indices in train_minibatch_indices:
                use_noise_flag.set_value(1.)

                # Select the random sequences for this minibatch
                batch_features = [training_dataset.features[t] for t in training_minibatch_indices]
                batch_labels = [training_dataset.labels[t] for t in training_minibatch_indices]

                # Get the data in np.ndarray format
                # Swaps the axes!
                # Returns a matrix of shape (minibatch max. len., n samples)
                batch_features, mask, batch_labels = prepare_data(batch_features, batch_labels)
                if args.weighted:
                    w = [training_dataset.weights[t] for t in training_minibatch_indices]
                    inputs = [batch_features, mask, batch_labels, w]
                else:
                    inputs = [batch_features, mask, batch_labels]

                n_samples += batch_features.shape[1]

                # # Check gradients
                if check_gradients:
                    grads = f_grad(*inputs)
                    grads_value = theano_to_numpy_grad_array(grads)
                    print('gradients :', [np.mean(g) for g in grads_value])
                    print('parameters :', [np.mean(vv) for kk, vv in parameters.as_dict().items()])

                compute_loss = f_grad_shared(*inputs)
                f_update(args.learning_rate)

                if np.isinf(compute_loss):
                    raise ValueError("Inf dectected in cost. Aborting.")
                elif np.isnan(compute_loss):
                    raise ValueError("NaN dectected in cost. Aborting.")

                if current_update_index % args.display_interval == 0:
                    print('Epoch ', epoch_index, 'Update ', current_update_index, 'Cost ', compute_loss)

                if model_output_path and (current_update_index + 1) % save_interval == 0:
                    print('Saving...', end=' ')
                    parameters.save_to_numpy_archive(model_output_path)
                    print('Done')

                if (current_update_index + 1) % args.validation_interval == 0:
                    use_noise_flag.set_value(0.)
                    training_error = compute_prediction_error(classify_sequence, training_dataset,
                                                              train_minibatch_indices)
                    validation_error = compute_prediction_error(classify_sequence, validation_dataset,
                                                                validation_minibatch_indices)
                    test_error = compute_prediction_error(classify_sequence, test_dataset, test_minibatch_indices)

                    error_history.append([training_error, validation_error, test_error])
                    epoch_index_aggregate.append([epoch_index, epoch_index, epoch_index])

                    plt.figure(1)
                    plt.clf()
                    lines = plt.plot(np.array(epoch_index_aggregate), np.array(error_history))
                    plt.legend(lines, ['train', 'valid', 'test'])
                    plt.savefig("err.png")
                    time.sleep(0.1)

                    if current_update_index == 0 or validation_error <= np.array(error_history)[:, 1].min():
                        best_parameters = parameters.as_dict()
                        bad_counter = 0
                        if validation_error < np.array(error_history)[:, 1].min():
                            print('  New best validation results.')

                    print('TrainErr=%.06f  ValidErr=%.06f  TestErr=%.06f'
                          % (training_error, validation_error, test_error))

                    if (len(error_history) > args.patience
                        and validation_error >= np.array(error_history)[:-args.patience, 1].min()):
                        bad_counter += 1
                        if bad_counter > args.patience:
                            print('Early stop: validation error exceeded the minimum error ' +
                                  'in the last few epochs too many times!')
                            early_stop = True
                            break

                current_update_index += 1

            print('Seen %d samples' % n_samples)

            if early_stop:
                break

    except KeyboardInterrupt:
        print("Training interrupted")

    end_time = time.time()
    if best_parameters is None:
        best_parameters = parameters.as_dict()

    use_noise_flag.set_value(0.)
    sorted_train_minibatch_index_sets = get_minibatch_indices(len(training_dataset), args.batch_size)
    training_error = compute_prediction_error(classify_sequence, training_dataset, sorted_train_minibatch_index_sets)
    validation_error = compute_prediction_error(classify_sequence, validation_dataset, validation_minibatch_indices)
    test_error = compute_prediction_error(classify_sequence, test_dataset, test_minibatch_indices)

    print('TrainErr=%.06f  ValidErr=%.06f  TestErr=%.06f' % (training_error, validation_error, test_error))
    if model_output_path:
        np.savez(model_output_path, train_err=training_error,
                 valid_err=validation_error, test_err=test_error,
                 history_errs=error_history, **best_parameters)
    print('The code run for %d epochs, with %f sec/epochs' % (
        (epoch_index + 1), (end_time - start_time) / (1. * (epoch_index + 1))))
    print(('Training took %.1fs' %
           (end_time - start_time)), file=sys.stderr)
    return training_error, validation_error, test_error


def calculate_confusion_matrix(true_labels, predicted_labels, category_count):
    confusion_matrix = np.zeros((category_count, category_count))
    for i in range(category_count):
        idx_category = np.where(true_labels == i)[0]
        if idx_category.size == 0:
            continue
        predicted_category = predicted_labels[idx_category]
        for j in range(category_count):
            confusion_matrix[j, i] = np.where(predicted_category == j)[0].shape[0]

    return confusion_matrix


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
