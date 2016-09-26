#  ================================================================
#  Created by Gregory Kramida on 9/20/16.
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

# system
import time
import sys
# 3rd party
import theano
import os.path
from theano import tensor, config
import numpy as np
from matplotlib import pyplot as plt
# local
from lstm.network_construction import build_network
from lstm.common import to_numpy_theano_float
from lstm.optimizer import get_optimizer_constructor
from lstm.params import Parameters
from lstm.data_io import prepare_data


class Model(object):
    """
    Class that represents an LSTM network padded by embedding & classification layers.
    """

    def __init__(self, optimizer_name, use_dropout=True, weighted_loss=False, random_seed=2016,
                 decay_constant=0.0, parameters=None, hidden_unit_count=None, category_count=None, feature_count=None,
                 model_output_path=None):
        """
        Constructs an LSTM-based ML model with an embedding & a classification layer.
        :type optimizer_name: str
        :param optimizer_name: one of keys of lstm.optimizer.optimizer_dict, represents name of the
        learning rate/gradients optimizer to perform weight & bias updates.
        :type use_dropout: bool
        :param use_dropout: whether or not to use dropout. Reference:
    Srivastava, Nitish, Geoffrey E. Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov.
    "Dropout: a simple way to prevent neural networks from overfitting." Journal of Machine Learning Research 15, no. 1
    (2014): 1929-1958.
        :type weighted_loss: bool
        :param weighted_loss: whether or not to weigh loss by category (in which case, proper weights need to be set
        for very sample/sequence in the training datasets.)
        :type random_seed: int
        :param random_seed: random seed to use for the training.
        :type decay_constant: float
        :param decay_constant: some kind of L2 regularization on the gradients before feeding them into the optimizer.
        :type parameters: lstm.params.Parameters
        :param parameters: optional starting parameters for the training. These override the hidden_unit_count,
        category_count, and feature_count settings
        :type hidden_unit_count: int
        :param hidden_unit_count: in case starting parameters are not provided, number of hidden units in the LSTM
        layer.
        """
        if parameters is None and hidden_unit_count is None:
            raise ValueError("Either all parameters or the hidden unit count should be specified.")

        if parameters is not None:
            self.hidden_unit_count = parameters.hidden_unit_count
            self.category_count = parameters.category_count
            self.feature_count = parameters.feature_count
        else:
            self.hidden_unit_count = hidden_unit_count
            self.category_count = category_count
            self.feature_count = feature_count

        self.model_output_path = model_output_path
        self.output_directory = None
        if model_output_path is not None:
            self.output_directory = os.path.dirname(self.model_output_path)

        self.parameters = parameters
        self.random_seed = random_seed

        # build the actual network and compile all the necessary functions for training & testing
        (self.noise_flag, self.batch_features, self.mask, self.cost_weights, self.batch_labels,
         self.compute_sequence_class_probabilities, self.classify_sequence, self.compute_loss,
         self.classify_timestep, self.get_network_state) = \
            build_network(parameters, use_dropout, weighted_loss, random_seed)
        build_optimizer = get_optimizer_constructor(optimizer_name)

        if decay_constant > 0.:
            decay_constant = theano.shared(to_numpy_theano_float(decay_constant), name='decay_c')
            weight_decay = 0.
            weight_decay += (parameters.globals.classifier_weights ** 2).sum()
            weight_decay *= decay_constant
            self.compute_loss += weight_decay

        self.weighted_loss = weighted_loss
        if weighted_loss:
            input_tensors = [self.batch_features, self.mask, self.batch_labels, self.cost_weights]
        else:
            input_tensors = [self.batch_features, self.mask, self.batch_labels]

        # gradients of loss with respect to parameters
        gradients = tensor.grad(self.compute_loss, wrt=list(parameters.values()))
        self.compute_gradients = theano.function(input_tensors, gradients, name='f_grad')

        learing_rate = tensor.scalar(name='learning_rate')
        self.compute_shared_gradient, self.update_parameters = build_optimizer(learing_rate, parameters, gradients,
                                                                               self.batch_features, self.mask,
                                                                               self.batch_labels,
                                                                               self.compute_loss, self.cost_weights)
        self.noise_flag.set_value(1.)

    def compute_prediction_error(self, data, index_sets, compute_histogram_function=None):
        """
        Compute the prediction error.
        :param compute_histogram_function: function to compute the histogram
        :param index_sets: sets of indexes for batches
        :type data: lstm.data_io.SequenceDataset
        :param data: the dataset for which to get classification error consistent of sequences and their labels
        prepare_data: usual prepare_data for that dataset.
        """
        error = 0
        for index_set in index_sets:
            batch_features, mask, batch_labels = prepare_data([data.features[t] for t in index_set],
                                                              np.array(data.labels)[index_set])
            sequence_classifications = self.classify_sequence(batch_features, mask)
            target_labels = np.array(data.labels)[index_set]
            error += (sequence_classifications == target_labels).sum()

        error = 1. - to_numpy_theano_float(error) / len(data)

        if compute_histogram_function is not None:
            batch_features = data.features[:, None, :].astype('float32')
            mask = np.ones((batch_features.shape[0], 1), dtype='float32')
            # h, c, i, f, o
            hs = compute_histogram_function(batch_features, mask)
            plt.figure(1)
            plt.clf()
            for s in range(5):
                plt.subplot(1, 5, s + 1)
                plt.imshow(np.squeeze(hs[s][:, 0, :]), interpolation='nearest')
                plt.colorbar()
            if self.output_directory is not None:
                plt.savefig(os.path.join(self.output_directory, "hs_test_tmp.png"))

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
            if self.output_directory is not None:
                plt.savefig(os.path.join(self.output_directory, "hs_matrix.png"))

            plt.figure(3)
            plt.clf()
            plt.subplot(2, 1, 1)
            plt.plot(hs[7])
            plt.title("hs_Bvec_lstm")
            plt.subplot(2, 1, 2)
            plt.plot(hs[9])
            plt.title("hs_Bvec")
            if self.output_directory is not None:
                plt.savefig(os.path.join(self.output_directory, "hs_vector.png"))

            time.sleep(0.1)

        return error

    @staticmethod
    def get_batch_indices(dataset_length, minibatch_size, shuffle=False):
        """
        Used to shuffle the dataset at each iteration.
        """

        index_list = np.arange(dataset_length, dtype="int32")

        if shuffle:
            np.random.shuffle(index_list)

        minibatch_index_sets = []
        minibatch_start = 0
        for _ in range(dataset_length // minibatch_size):
            minibatch_index_sets.append(index_list[minibatch_start:minibatch_start + minibatch_size])
            minibatch_start += minibatch_size

        if minibatch_start != dataset_length:
            # Make a minibatch out of what is left
            minibatch_index_sets.append(index_list[minibatch_start:])

        # enumerate minibatch_index_sets,
        # returns a list of tuples in the form (<index_of_minibatch>, <minibatch_index_set>)
        return minibatch_index_sets

    @staticmethod
    def theano_to_numpy_grad_array(theano_gradients):
        return [np.asarray(g) for g in theano_gradients]

    def initialize_parameters(self, dataset):
        """
        Initialize parameters. If some values were not given,
        such as feature_count and category count, try to derive them from the given dataset.
        :param dataset:
        :return:
        """
        if self.feature_count is None:
            self.feature_count = len(dataset.features[0])
        if self.category_count is None:
            self.category_count = len(np.unique(dataset.labels))
        parameters = Parameters(feature_count=self.feature_count, hidden_unit_count=self.hidden_unit_count,
                                category_count=self.category_count)

    def train_on(self, training_data, validation_data, test_data,
                 batch_size=10, validation_batch_size=5,
                 report_interval=50, validation_interval=20, save_interval=20,
                 patience=15, max_epochs=300, learning_rate=0.0001, check_gradients=False, verbose=True):
        """
        Train the model.

        :type training_data: lstm.data_io.SequenceDataset
        :param training_data: dataset to use for actual training/learning
        :type validation_data: lstm.data_io.SequenceDataset
        :param validation_data: dataset to use for validation only, i.e. comparison with labels is only used for early
        stop to prevent overfitting.
        :type test_data: lstm.data_io.SequenceDataset
        :param test_data: dataset set aside for testing
        :type batch_size: int
        :param batch_size: size of batches to use for training
        :type validation_batch_size: int
        :param validation_batch_size: size of batches to use for validation
        :type save_interval: int
        :param save_interval: number of updates until the model is written to disk
        (only if model_output_path is specified)
        :type patience: int
        :param patience: number of validation runs to check whether a better validation result is obtained. After
        the number of checks reaches "patience", training will be stopped "early"
        :type learning_rate: float
        :param learning_rate: factor for weight updates
        :type validation_interval: int
        :param validation_interval: number of updates until the next validation run
        :type report_interval: int
        :param report_interval: number of updates until printing the results again to the console
        :type max_epochs: int
        :param max_epochs: maximum number of epochs to train (each epoch will cover batches based on entire dataset)
        :type check_gradients: bool
        :param check_gradients: whether to print out gradients during training
        :type verbose: bool
        :param verbose: print supplementary output
        :return:
        """

        if self.parameters is None:
            self.initialize_parameters(training_data)

        validation_batch_indices = Model.get_batch_indices(len(validation_data), validation_batch_size)
        test_batch_indices = Model.get_batch_indices(len(test_data), validation_batch_size)

        error_history = []
        epoch_index_aggregate = []

        if save_interval == -1:
            save_interval = len(training_data) / batch_size

        current_update_index = 0
        early_stop = False
        start_time = time.time()

        best_parameters = None

        try:
            for epoch_index in range(max_epochs):

                epoch_samples_processed = 0

                # Get new shuffled index for the training set.
                train_minibatch_indices = Model.get_batch_indices(len(training_data), batch_size, shuffle=True)

                # traverse all the mini-batches
                for training_minibatch_indices in train_minibatch_indices:

                    # Select the random sequences for this minibatch
                    batch_features = [training_data.features[t] for t in training_minibatch_indices]
                    batch_labels = [training_data.labels[t] for t in training_minibatch_indices]

                    # Get the data in np.ndarray format
                    # Swaps the axes!
                    # Returns a matrix of shape (minibatch max. len., n samples)
                    batch_features, mask, batch_labels = prepare_data(batch_features, batch_labels)
                    if self.weighted_loss:
                        w = [training_data.weights[t] for t in training_minibatch_indices]
                        inputs = [batch_features, mask, batch_labels, w]
                    else:
                        inputs = [batch_features, mask, batch_labels]

                    epoch_samples_processed += batch_features.shape[1]

                    # # Check gradients
                    if check_gradients:
                        gradients = self.compute_gradients(*inputs)
                        print('gradients :', [np.mean(g) for g in Model.theano_to_numpy_grad_array(gradients)])
                        print('parameters :', [np.mean(vv) for kk, vv in self.parameters.as_dict().items()])

                    loss = self.compute_shared_gradient(*inputs)
                    self.update_parameters(learning_rate)

                    if np.isinf(loss):
                        raise ValueError("Inf detected in cost. Aborting.")
                    elif np.isnan(loss):
                        raise ValueError("NaN detected in cost. Aborting.")

                    if (current_update_index + 1) % report_interval == 0:
                        if verbose:
                            print('Epoch: ', epoch_index, ' | Update: ', current_update_index, '|Loss/penalty: ', loss)

                    if self.model_output_path is not None and (current_update_index + 1) % save_interval == 0:
                        if verbose:
                            print('Saving...', end=' ')
                        self.parameters.save_to_numpy_archive(self.model_output_path)
                        if verbose:
                            print('Done')

                    if (current_update_index + 1) % validation_interval == 0:
                        self.noise_flag.set_value(0.)
                        training_error = self.compute_prediction_error(training_data, train_minibatch_indices)
                        validation_error = self.compute_prediction_error(validation_data, validation_batch_indices)
                        test_error = self.compute_prediction_error(test_data, test_batch_indices)

                        error_history.append([training_error, validation_error, test_error])
                        epoch_index_aggregate.append([epoch_index, epoch_index, epoch_index])

                        plt.figure(1)
                        plt.clf()
                        lines = plt.plot(np.array(epoch_index_aggregate), np.array(error_history))
                        plt.legend(lines, ['training error', 'validation error', 'test error'])
                        if self.output_directory:
                            plt.savefig(os.path.join(self.output_directory, "error.png"))
                        time.sleep(0.1)

                        # TODO: check the logic here to see if it actually works
                        if current_update_index == 0 or validation_error <= np.array(error_history)[:, 1].min():
                            best_parameters = self.parameters.as_dict()  # save best validation results so far
                            bad_counter = 0
                            if validation_error < np.array(error_history)[:, 1].min() and verbose:
                                print('  New best validation results.')
                        if verbose:
                            print("Training error=%.06f |  validation error=%.06f | test error=%.06f" % (
                                training_error, validation_error, test_error))

                        if (len(error_history) > patience
                            and validation_error >= np.array(error_history)[:-patience, 1].min()):
                            bad_counter += 1
                            if bad_counter > patience:
                                print("Early stop: validation error exceeded the minimum error " +
                                      "in the last few epochs too many times!")
                                early_stop = True
                                break
                        self.noise_flag.set_value(1.)

                    current_update_index += 1

                if verbose:
                    print('Seen %d samples.' % epoch_samples_processed)

                if early_stop:
                    break

        except KeyboardInterrupt:
            print("Training interrupted")

        end_time = time.time()
        if best_parameters is None:
            best_parameters = self.parameters.as_dict()

        self.noise_flag.set_value(0.)
        sorted_train_minibatch_index_sets = Model.get_batch_indices(len(training_data), batch_size)
        training_error = self.compute_prediction_error(training_data, sorted_train_minibatch_index_sets)
        validation_error = self.compute_prediction_error(validation_data, validation_batch_indices)
        test_error = self.compute_prediction_error(test_data, test_batch_indices)
        if verbose:
            print("Training error=%.06f |  Validation error=%.06f |  Test error=%.06f"
                  % (training_error, validation_error, test_error))
            print("The code run for %d epochs, with %f sec/epochs" % (
                (epoch_index + 1), (end_time - start_time) / (1. * (epoch_index + 1))))
            print(("Training took %.1fs" %
                   (end_time - start_time)), file=sys.stderr)
        if self.model_output_path is not None:
            np.savez(self.model_output_path, training_error=training_error, test_error=test_error,
                     validation_error=validation_error, history_errs=error_history, **best_parameters)

        return training_error, validation_error, test_error

    @staticmethod
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

    def compute_prediction_precision_and_recall(self, sequence_dataset,
                                                batch_index_sets, verbose=False):
        n_samples = len(sequence_dataset)
        category_probabilities = np.zeros((n_samples, self.category_count)).astype(config.floatX)
        true_labels = np.zeros((n_samples,)).astype('int32')

        for batch_index_set in batch_index_sets:
            x, mask, y = prepare_data([sequence_dataset.features[t] for t in batch_index_set],
                                      np.array(sequence_dataset.labels)[batch_index_set])
            minibatch_category_probabilities = self.compute_sequence_class_probabilities(x, mask)
            category_probabilities[batch_index_set, :] = minibatch_category_probabilities
            true_labels[batch_index_set] = np.array(sequence_dataset.labels)[batch_index_set]

        predicted_labels = np.argmax(category_probabilities, axis=1)
        confusion_matrix = Model.calculate_confusion_matrix(true_labels, predicted_labels, self.category_count)
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

    def test_on(self, test_data, verbose=True):

        if self.parameters is None:
            # the odd case when the user wants to test a completely un-trained model
            self.initialize_parameters(test_data)

        validation_batch_size = 1

        test_minibatch_index_sets = Model.get_batch_indices(len(test_data), validation_batch_size)
        print("%d test examples" % len(test_data))

        class_probabilities, true_labels, precision, recall = \
            self.compute_prediction_precision_and_recall(test_data, test_minibatch_index_sets, verbose=False)

        predicted_labels = np.argmax(class_probabilities, axis=1)
        cm = Model.calculate_confusion_matrix(true_labels, predicted_labels, category_count=self.category_count)
        cm = np.asarray(cm, 'float32')
        cm = cm / np.sum(cm, axis=0)
        cm[np.where(np.isnan(cm))] = 0
        f = plt.figure(2)
        f.clf()
        ax = f.add_axes([0.1, 0.1, 0.8, 0.8])
        im = ax.imshow(cm, interpolation='nearest')
        f.colorbar(im)
        if self.output_directory is not None:
            plt.savefig(os.path.join(self.output_directory, "confusion_matrix_sub.png"))

        results = {'class_scores': class_probabilities,
                   'true_labels': true_labels,
                   'precision': precision,
                   'recall': recall}
        if self.output_directory is not None:
            np.savez_compressed(os.path.join(self.output_directory, "aggregate_test_results.npz"), **results)

        predicted_labels = []
        for t in range(len(test_data)):
            x, mask, y = prepare_data([test_data.features[t]], np.array(test_data.labels)[t])
            predicted_labels.append(self.classify_timestep(x, mask))

        results_all = {'predicted_labels': predicted_labels,
                       'true_labels': true_labels,
                       'start_frame': [d['start'] for d in test_data.meta_information],
                       'end_frame': [d['end'] for d in test_data.meta_information],
                       'label': [d['label'] for d in test_data.meta_information]}
        if self.output_directory is not None:
            np.savez_compressed(os.path.join(self.output_directory, "detailed_test_results.npz"), **results_all)

        return
