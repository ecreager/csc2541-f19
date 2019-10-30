"""Empirical Sensitivity.

Use command-line arguments to override default parameters as needed, e.g.:
    >>> python empirical_sensitivity.py --n 100 --batch_size 512
"""
import argparse
import os

import numpy as np
import torch

from utils import get_data_loaders


def plot_hist(array_of_empirical_sensitivities, n, lmbda, name):
    """Plot histogram."""
    if not isinstance(array_of_empirical_sensitivities, np.ndarray):
        msg = 'array_of_empirical_sensitivities should be a np.ndarray.'
        raise ValueError(msg)
    if not isinstance(name, str):
        raise ValueError('name should be a str')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ############################################################################
    # TODO(student): replace below with correct theoretical max sensitivity
    max_theoretical_sensitivity = -1.
    ############################################################################

    num_bins = 20
    dirname = './figs'
    filename = os.path.join(dirname, name) + '.histogram.png'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    _, _, _ = ax.hist(array_of_empirical_sensitivities,
                      num_bins, label='empirical sensitivities')
    ax.set_title('histogram of sensitivities: ' + name)
    ax.axvline(x=max_theoretical_sensitivity, color='r', linestyle='dashed',
               linewidth=2, label='theoretical max sensitivity')
    ax.legend()
    fig.savefig(filename)
    return filename


def plot_extreme_neighbors(sensitivities, list_of_neighboring_examples, name):
    """Plots neighboring-example pairs with the most/least empirical sensitivity

    Note on the data structures used:
        sensitivities: np array containing empirical sensitivities for each run
        list_of_neighboring_examples: a list of neighboring example pairs, one
                                      for each run. in other words:

        list_of_neighboring_examples = [
            neighboring_example_1,
            neighboring_example_2,
            ...
            neighboring_example_n,
            ]

        where each tuple in the list represents the data diff between the
        neighboring datasets and is formatted like this:

        neighboring_example_i = (
            (neighbor_img_i, neighbor_label_i),
            (neighbor_img_i_prime, neighbor_label_i_prime),
        )

        See utils.py if you are still confused.
    """
    if not isinstance(sensitivities, np.ndarray):
        raise ValueError('sensitivies should be a np.ndarray.')
    first_neighbor_pair = list_of_neighboring_examples[0]
    if not isinstance(list_of_neighboring_examples, list) \
            or not isinstance(first_neighbor_pair, tuple) \
            or not isinstance(first_neighbor_pair[0][0], torch.Tensor):
        msg = ('neighbors should be a list of tuple pairs, ',
               'where tuple contains img tensors')
        raise ValueError(msg)
    if not isinstance(name, str):
        raise ValueError('name should be a str')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ############################################################################
    # TODO(student)
    #
    # using list_of_empirical_sensitivies and neighboring_examples, create two
    # image plots:
    # 1) side-by-side images for neighbor-pair that maximizes sensitivity
    # 2) side-by-side images for neighbor-pair that minimizes sensitivity
    #
    # matplotlib.subplots and matplotlib.imshow may come in handy
    #
    filenames = None, None
    raise NotImplementedError
    ############################################################################

    return filenames


def compute_empricial_sensivity(train_loader,
                                neighbor_loader,
                                num_epochs,
                                learning_rate,
                                lmbda,
                                model_seed=None):
    """Compute empirical sensitivity."""
    ############################################################################
    # TODO(student)
    #
    # your code here...
    #
    #
    sensitivity = None
    raise NotImplementedError
    ############################################################################

    return sensitivity



def main(n, runs, epochs, lr, batch_size, model_seed, lmbda):
    """Run main private algo."""
    list_of_empirical_sensitivies = []
    list_of_neighboring_examples = []
    for data_seed in range(runs):
        # Want deterministic training, so don't shuffle the data
        loaders, neighboring_examples = get_data_loaders(data_seed, batch_size,
                                                         num_train=n, shuffle=False)
        sensitivity = compute_empricial_sensivity(
            loaders['train'], loaders['neighbor'],
            epochs, lr, lmbda, model_seed)
        list_of_empirical_sensitivies.append(sensitivity)
        list_of_neighboring_examples.append(neighboring_examples)

    list_of_empirical_sensitivies = np.array(list_of_empirical_sensitivies)
    name = 'lambda={},n={}'.format(lmbda, n)
    filename = plot_hist(list_of_empirical_sensitivies, n, lmbda, name)
    print('see plot at', filename)

    filenames = plot_extreme_neighbors(
        list_of_empirical_sensitivies, list_of_neighboring_examples, name)
    print('see plots at {} and {}'.format(*filenames))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Computing empirical sensitivity of reularized logistic regression')
    parser.add_argument(
        '--n', type=int, help='number of examples in train set', default=1000)
    parser.add_argument(
        '--runs', type=int,
        help='number of runs used to plot the histogram', default=100)
    parser.add_argument(
        '--epochs', type=int,
        help='number of epochs used to train the logistic regressor',
        default=100)
    parser.add_argument(
        '--lr', type=float, help='learning rate', default=10.0)
    parser.add_argument(
        '--batch_size', type=int, help='batch size', default=1000)
    parser.add_argument(
        '--model_seed', type=float, help='random seed for model initialization',
        default=0)
    parser.add_argument(
        '--lmbda', type=float, help='privacy parameter', default=5e-4)
    args = parser.parse_args()

    main(args.n,
         args.runs,
         args.epochs,
         args.lr,
         args.batch_size,
         args.model_seed,
         args.lmbda)
