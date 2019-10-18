"""Private Model Selection.

Use command-line arguments to override default parameters as needed, e.g.:
    >>> python model_selection.py --n 100 --epsilon 0.1
"""
import argparse
from glob import glob
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import get_data_loaders


def plot_probs(ndarray_of_probs, name):
    """Plot probs."""
    if not isinstance(ndarray_of_probs, np.ndarray):
        msg = 'ndarray_of_probs should be a np.ndarray. ' + \
                'Make sure to convert from torch.tensor if need be.'
        raise ValueError(msg)
    if not isinstance(name, str):
        raise ValueError('name should be a str')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    dirname = './figs'
    filename = os.path.join(dirname, name) + '.model-selection-probs.png'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    fig, ax = plt.subplots()
    model_idxs = np.arange(len(ndarray_of_probs))
    ax.bar(model_idxs, ndarray_of_probs)
    ax.set_xlabel('model idx')
    ax.set_ylabel('prob of being selected under Exp Mech')
    ax.set_title(name)
    ax.set_xticks(model_idxs)
    fig.savefig(filename)
    return filename


def load_models(num_pixels):
    """Randomly samples k pre-trained models parameters (from the list of ten)
    """
    list_of_model_filenames = glob('./pretrained_models/*.pt')
    list_of_model_filenames.sort()
    list_of_models = []
    for model_filename in list_of_model_filenames:
        model = nn.Linear(num_pixels, 1, bias=False)
        model.load_state_dict(torch.load(model_filename))
        list_of_models.append(model)
    return list_of_models


def compute_scores(list_of_models, test_loader):
    """Compute score (performance on private test data) for each model"""
    if not isinstance(list_of_models, list):
        raise ValueError('first argument should be a list')
    if not isinstance(test_loader, DataLoader):
        raise ValueError('second argument should be pytorch data loader')

    ############################################################################
    # TODO(student)
    #
    # your code here...
    #
    # You can look into logistic_regression.py to see how various training u
    # metrics are computed given the model
    #
    vector_of_scores = None
    raise NotImplementedError
    ############################################################################

    return vector_of_scores


def exponential_mechanism(list_of_models, test_loader, epsilon):
    """Sample from model list, where sampling probability scales with test score

    Return both the sampled model and the sample index
    """
    if not isinstance(list_of_models, list):
        raise ValueError('first argument should be a list')
    if not isinstance(test_loader, DataLoader):
        raise ValueError('second argument should be pytorch data loader')

    scores = compute_scores(list_of_models, test_loader)
    num_test_examples = len(test_loader.dataset)

    ############################################################################
    # TODO(student)
    #
    # your code here...
    #
    # hint: you're exponential mechanism should somehow depend on the number of
    # training data in test loader
    #
    sampled_model, sampled_idx, sample_probs = None, None, None
    raise NotImplementedError
    ############################################################################

    return sampled_model, sampled_idx, sample_probs


def main(n, epsilon, data_seed, batch_size):
    """Run main private algo."""
    loaders, _ = get_data_loaders(data_seed, batch_size, num_train=13006,
                                  num_test=n)
    num_pixels = loaders['train'].dataset.num_pixels
    models = load_models(num_pixels)

    private_best_model, private_best_model_idx, sample_probs = \
        exponential_mechanism(models, loaders['test'], epsilon)

    print('selected model', private_best_model_idx)
    name = 'eps={},n={}'.format(epsilon, n)
    filename = plot_probs(sample_probs, name)
    print('see plot at', filename)

if __name__ == '__main__':
    # TODO(student): Sweep over the required values for N and EPSILON and
    #                produce several plots. If you like, you may use a bash
    #                script to call this python script several times with
    #                different parameter values.

    parser = argparse.ArgumentParser(
        'Exponential Mechanism for private model selection')
    parser.add_argument(
        '--n', type=int, help='number of examples in test set', default=2163)
    parser.add_argument(
        '--epsilon', type=float, help='privacy parameter', default=2.)
    parser.add_argument(
        '--batch_size', type=int, help='batch size', default=256)
    parser.add_argument(
        '--data_seed', type=int, help='random seed for data', default=0)
    args = parser.parse_args()

    main(args.n, args.epsilon, args.data_seed, args.batch_size)
