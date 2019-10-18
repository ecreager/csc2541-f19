"""Private Training by Output Perturbation.

Use command-line arguments to override default parameters as needed, e.g.:
    >>> python output_perturbation.py --n 100 --batch_size 512
"""
import argparse

import numpy as np
from scipy.stats import ortho_group
import torch
from torch.distributions.gamma import Gamma
from torch import nn

from logistic_regression import nonprivate_logistic_regression
from utils import get_data_loaders

def gamma_sample_pytorch_parameterization(concentration, rate):
    """The Gamma dist'n as it is parameterized in PyTorch"""
    return Gamma(concentration, rate).sample()


def gamma_sample_chaudhuri_parameterization(concentration, scale):
    """The Gamma dist'n as it is parameterized in Chaudhuri and Monteleoni"""
    rate = 1. / scale
    return gamma_sample_pytorch_parameterization(concentration, rate)


def random_unit_norm_vector(num_dims):
    """Produce random unit-norm vector."""
    random_rotation_matrix = ortho_group.rvs(num_dims)
    basis_vector_one = np.eye(num_dims)[0]
    vector = np.matmul(random_rotation_matrix, basis_vector_one)
    return torch.tensor(vector, dtype=torch.float32)


def private_logistic_regression(dset_loader, num_epochs, learning_rate,
                                lmbda, epsilon, seed=None):
    """Run private logistic regression."""
    ############################################################################
    # TODO(student)
    #
    # your code here...
    #
    # hint: use the code we have given you. For example you don't have to
    # implement non-private logistic regression from scratch because an
    # implementation exists in logistic_regression.py. There are also functions
    # in this file for sampling Laplace noise
    #
    # hint: the input dim d can be found as a attr of the dset_loader's dset
    #       >>> num_pixels = dset_loader.dataset.num_pixels
    #
    private_params = {
        'weight': torch.tensor([-1.]),  # replace me (but this is how to format
                                        #             the state_dict)
        }
    raise NotImplementedError
    ############################################################################

    return private_params


def main(n, epsilon, lmbda, epochs, batch_size, lr, data_seed, model_seed):
    """Run main private algo."""
    # load data
    loaders, _ = get_data_loaders(data_seed, batch_size, n)
    loaders.pop('neighbor')  # don't need this loader for this question

    # train model
    nonprivate_params = nonprivate_logistic_regression(
        loaders['train'], epochs, lr, lmbda, seed=model_seed)

    private_params = private_logistic_regression(
        loaders['train'], epochs, lr, lmbda, epsilon, seed=model_seed)

    # evaluate
    test_losses = dict()
    test_accs = dict()
    for name, params in zip(['nonprivate', 'private'],
                            [nonprivate_params, private_params]):
        num_pixels = loaders['train'].dataset.num_pixels
        model = nn.Linear(num_pixels, 1, bias=False)
        criterion = nn.BCEWithLogitsLoss()  # binary cross entropy
        model.load_state_dict(params)
        model.eval()
        num_test_examples = len(loaders['test'].dataset)
        with torch.no_grad():
            test_loss = 0.
            correct = 0
            total = 0
            for images, labels in loaders['test']:
                images = images.reshape(-1, 28*28)
                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels.float())
                test_loss += \
                        loss.item() * len(images) / float(num_test_examples)
                predicted = (outputs.squeeze() > 0.).long()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            test_acc = float(correct) / float(total)
            test_losses[name] = test_loss
            test_accs[name] = 100. * test_acc  # format as a percentage

    print('final test losses')
    print('nonprivate: {nonprivate:.2f}, private: {private:.2f}'
          .format(**test_losses))
    print('final test accs')
    print('nonprivate: {nonprivate:.2f}, private: {private:.2f}'
          .format(**test_accs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Private Training by Output Perturbation')
    parser.add_argument(
        '--n', type=int, help='number of examples in train set', default=13006)
    parser.add_argument(
        '--epsilon', type=float, help='privacy parameter', default=2.)
    parser.add_argument(
        '--lmbda', type=float, help='L2 regularizer', default=5e-4)
    parser.add_argument(
        '--epochs', type=int, help='# epochs', default=100)
    parser.add_argument(
        '--batch_size', type=int, help='batch size', default=256)
    parser.add_argument(
        '--lr', type=float, help='learning rate', default=.3)
    parser.add_argument(
        '--data_seed', type=int, help='random seed for data', default=0)
    parser.add_argument(
        '--model_seed', type=int, help='random seed for model initialization',
        default=0)
    args = parser.parse_args()
    main(args.n,
         args.epsilon,
         args.lmbda,
         args.epochs,
         args.batch_size,
         args.lr,
         args.data_seed,
         args.model_seed)
