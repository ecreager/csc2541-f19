"""Trains a vanilla neural net on the Adult dataset.

Instructions: Run with arguments from command line like this
>>> python train_naive_nnet.py --num_epochs 10 --wd 1e-4 --lr 0.01
"""

import functools
import json
import logging
import os

import argparse
import torch
from torch import nn
from tqdm import tqdm

import utils

def compute_test_metrics(test_batches, model, batch_size):
    """Returns relevant test metrics (to be completed by student)."""
    model.eval()  # disable autograd
    num_test_batches = \
        utils.compute_num_batches(utils.NUM_TEST_DATA, batch_size)

    ############################################################################
    # TODO(student): In this code block, replace the phony_metrics dict with the
    #                dictionary containing the _actual_ test metrics we care
    #                about: test accuracy, test reweighted accuracy, and test
    #                Delta DP.
    phony_metrics = dict(avg_sens_attr=0.)
    for _ in range(num_test_batches):
        x, a, y = next(test_batches)
        x = torch.tensor(x)
        a = torch.tensor(a)
        y = torch.tensor(y)
        batch_avg_sens_attr = torch.mean(a).item()
        phony_metrics['avg_sens_attr'] += \
                batch_avg_sens_attr * (len(a) / float(utils.NUM_TRAIN_DATA))
    metrics = phony_metrics
    ############################################################################

    return metrics


def main(args):
    """Run main training script given parsed arugments"""
    if not isinstance(args, argparse.Namespace):
        raise ValueError("args must be parsed command line arguments.")

    # Set up logging.
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    # NOTE: You may want to choose a different log basename for each question.
    log_basename = 'train_naive_nnet.log'
    log_filename = os.path.join(args.log_dir, log_basename)
    logging.basicConfig(
        filename=log_filename, level=logging.DEBUG, filemode='w')
    logging.info('Script arguments: \n%s', str(args))

    torch.manual_seed(args.seed)
    logging.info('Set random seed to %d', args.seed)

    logging.info('Loading data.')
    train_batches, test_batches = utils.get_batches(args.batch_size, args.seed)
    logging.info('Done loading data.')

    logging.info('Building model.')
    model = nn.Sequential(
        nn.Linear(utils.NUM_FEATURES, 1000),
        nn.Linear(1000, 1),
        )
    logging.info('Done building model: \n%s', str(model))



    logging.info('Building optimizer.')
    criterion = nn.BCEWithLogitsLoss()  # binary cross entropy
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.wd)
    logging.info('Done building optimizer: \n%s', str(optimizer))

    # Train the model
    logging.info('Begin training.')
    model.train()  # enable autograd
    num_batches_per_epoch = \
            utils.compute_num_batches(utils.NUM_TRAIN_DATA, args.batch_size)
    for epoch in tqdm(range(args.num_epochs)):
        train_loss = 0.  # keep track of train loss throughout epoch
        correct = 0  # keep track of num correct guesses throughout epoch
        tensor = functools.partial(torch.tensor, dtype=torch.float32)
        for _ in range(num_batches_per_epoch):
            x, a, y = next(train_batches)
            x = tensor(x)
            a = tensor(a)
            y = tensor(y)
            # Forward pass
            y_hat_logit = model(x)
            y_hat = (y_hat_logit > 0.).float()  # hard predictions
            loss = criterion(y_hat_logit, y.float())
            train_loss += loss * (len(x) / float(utils.NUM_TRAIN_DATA))
            correct += torch.eq(y_hat, y).long().sum().item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_acc = correct / float(utils.NUM_TRAIN_DATA)
        logging.info('Done with epoch %d, train loss = %.2f, train acc = %.2f',
                     epoch, train_loss, train_acc)

    logging.info('Done training.')
    logging.info('Computing test metrics.')
    test_metrics = compute_test_metrics(test_batches, model, args.batch_size)
    logging.info('Done computing test metrics: \n%s', str(test_metrics))

    with open(os.path.join(args.log_dir, 'test_metrics.json'), 'w') as f:
        f.write(json.dumps(test_metrics, indent=4, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Trains a vanilla neural net on the Adult dataset.")
    parser.add_argument("--lr",
                        help="Learning rate.",
                        type=float,
                        default=1e-4)
    parser.add_argument("--wd",
                        help="Weight decay.",
                        type=float,
                        default=1e-5)
    parser.add_argument("--seed",
                        help="Experiment random seed.",
                        type=int,
                        default=0)
    parser.add_argument("--num_epochs",
                        help="Number of training epochs.",
                        type=int,
                        default=5)
    parser.add_argument("--batch_size",
                        help="Batch size.",
                        type=int,
                        default=256)
    parser.add_argument("--log_dir",
                        help="Log directory.",
                        type=str,
                        default="./output")

    args = parser.parse_args()
    main(args)
