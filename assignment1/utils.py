"""Helper functions for datasets."""
import numpy as np
import numpy.random as npr

TRAIN_DATA_FILENAME = "./adult/adult_train.npz"
TEST_DATA_FILENAME = "./adult/adult_test.npz"
NUM_TRAIN_DATA = 32561
NUM_TEST_DATA = 16281
NUM_FEATURES = 113


def compute_num_batches(num_examples, batch_size):
    """Compute number of batches per epoch given dataset and batch sizes."""
    num_complete_batches, leftover = divmod(num_examples, batch_size)
    num_batches = num_complete_batches + bool(leftover)
    return num_batches


def get_batches(batch_size, seed=None):
    """Build dataset loaders for Adult UCI data"""
    train_data_dict = np.load(TRAIN_DATA_FILENAME)
    test_data_dict = np.load(TEST_DATA_FILENAME)

    train_features = train_data_dict['x']
    train_sens_attrs = train_data_dict['a']
    train_labels = train_data_dict['y']

    test_features = test_data_dict['x']
    test_sens_attrs = test_data_dict['a']
    test_labels = test_data_dict['y']

    def data_stream(features, sens_attrs, labels):
        """Returns a generator of batches."""
        rng = npr.RandomState(seed)
        num_examples = len(features)
        num_batches = compute_num_batches(num_examples, batch_size)

        while True:
            perm = rng.permutation(num_examples)
            for i in range(num_batches):
                batch_idx = perm[i * batch_size:(i + 1) * batch_size]
                batch_features = features[batch_idx]
                batch_labels = labels[batch_idx]
                batch_sens_attrs = sens_attrs[batch_idx]
                yield batch_features, batch_sens_attrs, batch_labels

    train_batches = data_stream(train_features, train_sens_attrs, train_labels)
    test_batches = data_stream(test_features, test_sens_attrs, test_labels)

    return train_batches, test_batches
