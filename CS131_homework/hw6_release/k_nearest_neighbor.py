import numpy as np
from scipy.spatial.distance import cdist

"""Utilities for downloading the face dataset.
"""

import os
import math
import numpy as np
from skimage import io
from skimage import img_as_float

from time import time
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from skimage import io

plt.rcParams['figure.figsize'] = (15.0, 12.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def load_dataset(data_dir, train=True, as_grey=False, shuffle=True):
    """ Load faces dataset

    The face dataset for CS131 assignment.
    The directory containing the dataset has the following structure:

        faces/
            train/
                angelina jolie/
                ...
            test/
                angelina jolie/
                ...

    Args:
        data_dir - Directory containing the face datset.
        train - If True, load training data. Load test data otherwise.
        as_grey - If True, open images as grayscale.
        shuffle - shuffle dataset

    Returns:
        X - array of N images (N, 64, 64, 3)
        y - array of class labels (N,)
        class_names - list of class names (string)
    """
    y = []
    X = []
    class_names = []

    if train:
        data_dir = os.path.join(data_dir, 'train')
    else:
        data_dir = os.path.join(data_dir, 'test')

    for i, cls in enumerate(sorted(os.listdir(data_dir))):
        for img_file in os.listdir(os.path.join(data_dir, cls)):
            img_path = os.path.join(data_dir, cls, img_file)
            img = img_as_float(io.imread(img_path, as_grey=as_grey))
            X.append(img)
            y.append(i)
        class_names.append(cls)
    # Convert list of imgs and labels into array
    X = np.array(X)
    y = np.array(y)

    if shuffle:
        idxs = np.arange(len(y))
        np.random.shuffle(idxs)
        X = X[idxs]
        y = y[idxs]

    return np.array(X), np.array(y), class_names


X_train, y_train, classes_train = load_dataset(
    'faces', train=True, as_grey=True)
X_test, y_test, classes_test = load_dataset('faces', train=False, as_grey=True)

assert classes_train == classes_test
classes = classes_train

print('Class names:', classes)
print('Training data shape:', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape:', X_test.shape)
print('Test labels shape: ', y_test.shape)

print(y_train[2])
# Visualize some examples from the dataset.
# We show a few examples of training images from each class.
num_classes = len(classes)
samples_per_class = 10
# for y, cls in enumerate(classes):
#     idxs = np.flatnonzero(y_train == y)
#     idxs = np.random.choice(idxs, samples_per_class, replace=False)
#     for i, idx in enumerate(idxs):
#         plt_idx = i * num_classes + y + 1
#         plt.subplot(samples_per_class, num_classes, plt_idx)
#         plt.imshow(X_train[idx])
#         plt.axis('off')
#         if i == 0:
#             plt.title(y)
# plt.show()

# Flatten the image data into rows
# we now have one 4096 dimensional featue vector for each example
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)
# print(y_test, y_train)


def compute_distances(X1, X2):
    """Compute the L2 distance between each point in X1 and each point in X2.
    It's possible to vectorize the computation entirely (i.e. not use any loop).

    Args:
        X1: numpy array of shape (M, D) normalized along axis=1
        X2: numpy array of shape (N, D) normalized along axis=1

    Returns:
        dists: numpy array of shape (M, N) containing the L2 distances.
    """
    M = X1.shape[0]
    N = X2.shape[0]
    assert X1.shape[1] == X2.shape[1]

    dists = np.zeros((M, N))

    # YOUR CODE HERE
    # Compute the L2 distance between all X1 features and X2 features.
    # Don't use any for loop, and store the result in dists.
    #
    # You should implement this function using only basic array operations;
    # in particular you should not use functions from scipy.
    #
    # HINT: Try to formulate the l2 distance using matrix multiplication
    dists = cdist(X1, X2)

    pass
    # END YOUR CODE

    assert dists.shape == (
        M, N), "dists should have shape (M, N), got %s" % dists.shape

    return dists


# Step 1: compute the distances between all features from X_train and from X_test
dists = compute_distances(X_test, X_train)
assert dists.shape == (160, 800)
print("dists shape:", dists.shape)


def predict_labels(dists, y_train, k=1):
    """Given a matrix of distances `dists` between test points and training points,
    predict a label for each test point based on the `k` nearest neighbors.

    Args:
        dists: A numpy array of shape (num_test, num_train) where dists[i, j] gives
               the distance betwen the ith test point and the jth training point.

    Returns:
        y_pred: A numpy array of shape (num_test,) containing predicted labels for the
                test data, where y[i] is the predicted label for the test point X[i].
    """
    num_test, num_train = dists.shape
    y_pred = np.zeros(num_test, dtype=np.int)
    for i in range(num_test):
        # A list of length k storing the labels of the k nearest neighbors to
        # the ith test point.
        closest_y = []
        # Use the distance matrix to find the k nearest neighbors of the ith
        # testing point, and use self.y_train to find the labels of these
        # neighbors. Store these labels in closest_y.
        # Hint: Look up the function numpy.argsort.

        # Now that you have found the labels of the k nearest neighbors, you
        # need to find the most common label in the list closest_y of labels.
        # Store this label in y_pred[i]. Break ties by choosing the smaller
        # label.

        # YOUR CODE HERE
        dists[i] = np.argsort(dists[i])
        for m in range(0, k):
            a = y_train[dists[i][m]]
            closest_y.append(a)
        cnt = 0
        cnt_max = 0
        max_label = 0
        for n in closest_y:
            cnt = closest_y.count(n)
            if(cnt > cnt_max):
                cnt_max = cnt
                max_label = n
            pass
        y_pred[i] = max_label
        # END YOUR CODE

    return y_pred


# We use k = 1 (which corresponds to only taking the nearest neighbor to decide)
y_test_pred = predict_labels(dists, y_train, k=1)

# Compute and print the fraction of correctly predicted examples
num_test = y_test.shape[0]
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' %
      (num_correct, num_test, accuracy))


def split_folds(X_train, y_train, num_folds):
    """Split up the training data into `num_folds` folds.

    The goal of the functions is to return training sets (features and labels) along with
    corresponding validation sets. In each fold, the validation set will represent (1/num_folds)
    of the data while the training set represent (num_folds-1)/num_folds.
    If num_folds=5, this corresponds to a 80% / 20% split.

    For instance, if X_train = [0, 1, 2, 3, 4, 5], and we want three folds, the output will be:
        X_trains = [[2, 3, 4, 5],
                    [0, 1, 4, 5],
                    [0, 1, 2, 3]]
        X_vals = [[0, 1],
                  [2, 3],
                  [4, 5]]

    Args:
        X_train: numpy array of shape (N, D) containing N examples with D features each
        y_train: numpy array of shape (N,) containing the label of each example
        num_folds: number of folds to split the data into

    returns:
        X_trains: numpy array of shape (num_folds, train_size * (num_folds-1) / num_folds, D)
        y_trains: numpy array of shape (num_folds, train_size * (num_folds-1) / num_folds)
        X_vals: numpy array of shape (num_folds, train_size / num_folds, D)
        y_vals: numpy array of shape (num_folds, train_size / num_folds)

    """
    assert X_train.shape[0] == y_train.shape[0]

    validation_size = X_train.shape[0] // num_folds
    training_size = X_train.shape[0] - validation_size

    X_trains = np.zeros((num_folds, training_size, X_train.shape[1]))
    y_trains = np.zeros((num_folds, training_size), dtype=np.int)
    X_vals = np.zeros((num_folds, validation_size, X_train.shape[1]))
    y_vals = np.zeros((num_folds, validation_size), dtype=np.int)

    # YOUR CODE HERE
    # Hint: You can use the numpy array_split function.
    for i in range(0, num_folds):
        X_vals[i] = np.array_split(
            X_train, [validation_size, validation_size + training_size])[0]
        X_trains[i] = np.array_split(
            X_train, [validation_size, validation_size + training_size])[1]
        y_vals[i] = np.array_split(
            y_train, [validation_size, validation_size + training_size])[0]
        y_trains[i] = np.array_split(
            y_train, [validation_size, validation_size + training_size])[1]

    # END YOUR CODE

    return X_trains, y_trains, X_vals, y_vals


num_folds = 5

X_trains, y_trains, X_vals, y_vals = split_folds(X_train, y_train, num_folds)

assert X_trains.shape == (5, 640, 4096)
assert y_trains.shape == (5, 640)
assert X_vals.shape == (5, 160, 4096)
assert y_vals.shape == (5, 160)

# Step 3: Measure the mean accuracy for each value of `k`

# List of k to choose from
k_choices = list(range(5, 101, 5))

# Dictionnary mapping k values to accuracies
# For each k value, we will have `num_folds` accuracies to compute
# k_to_accuracies[1] will be for instance [0.22, 0.23, 0.19, 0.25, 0.20] for 5 folds
k_to_accuracies = {}

for k in k_choices:
    print("Running for k=%d" % k)
    accuracies = []
    for i in range(num_folds):
        # Make predictions
        fold_dists = compute_distances(X_vals[i], X_trains[i])
        y_pred = predict_labels(fold_dists, y_trains[i], k)

        # Compute and print the fraction of correctly predicted examples
        num_correct = np.sum(y_pred == y_vals[i])
        accuracy = float(num_correct) / len(y_vals[i])
        accuracies.append(accuracy)

    k_to_accuracies[k] = accuracies

# plot the raw observations
for k in k_choices:
    accuracies = k_to_accuracies[k]
    plt.scatter([k] * len(accuracies), accuracies)

# plot the trend line with error bars that correspond to standard deviation
accuracies_mean = np.array([np.mean(v)
                            for k, v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v)
                           for k, v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()
