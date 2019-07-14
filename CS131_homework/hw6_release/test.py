import numpy as np
import scipy
import scipy.linalg


"""Utilities for downloading the face dataset.
"""

import os


from skimage import io
from skimage import img_as_float

from time import time
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib import rc


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


class PCA(object):
    """Class implementing Principal component analysis (PCA).

    Steps to perform PCA on a matrix of features X:
        1. Fit the training data using method `fit` (either with eigen decomposition of SVD)
        2. Project X into a lower dimensional space using method `transform`
        3. Optionally reconstruct the original X (as best as possible) using method `reconstruct`
    """

    def __init__(self):
        self.W_pca = None
        self.mean = None

    def fit(self, X, method='svd'):
        """Fit the training data X using the chosen method.

        Will store the projection matrix in self.W_pca and the mean of the data in self.mean

        Args:
            X: numpy array of shape (N, D). Each of the N rows represent a data point.
               Each data point contains D features.
            method: Method to solve PCA. Must be one of 'svd' or 'eigen'.
        """
        _, D = X.shape
        self.mean = None   # empirical mean, has shape (D,)
        X_centered = None  # zero-centered data

        # YOUR CODE HERE
        # 1. Compute the mean and store it in self.mean
        # 2. Apply either method to `X_centered`

        pass
        # END YOUR CODE

        # Make sure that X_centered has mean zero
        assert np.allclose(X_centered.mean(), 0.0)

        # Make sure that self.mean is set and has the right shape
        assert self.mean is not None and self.mean.shape == (D,)

        # Make sure that self.W_pca is set and has the right shape
        assert self.W_pca is not None and self.W_pca.shape == (D, D)

        # Each column of `self.W_pca` should have norm 1 (each one is an eigenvector)
        for i in range(D):
            assert np.allclose(np.linalg.norm(self.W_pca[:, i]), 1.0)

    def _eigen_decomp(self, X):
        """Performs eigendecompostion of feature covariance matrix.

        Args:
            X: Zero-centered data array, each ROW containing a data point.
               Numpy array of shape (N, D).

        Returns:
            e_vecs: Eigenvectors of covariance matrix of X. Eigenvectors are
                    sorted in descending order of corresponding eigenvalues. Each
                    column contains an eigenvector. Numpy array of shape (D, D).
            e_vals: Eigenvalues of covariance matrix of X. Eigenvalues are
                    sorted in descending order. Numpy array of shape (D,).
        """
        N, D = X.shape
        e_vecs = None
        e_vals = None
        # YOUR CODE HERE
        # Steps:
        #     1. compute the covariance matrix of X, of shape (D, D)
        #     2. compute the eigenvalues and eigenvectors of the covariance matrix
        #     3. Sort both of them in decreasing order (ex: 1.0 > 0.5 > 0.0 > -0.2 > -1.2)
        X = np.dot(X.T, X)
        u, s, v = np.linalg.svd(X)
        e_vecs = v.T
        e_vals = s.T
        pass
        # END YOUR CODE

        # Check the output shapes
        assert e_vals.shape == (D,)
        assert e_vecs.shape == (D, D)

        return e_vecs, e_vals

    def _svd(self, X):
        """Performs Singular Value Decomposition (SVD) of X.

        Args:
            X: Zero-centered data array, each ROW containing a data point.
                Numpy array of shape (N, D).
        Returns:
            vecs: right singular vectors. Numpy array of shape (D, D)
            vals: singular values. Numpy array of shape (K,) where K = min(N, D)
        """
        vecs = None  # shape (D, D)
        N, D = X.shape
        vals = None  # shape (K,)
        # YOUR CODE HERE
        # Here, compute the SVD of X
        # Make sure to return vecs as the matrix of vectors where each column is a singular vector
        # X = np.dot(X.T, X)
        u, s, v = np.linalg.svd(X)
        vecs = v.T
        vals = s.T
        pass
        # END YOUR CODE
        assert vecs.shape == (D, D)
        K = min(N, D)
        assert vals.shape == (K,)

        return vecs, vals

    def transform(self, X, n_components):
        """Center and project X onto a lower dimensional space using self.W_pca.

        Args:
            X: numpy array of shape (N, D). Each row is an example with D features.
            n_components: number of principal components..

        Returns:
            X_proj: numpy array of shape (N, n_components).
        """
        N, D = X.shape
        X_proj = np.zeros((N, n_components))
        # YOUR CODE HERE
        # We need to modify X in two steps:
        #     1. first substract the mean stored during `fit`
        #     2. then project onto a subspace of dimension `n_components` using `self.W_pca`
        # e_vecs, e_vals = pca._eigen_decomp(X - X.mean(axis=0))
        X = X - X.mean(axis=0)

        # X = np.dot(X.T, X)
        # X = np.dot(X, X.T)
        u, s, v = np.linalg.svd(X)
        e_vecs = v.T
        e_vals = s.T

        # e_vals[n_components: -1] = 0
        e_vals = np.diag(e_vals)
        D_D = np.dot(e_vecs, e_vals)

        X_proj[:, : n_components] = D_D[:, : n_components]

        # END YOUR CODE

        assert X_proj.shape == (
            D, n_components), "X_proj doesn't have the right shape"

        return X_proj

    def reconstruct(self, X_proj):
        """Do the exact opposite of method `transform`: try to reconstruct the original features.

        Given the X_proj of shape (N, n_components) obtained from the output of `transform`,
        we try to reconstruct the original X.

        Args:
            X_proj: numpy array of shape (N, n_components). Each row is an example with D features.

        Returns:
            X: numpy array of shape (N, D).
        """
        N, n_components = X_proj.shape
        X = None

        # YOUR CODE HERE
        # Steps:
        #     1. project back onto the original space of dimension D
        #     2. add the mean that we substracted in `transform`
        D = X_train.shape[1]

        R = np.zeros((D, n_components))

        X = X_train - X_train.mean(axis=0)
        # X = np.dot(X.T, X)
        u, s, v = np.linalg.svd(X)

        R[:, : n_components] = v[:, : n_components]
        X = np.dot(X_proj, R.T)
        X = X + X_train.mean(axis=0)
        pass
        # END YOUR CODE

        return X


pca = PCA()


# Plot reconstruction errors for different k
N = X_train.shape[0]
d = X_train.shape[1]

ns = range(1, d, 100)
errors = []

for n_components in ns:
    X_proj = pca.transform(X_train, n_components)
    X_rec = pca.reconstruct(X_proj)

    # Compute reconstruction error
    error = np.mean((X_rec - X_train) ** 2)
    errors.append(error)

plt.plot(ns, errors)
plt.xlabel('Number of Components')
plt.ylabel('Reconstruction Error')
plt.show()

# n_components = 2

# X_proj = pca.transform(X_train, n_components)
# e_vecs, e_vals = pca._eigen_decomp(X_train - X_train.mean(axis=0))
# u, s = pca._eigen_decomp(X_train - X_train.mean(axis=0))
# # Plot reconstruction errors for different k
# N = X_train.shape[0]
# d = X_train.shape[1]

# # Plot captured variance
# ns = range(0, N, 100)
# var_cap = []

# for n in ns:
#     var_cap.append(np.sum(s[:n] ** 2) / np.sum(s ** 2))

# plt.plot(ns, var_cap)
# plt.xlabel('Number of Components')
# plt.ylabel('Variance Captured')
# plt.show()

# for i in range(10):
#     plt.subplot(1, 10, i + 1)
#     plt.imshow(e_vecs[:, i].reshape(64, 64))
#     plt.title("%.2f" % e_vals[i])
# plt.show()

# # Plot the top two principal components
# for y in np.unique(y_train):
#     plt.scatter(X_proj[y_train == y, 0],
#                 X_proj[y_train == y, 1], label=classes[y])

# plt.xlabel('1st component')
# plt.ylabel('2nd component')
# plt.legend()
# plt.show()
# e_vecs, e_vals = pca._eigen_decomp(X_train - X_train.mean(axis=0))
# u, s = pca._svd(X_train - X_train.mean(axis=0))

# Reconstruct data with principal components
# n_components = 100  # Experiment with different number of components.
# X_proj = pca.transform(X_train, n_components)
# X_rec = pca.reconstruct(X_proj)

# print(X_rec.shape)
# print(classes)

# # Visualize reconstructed faces
# samples_per_class = 10
# for y, cls in enumerate(classes):
#     idxs = np.flatnonzero(y_train == y)
#     idxs = np.random.choice(idxs, samples_per_class, replace=False)
#     for i, idx in enumerate(idxs):
#         plt_idx = i * num_classes + y + 1
#         plt.subplot(samples_per_class, num_classes, plt_idx)
#         plt.imshow((X_rec[idx]).reshape((64, 64)))
#         plt.axis('off')
#         if i == 0:
#             plt.title(y)
# plt.show()


class LDA(object):
    """Class implementing Principal component analysis (PCA).

    Steps to perform PCA on a matrix of features X:
        1. Fit the training data using method `fit` (either with eigen decomposition of SVD)
        2. Project X into a lower dimensional space using method `transform`
        3. Optionally reconstruct the original X (as best as possible) using method `reconstruct`
    """

    def __init__(self):
        self.W_lda = None

    def fit(self, X, y):
        """Fit the training data `X` using the labels `y`.

        Will store the projection matrix in `self.W_lda`.

        Args:
            X: numpy array of shape (N, D). Each of the N rows represent a data point.
               Each data point contains D features.
            y: numpy array of shape (N,) containing labels of examples in X
        """
        N, D = X.shape

        scatter_between = self._between_class_scatter(X, y)
        scatter_within = self._within_class_scatter(X, y)

        e_vecs = None

        # YOUR CODE HERE
        # Solve generalized eigenvalue problem for matrices `scatter_between` and `scatter_within`
        # Use `scipy.linalg.eig` instead of numpy's eigenvalue solver.
        # Don't forget to sort the values and vectors in descending order.
        pass
        # END YOUR CODE

        self.W_lda = e_vecs

        # Check that the shape of `self.W_lda` is correct
        assert self.W_lda.shape == (D, D)

        # Each column of `self.W_lda` should have norm 1 (each one is an eigenvector)
        for i in range(D):
            assert np.allclose(np.linalg.norm(self.W_lda[:, i]), 1.0)

    def _within_class_scatter(self, X, y):
        """Compute the covariance matrix of each class, and sum over the classes.

        For every label i, we have:
            - X_i: matrix of examples with labels i
            - S_i: covariance matrix of X_i (per class covariance matrix for class i)
        The formula for covariance matrix is: X_centered^T X_centered
            where X_centered is the matrix X with mean 0 for each feature.

        Our result `scatter_within` is the sum of all the `S_i`

        Args:
            X: numpy array of shape (N, D) containing N examples with D features each
            y: numpy array of shape (N,), labels of examples in X

        Returns:
            scatter_within: numpy array of shape (D, D), sum of covariance matrices of each label
        """
        _, D = X.shape
        assert X.shape[0] == y.shape[0]
        scatter_within = np.zeros((D, D))

        for i in np.unique(y):
            # YOUR CODE HERE
            # Get the covariance matrix for class i, and add it to scatter_within
            pass
            # END YOUR CODE

        return scatter_within

    def _between_class_scatter(self, X, y):
        """Compute the covariance matrix as if each class is at its mean.

        For every label i, we have:
            - X_i: matrix of examples with labels i
            - mu_i: mean of X_i.

        Our result `scatter_between` is the covariance matrix of X where we replaced every
        example labeled i with mu_i.

        Args:
            X: numpy array of shape (N, D) containing N examples with D features each
            y: numpy array of shape (N,), labels of examples in X

        Returns:
            scatter_between: numpy array of shape (D, D)
        """
        _, D = X.shape
        assert X.shape[0] == y.shape[0]
        scatter_between = np.zeros((D, D))

        mu = X.mean(axis=0)
        for i in np.unique(y):
            # YOUR CODE HERE
            pass
            # END YOUR CODE

        return scatter_between

    def transform(self, X, n_components):
        """Project X onto a lower dimensional space using self.W_pca.

        Args:
            X: numpy array of shape (N, D). Each row is an example with D features.
            n_components: number of principal components..

        Returns:
            X_proj: numpy array of shape (N, n_components).
        """
        N, _ = X.shape
        X_proj = None
        # YOUR CODE HERE
        # project onto a subspace of dimension `n_components` using `self.W_lda`
        pass
        # END YOUR CODE

        assert X_proj.shape == (
            N, n_components), "X_proj doesn't have the right shape"

        return X_proj
