from __future__ import print_function

import math
import numpy as np
import random
from scipy.spatial.distance import squareform, pdist
from skimage.util import img_as_float

# Setup
from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from skimage import io
from skimage import color

# from utils import visualize_mean_color_image

# %matplotlib inline
# plt.rcParams['figure.figsize'] = (15.0, 12.0)  # set default size of plots
# plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'

# for auto-reloading extenrnal modules
# %load_ext autoreload
# %autoreload 2

# Generate random data points for clustering

mean1 = [-1, 0]
cov1 = [[0.1, 0], [0, 0.1]]
X1 = np.random.multivariate_normal(mean1, cov1, 100)

# Cluster 2
mean2 = [0, 1]
cov2 = [[0.1, 0], [0, 0.1]]
X2 = np.random.multivariate_normal(mean2, cov2, 100)

# Cluster 3
mean3 = [1, 0]
cov3 = [[0.1, 0], [0, 0.1]]
X3 = np.random.multivariate_normal(mean3, cov3, 100)

# Cluster 4
mean4 = [0, -1]
cov4 = [[0.1, 0], [0, 0.1]]
X4 = np.random.multivariate_normal(mean4, cov4, 100)

# Merge two sets of data points
X = np.concatenate((X1, X2, X3, X4))


def kmeans(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    K-Means algorithm can be broken down into following steps:
        1. Randomly initialize cluster centers
        2. Assign each point to the closest center
        3. Compute new center of each cluster
        4. Stop if cluster assignments did not change
        5. Go to step 2

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N)
    d = np.zeros(k)
    for n in range(num_iters):
        # YOUR CODE HERE
        for i in range(0, N):
            for j in range(0, k):
                d[j] = np.linalg.norm(features[i] - centers[j])
            assignments[i] = np.argmin(d)
            # print(d)
        for i in range(0, k):
            sum = 0
            cnt = 0
            for j in np.argwhere(assignments == i):
                cnt += 1
                sum += features[j]
            centers[i] = sum / cnt
        # END YOUR CODE
    return assignments


# np.random.seed(0)
# start = time()
# assignments = kmeans(X, 4)
# end = time()

# kmeans_runtime = end - start

# print("kmeans running time: %f seconds." % kmeans_runtime)

# for i in range(4):
#     cluster_i = X[assignments == i]
#     print(cluster_i)
#     plt.scatter(cluster_i[:, 0], cluster_i[:, 1])

# plt.axis('equal')
# plt.show()


def kmeans_fast(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    This function makes use of numpy functions and broadcasting to speed up the
    first part(cluster assignment) of kmeans algorithm.

    Hints
    - You may find np.repeat and np.argmin useful

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N)

    for n in range(num_iters):
        # YOUR CODE HERE
        pass
        # END YOUR CODE

    return assignments


def hierarchical_clustering(features, k):
    """ Run the hierarchical agglomerative clustering algorithm.

    The algorithm is conceptually simple:

    Assign each point to its own cluster
    While the number of clusters is greater than k:
        Compute the distance between all pairs of clusters
        Merge the pair of clusters that are closest to each other

    We will use Euclidean distance to defeine distance between two clusters.

    Recomputing the centroids of all clusters and the distances between all
    pairs of centroids at each step of the loop would be very slow. Thankfully
    most of the distances and centroids remain the same in successive
    iterations of the outer loop; therefore we can speed up the computation by
    only recomputing the centroid and distances for the new merged cluster.

    Even with this trick, this algorithm will consume a lot of memory and run
    very slowly when clustering large set of points. In practice, you probably
    do not want to use this algorithm to cluster more than 10,000 points.

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Assign each point to its own cluster
    assignments = np.arange(N)
    centers = np.copy(features)
    n_clusters = N

    while n_clusters > k:
        # YOUR CODE HERE

        pass
        # END YOUR CODE

    return assignments


# Load and display image
img = io.imread('train.jpg')
H, W, C = img.shape

# plt.imshow(img)
# plt.axis('off')
# plt.show()
# Pixel-Level Features


def color_features(img):
    """ Represents a pixel by its color.

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    H, W, C = img.shape
    img = img_as_float(img)
    features = np.zeros((H * W, C))

    # YOUR CODE HERE
    cnt = 0
    for i in range(0, H):
        for j in range(0, W):
            features[cnt] = img[i][j]
            cnt += 1
            pass
        pass
    pass
    # END YOUR CODE

    return features


def visualize_mean_color_image(img, segments):

    img = img_as_float(img)
    k = np.max(segments) + 1
    mean_color_img = np.zeros(img.shape)

    for i in range(k.astype(int)):
        mean_color = np.mean(img[segments == i], axis=0)
        mean_color_img[segments == i] = mean_color

    plt.imshow(mean_color_img)
    plt.axis('off')
    plt.show()


# np.random.seed(0)

# features = color_features(img)

# # Sanity checks
# assert features.shape == (H * W, C),\
#     "Incorrect shape! Check your implementation."

# assert features.dtype == np.float,\
#     "dtype of color_features should be float."

# assignments = kmeans(features, 8)
# segments = assignments.reshape((H, W))

# # Display segmentation
# # plt.imshow(segments, cmap='viridis')
# # plt.axis('off')
# # plt.show()

# visualize_mean_color_image(img, segments)


def simple_descriptor(patch):
    """
    Describe the patch by normalizing the image values into a standard 
    normal distribution (having mean of 0 and standard deviation of 1) 
    and then flattening into a 1D array. 

    The normalization will make the descriptor more robust to change 
    in lighting condition.

    Hint:
        If a denominator is zero, divide by 1 instead.

    Args:
        patch: grayscale image patch of shape (h, w)

    Returns:
        feature: 1D array of shape (h * w)
    """
    feature = []
    # YOUR CODE HERE

    h = patch.shape
    feature = np.zeros(h)
    x = 0
    sum = np.sum(patch)
    ave = sum / (h)
    d = np.sum(pow(patch - ave, 2))
    s = pow(d, 0.5)
    feature = (patch - ave) / s

    # END YOUR CODE
    return feature


def color_position_features(img):
    """ Represents a pixel by its color and position.

    Combine pixel's RGB value and xy coordinates into a feature vector.
    i.e. for a pixel of color (r, g, b) located at position (x, y) in the
    image. its feature vector would be (r, g, b, x, y).
    Don't forget to normalize features.

    Hints
    - You may find np.mgrid and np.dstack useful
    - You may use np.mean and np.std

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C+2)
    """
    H, W, C = img.shape
    color = img_as_float(img)
    features = np.zeros((H * W, C + 2))

    # YOUR CODE HERE
    cnt = 0
    for i in range(0, H):
        for j in range(0, W):
            for m in range(0, C):
                features[cnt][m] = img[i][j][m]
                features[cnt][C] = i
                features[cnt][C + 1] = j
            cnt += 1
            pass
        pass
    pass
    for i in range(0, H * W):
        mean = np.mean(features[i])
        # d = np.sum(pow(features - mean, 2))
        # s = pow(d, 0.5)
        s = np.std(features[i])
        features[i] = (features[i] - mean) / s
    # END YOUR CODE

    return features


# np.random.seed(0)

# features = color_position_features(img)
# # print(features)
# # Sanity checks
# assert features.shape == (H * W, C + 2),\
#     "Incorrect shape! Check your implementation."

# assert features.dtype == np.float,\
#     "dtype of color_features should be float."

# assignments = kmeans(features, 8)
# segments = assignments.reshape((H, W))

# # Display segmentation
# plt.imshow(segments, cmap='viridis')
# plt.axis('off')
# plt.show()

def partial_x(img):
    """ Computes partial x-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W)
    Returns:
        out: x-derivative image
    """

    out = None

    # YOUR CODE HERE

    Hi, Wi = img.shape
    out = np.zeros((Hi, Wi))
    img = np.array(img)
    for i in range(0, Hi):
        for j in range(0, Wi):
            m = j
            if((j + 1) >= Wi):
                m = j - 1
                out[i][j] = (img[i][m + 1] - img[i][j - 1]) / 1
            elif((j - 1) < 0):
                m = j + 1
                out[i][j] = (img[i][j + 1] - img[i][m - 1]) / 1
            else:
                out[i][j] = (img[i][j + 1] - img[i][j - 1]) / 2

    # END YOUR CODE

    return out


def partial_y(img):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W)
    Returns:
        out: y-derivative image
    """

    out = None

    # YOUR CODE HERE
    Hi, Wi = img.shape
    out = np.zeros((Hi, Wi))
    img = np.array(img)
    for i in range(0, Hi):
        for j in range(0, Wi):
            m = i
            if((i + 1) >= Hi):
                m = i - 1
                out[i][j] = (img[m + 1][j] - img[i - 1][j]) / 1
            elif((i - 1) < 0):
                m = i + 1
                out[i][j] = (img[i + 1][j] - img[m - 1][j]) / 1
            else:
                # elif(((i + 1) >= 0)and ((i + 1) < Hi) and ((i - 1) >= 0)and((i - 1) < Hi)):
                out[i][j] = (img[i + 1][j] - img[i - 1][j]) / 2
    pass
    # END YOUR CODE

    return out


def my_features(img):
    """ Implement your own features

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    # YOUR CODE HERE
    img_gray = color.rgb2gray(img)
    H, W, C = img.shape
    # color = img_as_float(img)
    features = np.zeros((H * W, C + 2))

    # YOUR CODE HERE
    cnt1 = 0
    img_dx = partial_x(img_gray)

    img_dy = partial_y(img_gray)

    grad_mag = pow((pow(img_dx, 2) + pow(img_dy, 2)), (1 / 2))
    grad_theta = np.arctan2(img_dy, img_dx)

    for i in range(0, H):
        for j in range(0, W):
            for m in range(0, C):
                features[cnt1][m] = img[i][j][m]
            features[cnt1][C] = grad_mag[i][j]
            features[cnt1][C + 1] = grad_theta[i][j]
            cnt1 += 1
            pass
        pass
    pass
    # features[:, :, 0] = color_img[:, :, 0]
    # features[:, :, 1] = color_img[:, :, 1]
    # features[:, :, 0] = color_img[:, :, 2]

    # dx = [-1, 0, 1]
    # dy = [-1, 0, 1]

    # features[:, :, 3] = grad_mag;
    # features[:, :, 4] = grad_theta;

    # for i in range(0, H * W):
    #     mean = np.mean(features[i])
    #     # d = np.sum(pow(features - mean, 2))
    #     # s = pow(d, 0.5)
    #     s = np.std(features[i])
    #     features[i] = (features[i] - mean) / s
    # END YOUR CODE
    return features

# Feel free to experiment with different images
# and varying number of segments


# img = io.imread('train.jpg')
# num_segments = 8

# H, W, C = img.shape

# # Extract pixel-level features
# features = my_features(img)

# # Run clustering algorithm
# assignments = kmeans(features, num_segments)

# segments = assignments.reshape((H, W))

# # Display segmentation
# plt.imshow(segments, cmap='viridis')
# plt.axis('off')
# plt.show()

# Quantitative Evaluation


def compute_accuracy(mask_gt, mask):
    """ Compute the pixel-wise accuracy of a foreground-background segmentation
        given a ground truth segmentation.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        mask - The estimated foreground-background segmentation. A logical
            array of the same size and format as mask_gt.

    Returns:
        accuracy - The fraction of pixels where mask_gt and mask agree. A
            bigger number is better, where 1.0 indicates a perfect segmentation.
    """

    accuracy = None
    # YOUR CODE HERE
    pass
    # END YOUR CODE

    return accuracy


def evaluate_segmentation(mask_gt, segments):
    """ Compare the estimated segmentation with the ground truth.

    Note that 'mask_gt' is a binary mask, while 'segments' contain k segments. 
    This function compares each segment in 'segments' with the ground truth and
    outputs the accuracy of the best segment.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        segments - An array of the same size as mask_gt. The value of a pixel
            indicates the segment it belongs.

    Returns:
        best_accuracy - Accuracy of the best performing segment.
            0 <= accuracy <= 1, where 1.0 indicates a perfect segmentation.
    """

    num_segments = np.max(segments) + 1
    best_accuracy = 0

    # Compare each segment in 'segments' with the ground truth
    for i in range(num_segments):
        mask = (segments == i).astype(int)
        accuracy = compute_accuracy(mask_gt, mask)
        best_accuracy = max(accuracy, best_accuracy)

    return best_accuracy


a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = np.zeros((3, 2))
# for i in range(2):
b[:, :2] = a[:, : 2]

print(b)
