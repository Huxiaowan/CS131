# from __future__ import print_function, division

import numpy as np
import math
import random
from skimage import filters
from skimage.util.shape import view_as_blocks
from scipy.spatial.distance import cdist
from scipy.ndimage.filters import convolve

from utils import pad, unpad

from skimage.feature import corner_peaks
from skimage.io import imread
import matplotlib.pyplot as plt
from time import time
from utils import plot_matches
from utils import get_output_space, warp_image

# %matplotlib inline
plt.rcParams['figure.figsize'] = (15.0, 12.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# # for auto-reloading extenrnal modules
# %load_ext autoreload
# %autoreload 2


def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.

    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp

    Args:
        size: int of the size of output matrix
        sigma: float of sigma to calculate kernel

    Returns:
        kernel: numpy array of shape (size, size)
    """

    kernel = np.zeros((size, size))
    k = (size - 1) / 2
    m = 1 / (2 * np.pi * pow(sigma, 2))
    for i in range(0, size):
        for j in range(0, size):
            kernel[i][j] = m * np.exp(-(pow((i - k), 2) +
                                        pow((j - k), 2)) / (2 * pow(sigma, 2)))

    return kernel


a = gaussian_kernel(3, 0.8)
b = gaussian_kernel(3, 0.1)
c = a - b
print(c)


def gaussian_kernel_2(size, sigma):
    """ Implementation of Gaussian Kernel.

    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp

    Args:
        size: int of the size of output matrix
        sigma: float of sigma to calculate kernel

    Returns:
        kernel: numpy array of shape (size, size)
    """

    kernel = np.zeros((size, size))
    # k = (size - 1) / 2
    m = 1 / (2 * np.pi * pow(sigma, 2))
    for i in range(-1, size - 1):
        for j in range(-1, size - 1):
            kernel[i + 1][j + 1] = m * np.exp(-(pow(i, 2) +
                                                pow(j, 2)) / (2 * pow(sigma, 2)))
    return kernel


# print(gaussian_kernel_2(3, 1.4))


def harris_corners(img, window_size=3, k=0.04):
    """
    Compute Harris corner response map. Follow the math equation
    R=Det(M)-k(Trace(M)^2).

    Hint:
        You may use the function scipy.ndimage.filters.convolve, 
        which is already imported above

    Args:
        img: Grayscale image of shape (H, W)
        window_size: size of the window function
        k: sensitivity parameter

    Returns:
        response: Harris response image of shape (H, W)
    """

    H, W = img.shape
    window = np.ones((window_size, window_size))
    response = np.zeros((H, W))
    max = 0
    for i in range(0, H):
        for j in range(0, W):
            if (img[i][j] > max):
                max = img[i][j]

    for i in range(0, H):
        for j in range(0, W):
            img[i][j] = img[i][j] / max * 255

    dx = filters.sobel_v(img)
    dy = filters.sobel_h(img)
    # YOUR CODE HERE
    window = gaussian_kernel_2(window_size, 0.21)
    # window = ([[4, 12, 4], [12, 36, 12], [4, 12, 4]])
    c = convolve(dx * dy, window)
    a = convolve(pow(dx, 2), window)
    b = convolve(pow(dy, 2), window)

    for i in range(0, H):
        for j in range(0, W):
            response[i][j] = pow((a[i][j] * b[i][j]) - (c[i][j]
                                                        * c[i][j]), 2) - k * pow(a[i][j] + b[i][j], 2)

    # response = response * 10e20
    # max = 0
    # for i in range(0, H):
    #     for j in range(0, W):
    #         if (response[i][j] > max):
    #             max = response[i][j]

    # for i in range(0, H):
    #     for j in range(0, W):
    #         response[i][j] = response[i][j] / max * 255

    # print("hxw")

    return response


# img = imread('sudoku.png', as_grey=True)

# Compute Harris corner response
# response = harris_corners(img)
# Display corner response
# plt.subplot(1, 2, 1)
# plt.imshow(response)
# plt.axis('off')
# plt.title('Harris Corner Response')

# plt.subplot(1, 2, 2)
# plt.imshow(imread('solution_harris.png', as_grey=True))
# plt.axis('off')
# plt.title('Harris Corner Solution')

# plt.show()

# img1 = imread('uttower1.jpg', as_grey=True)
# img2 = imread('uttower2.jpg', as_grey=True)
# img3 = imread('hy.jpg', as_grey=True)

# plt.imshow(img3)
# plt.axis('off')
# plt.title('Harris Corner Response')
# plt.show()
# Detect keypoints in two images
# keypoints1 = corner_peaks(harris_corners(img1, window_size=3),
#                           threshold_rel=0.05,
#                           exclude_border=8)
# keypoints2 = corner_peaks(harris_corners(img2, window_size=3),
#                           threshold_rel=0.05,
#                           exclude_border=8)


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

    h, w = patch.shape
    feature = np.zeros(h * w)
    x = 0
    sum = np.sum(patch)
    ave = sum / (h * w)
    d = np.sum(pow(patch - ave, 2))
    s = pow(d, 0.5)
    patch = (patch - ave) / s
    for i in range(0, h):
        for j in range(0, w):
            feature[x] = patch[i][j]
            x += 1
            pass
        pass
    # END YOUR CODE
    return feature


def describe_keypoints(image, keypoints, desc_func, patch_size=16):
    """
    Args:
        image: grayscale image of shape (H, W)
        keypoints: 2D array containing a keypoint (y, x) in each row
        desc_func: function that takes in an image patch and outputs
            a 1D feature vector describing the patch
        patch_size: size of a square patch at each keypoint

    Returns:
        desc: array of features describing the keypoints
    """

    image.astype(np.float32)
    desc = []

    for i, kp in enumerate(keypoints):
        y, x = kp
        patch = image[y - (patch_size // 2):y + ((patch_size + 1) // 2),
                      x - (patch_size // 2):x + ((patch_size + 1) // 2)]
        desc.append(desc_func(patch))
    return np.array(desc)


def match_descriptors(desc1, desc2, threshold):
    """
    Match the feature descriptors by finding distances between them. A match is formed 
    when the distance to the closest vector is much smaller than the distance to the 
    second-closest, that is, the ratio of the distances should be smaller
    than the threshold. Return the matches as pairs of vector indices.

    Args:
        desc1: an array of shape (M, P) holding descriptors of size P about M keypoints
        desc2: an array of shape (N, P) holding descriptors of size P about N keypoints

    Returns:
        matches: an array of shape (Q, 2) where each row holds the indices of one pair 
        of matching descriptors
    """
    matches = []

    N = desc1.shape[0]
    dists = cdist(desc1, desc2)
    # print(dists)
    # YOUR CODE HERE
    x = 0
    matches = np.zeros((N, 2))
    for i in range(0, N):
        a = np.argmin(dists[i])
        b = np.argmax(dists[i])
        dists[i][a] = b
        c = np.argmin(dists[i])
        if(c == 0):
            c = 1
        if(a / c <= threshold):
            matches[x][0] = int(i)
            matches[x][1] = int(a)
            x += 1
        pass
    pass
    # END YOUR CODE
    matches = matches[:x, :]
    return matches


# patch_size = 5

# desc1 = describe_keypoints(img1, keypoints1,
#                            desc_func=simple_descriptor,
#                            patch_size=patch_size)
# desc2 = describe_keypoints(img2, keypoints2,
#                            desc_func=simple_descriptor,
#                            patch_size=patch_size)
# matches = match_descriptors(desc1, desc2, 0.7)
# print(matches)
# # Plot matches
# fig, ax = plt.subplots(1, 1, figsize=(15, 12))
# ax.axis('off')
# plot_matches(ax, img1, img2, keypoints1, keypoints2, matches)
# plt.show()
# plt.imshow(imread('solution_simple_descriptor.png'))
# plt.axis('off')
# plt.title('Matched Simple Descriptor Solution')
# plt.show()


def fit_affine_matrix(p1, p2):
    """ Fit affine matrix such that p2 * H = p1 

    Hint:
        You can use np.linalg.lstsq function to solve the problem. 

    Args:
        p1: an array of shape (M, P)
        p2: an array of shape (M, P)

    Return:
        H: a matrix of shape (P * P) that transform p2 to p1.
    """

    assert (p1.shape[0] == p2.shape[0]),\
        'Different number of points in p1 and p2'
    p1 = pad(p1)
    p2 = pad(p2)

    # YOUR CODE HERE
    H = np.zeros((p1.shape[1], p1.shape[1]))
    A = []
    A = np.linalg.lstsq(p2, p1)
    H = A[0]
    # print(H)
    pass
    # END YOUR CODE

    # Sometimes numerical issues cause least-squares to produce the last
    # column which is not exactly [0, 0, 1]
    H[:, 2] = np.array([0, 0, 1])
    # print(H)
    return H


# keypoints2 = keypoints2[:keypoints1.shape[0], :]
# a = np.array([[0.5, 0.1], [0.4, 0.2], [0.8, 0.2]])
# b = np.array([[0.3, -0.2], [-0.4, -0.9], [0.1, 0.1]])

# H = fit_affine_matrix(b, a)

# # Target output
# sol = np.array(
#     [[1.25, 2.5, 0.0],
#      [-5.75, -4.5, 0.0],
#      [0.25, -1.0, 1.0]]
# )
# print(H)
# error = np.sum((H - sol) ** 2)

# if error < 1e-20:
#     print('Implementation correct!')
# else:
#     print('There is something wrong.')

# Extract matched keypoints
# x = matches[:, 0]
# y = matches[:, 1]
# for i in range(0, matches.shape[0]):
#     p1[i] = matches[i][0]
#     p2[i] = matches[i][1]
#     pass

# p1 = keypoints1[matches[:, 0].astype(int)]
# print(p1.shape)
# p2 = keypoints2[matches[:, 1].astype(int)]
# # Find affine transformation matrix H that maps p2 to p1
# H = fit_affine_matrix(p1, p2)


def ransac(keypoints1, keypoints2, matches, n_iters=200, threshold=20):
    """
    Use RANSAC to find a robust affine transformation

        1. Select random set of matches
        2. Compute affine transformation matrix
        3. Compute inliers
        4. Keep the largest set of inliers
        5. Re-compute least-squares estimate on all of the inliers

    Args:
        keypoints1: M1 x 2 matrix, each row is a point
        keypoints2: M2 x 2 matrix, each row is a point
        matches: N x 2 matrix, each row represents a match
            [index of keypoint1, index of keypoint 2]
        n_iters: the number of iterations RANSAC will run
        threshold: the number of threshold to find inliers

    Returns:
        H: a robust estimation of affine transformation from keypoints2 to
        keypoints 1
    """
    N = matches.shape[0]
    n_samples = int(N * 0.2)

    matched1 = pad(keypoints1[matches[:, 0].astype(int)])
    matched2 = pad(keypoints2[matches[:, 1].astype(int)])
    max_inliers = np.zeros(N)
    n_inliers = 0

    # RANSAC iteration start
    # YOUR CODE HERE
    p1 = []
    p2 = []
    p = random.sample(matches, n_samples)
    for k in enumerate(p):
        p1.append(matched1[k.astype(int)])
        p2.append(matched2[k.astype(int)])
        pass
    H = np.linalg.lstsq(p1, p2)
    for kp in enumerate(matches):
        d = cdist(keypoints1[kp[0]] * H, keypoints2[kp[1]])
        if(d <= threshold):
            n_inliers += 1
            matches.append(kp)
    # dists = cdist(desc1, desc2)
    pass
    # END YOUR CODE
    return H, matches[max_inliers.astype(int)]


# H, robust_matches = ransac(keypoints1, keypoints2, matches, threshold=1)

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

    h, w = patch.shape
    feature = np.zeros(h * w)
    x = 0
    sum = np.sum(patch)
    ave = sum / (h * w)
    d = np.sum(pow(patch - ave, 2))
    s = pow(d, 0.5)
    patch = (patch - ave) / s
    for i in range(0, h):
        for j in range(0, w):
            feature[x] = patch[i][j]
            x += 1
            pass
        pass
    # END YOUR CODE
    return feature


def flat(patch):
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

    h, w = patch.shape
    feature = np.zeros(h * w)
    x = 0
    for i in range(0, h):
        for j in range(0, w):
            feature[x] = patch[i][j]
            x += 1
            pass
        pass
    # END YOUR CODE
    return feature


def hog_descriptor(patch, pixels_per_cell=(8, 8)):
    """
    Generating hog descriptor by the following steps:

    1. compute the gradient image in x and y (already done for you)
    2. compute gradient histograms
    3. normalize across block 
    4. flattening block into a feature vector

    Args:
        patch: grayscale image patch of shape (h, w)
        pixels_per_cell: size of a cell with shape (m, n)

    Returns:
        block: 1D array of shape ((h*w*n_bins)/(m*n))
    """
    assert (patch.shape[0] % pixels_per_cell[0] == 0),\
        'Heights of patch and cell do not match'
    assert (patch.shape[1] % pixels_per_cell[1] == 0),\
        'Widths of patch and cell do not match'

    n_bins = 9
    degrees_per_bin = 180 // n_bins

    Gx = filters.sobel_v(patch)
    Gy = filters.sobel_h(patch)

    # Unsigned gradients
    G = np.sqrt(Gx**2 + Gy**2)
    theta = (np.arctan2(Gy, Gx) * 180 / np.pi) % 180

    G_cells = view_as_blocks(G, block_shape=pixels_per_cell)
    theta_cells = view_as_blocks(theta, block_shape=pixels_per_cell)
    rows = G_cells.shape[0]
    cols = G_cells.shape[1]
    cells = np.zeros((rows, cols, n_bins))
    o = pixels_per_cell[0]
    p = pixels_per_cell[1]
    b = np.zeros((n_bins))
    x_theta = np.zeros((o, p))
    y_G = np.zeros((o, p))
    block = np.zeros((patch.shape[0] * patch.shape[1] * n_bins / o / p))
    block = []
    for i in range(0, rows):
        for j in range(0, cols):
            x_theta = flat(theta_cells[i][j])
            y_G = flat(G_cells[i][j])
            x_theta = (x_theta / 20).astype(int)
            for m in range(0, o * p):
                b[x_theta[m]] += 1 * y_G[m]
                pass
            block.append(b)

    block = np.array(block)
    block = simple_descriptor(block)
    # YOUR CODE HERE

    return block


img1 = imread('uttower1.jpg', as_grey=True)
img2 = imread('uttower2.jpg', as_grey=True)
keypoints1 = corner_peaks(harris_corners(img1, window_size=3),
                          threshold_rel=0.05,
                          exclude_border=8)
keypoints2 = corner_peaks(harris_corners(img2, window_size=3),
                          threshold_rel=0.05,
                          exclude_border=8)
# Extract features from the corners
desc1 = describe_keypoints(img1, keypoints1,
                           desc_func=hog_descriptor,
                           patch_size=16)
desc2 = describe_keypoints(img2, keypoints2,
                           desc_func=hog_descriptor,
                           patch_size=16)
# Match descriptors in image1 to those in image2
matches = match_descriptors(desc1, desc2, 0.7)

# Plot matches
fig, ax = plt.subplots(1, 1, figsize=(15, 12))
# ax.axis('off')
# plot_matches(ax, img1, img2, keypoints1, keypoints2, matches)
# plt.show()
# plt.imshow(imread('solution_hog.png'))
# plt.axis('off')
# plt.title('HOG descriptor Solution')
# plt.show()

p1 = keypoints1[matches[:, 0].astype(int)]
p2 = keypoints2[matches[:, 1].astype(int)]

# Find affine transformation matrix H that maps p2 to p1
H = fit_affine_matrix(p1, p2)

output_shape, offset = get_output_space(img1, [img2], [H])

# Warp images into output sapce
img1_warped = warp_image(img1, np.eye(3), output_shape, offset)
img1_mask = (img1_warped != -1)  # Mask == 1 inside the image
img1_warped[~img1_mask] = 0     # Return background values to 0

img2_warped = warp_image(img2, H, output_shape, offset)
img2_mask = (img2_warped != -1)  # Mask == 1 inside the image
img2_warped[~img2_mask] = 0     # Return background values to 0

# Plot warped images
plt.subplot(1, 2, 1)
plt.imshow(img1_warped)
plt.title('Image 1 warped')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img2_warped)
plt.title('Image 2 warped')
plt.axis('off')

plt.show()
merged = img1_warped + img2_warped

# Track the overlap by adding the masks together
overlap = (img1_mask * 1.0 +  # Multiply by 1.0 for bool -> float conversion
           img2_mask)

# Normalize through division by `overlap` - but ensure the minimum is 1
normalized = merged / np.maximum(overlap, 1)
plt.imshow(normalized)
plt.axis('off')
plt.show()

plt.imshow(imread('solution_hog_panorama.png'))
plt.axis('off')
plt.title('HOG Descriptor Panorama Solution')
plt.show()
