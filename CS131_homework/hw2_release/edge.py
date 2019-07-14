from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from time import time
from skimage import io
import math
from os import listdir
from itertools import product


def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges  the boundary.
    # pad_width0 = Hk // 2
    # pad_width1 = Wk // 2
    pad_width0 = Hk - 1
    pad_width1 = Wk - 1
    pad_width = ((pad_width0, pad_width0), (pad_width1, pad_width1))
    padded = np.pad(image, pad_width, mode='edge')

    # YOUR CODE HERE
    kernel = np.fliplr(kernel)
    kernel = np.flipud(kernel)
    out1 = np.zeros((Hi + Hk - 1, Wi + Wk - 1))
    for i in range(0, Hi + Hk - 1):
        for j in range(0, Wi + Wk - 1):
            temp = 0.0
            for m in range(0, Hk):
                for n in range(0, Wk):
                    temp += kernel[m][n] * padded[m + i][n + j]
            out1[i][j] = temp

    for i in range(0, Hi):
        for j in range(0, Wi):
            out[i][j] = out1[i + (Hk - 1) / 2][j + (Wk - 1) / 2]
            pass
        pass
    pass
    # END YOUR CODE

    return out


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
                out[i][j] = (img[i][m + 1] - img[i][j - 1]) / 2
            elif((j - 1) < 0):
                m = j + 1
                out[i][j] = (img[i][j + 1] - img[i][m - 1]) / 2
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
                out[i][j] = (img[m + 1][j] - img[i - 1][j]) / 2
            elif((i - 1) < 0):
                m = i + 1
                out[i][j] = (img[i + 1][j] - img[m - 1][j]) / 2
            else:
                # elif(((i + 1) >= 0)and ((i + 1) < Hi) and ((i - 1) >= 0)and((i - 1) < Hi)):
                out[i][j] = (img[i + 1][j] - img[i - 1][j]) / 2
    pass
    # END YOUR CODE

    return out


def gradient(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W)

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W)
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W)
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)

    # YOUR CODE HERE
    Hi, Wi = img.shape
    Gx = partial_x(img)
    Gy = partial_y(img)
    for i in range(0, Hi):
        for j in range(0, Wi):
            G[i][j] = pow((pow(Gx[i][j], 2) + pow(Gy[i][j], 2)), 0.5)
            if(Gx[i][j] != 0):
                theta[i][j] = (math.atan(Gy[i][j] / Gx[i][j])) / np.pi * 360
                if(theta[i][j] < 0):
                    theta[i][j] = theta[i][j] + 360

    # END YOUR CODE

    return G, theta


def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).

    Args:
        G: gradient magnitude image with shape of (H, W)
        theta: direction of gradients with shape of (H, W)

    Returns:
        out: non-maxima suppressed image
    """
    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45

    # BEGIN YOUR CODE
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            if(theta[i][j] == 0 or theta[i][j] == 180 or theta[i][j] == 360):
                m0 = G[i][j]
                m1 = G[i][j - 1]
                m2 = G[i][j + 1]
                if ((m0 < m1) or (m0 < m2)):
                    G[i][j] = 0
            if(theta[i][j] == 45 or theta[i][j] == 225):
                m0 = G[i][j]
                m1 = G[i - 1][j + 1]
                m2 = G[i + 1][j - 1]
                if ((m0 < m1) or (m0 < m2)):
                    G[i][j] = 0
            if(theta[i][j] == 90 or theta[i][j] == 270):
                m0 = G[i][j]
                m1 = G[i + 1][j]
                m2 = G[i - 1][j]
                if ((m0 < m1) or (m0 < m2)):
                    G[i][j] = 0
            if(theta[i][j] == 135 or theta[i][j] == 315):
                m0 = G[i][j]
                m1 = G[i - 1][j - 1]
                m2 = G[i + 1][j + 1]
                if ((m0 < m1) or (m0 < m2)):
                    G[i][j] = 0

    for i in range(0, H):
        for j in range(0, W):
            out[i][j] = G[i][j]

    # END YOUR CODE

    return out


# plt.imshow(nms)
# plt.title('Non-maximum suppressed')
# plt.axis('off')
# plt.show()


def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response
        high: high threshold(float) for strong edges
        low: low threshold(float) for weak edges

    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edeges are the pixels with the values above
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values below the
            higher threshould and above the lower threshold.
    """

    strong_edges = np.zeros(img.shape)
    weak_edges = np.zeros(img.shape)

    # YOUR CODE HERE
    H, W = img.shape
    for i in range(0, H):
        for j in range(0, W):
            if (img[i][j] >= high):
                strong_edges[i][j] = 1
            elif(img[i][j] >= low):
                # weak_edges[i][j] = img[i][j]
                weak_edges[i][j] = 1
                pass
            pass
        pass
    pass
    # END YOUR CODE

    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x)

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)

    Args:
        y, x: location of the pixel
        H, W: size of the image
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)]
    """
    neighbors = []

    for i in (y - 1, y, y + 1):
        for j in (x - 1, x, x + 1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))

    return neighbors


def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.

    Iterate over each pixel in strong_edges and perform breadth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).

    Args:
        strong_edges: binary image of shape (H, W)
        weak_edges: binary image of shape (H, W)
    Returns:
        edges: numpy array of shape(H, W)
    """

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.zeros((H, W))

    # YOUR CODE HERE

    for i in range(0, H):
        for j in range(0, W):
            for i1 in range(0, H):
                for j1 in range(0, W):
                    if (strong_edges[i1][j1] == 0):
                        z = get_neighbors(i, j, H, W)
                        g = 0
                        for m in range(0, len(z)):
                            w = z[m]
                            if(strong_edges[w] == 0):
                                g += 1
                        if (g == len(z)):
                            for m in range(0, len(z)):
                                if(weak_edges[z[m]] != 0):
                                    strong_edges[i1][j1] = 1
                                    i1 = z[m][0]
                                    j1 = z[m][1]
                                    break
    edges = strong_edges
    return edges


kernel_size = 5
sigma = 1.4

# Load image
img = io.imread('iguana.png', as_grey=True)

# Define 5x5 Gaussian kernel with std = sigma
kernel = gaussian_kernel(kernel_size, sigma)

# Convolve image with kernel to achieve smoothed effect
smoothed = conv(img, kernel)

G, theta = gradient(smoothed)
nms = non_maximum_suppression(G, theta)

low_threshold = 0.02
high_threshold = 0.03

strong_edges, weak_edges = double_thresholding(
    nms, high_threshold, low_threshold)

# indices = np.stack(np.nonzero(strong_edges)).T
# x = get_neighbors(1, 1, 10, 10)
# print(x[5])
# y = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
# z = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]
# g = 0
# for m in range(0, 8):
#     if(y[z[m]] == 0):
#         g += 1
# print(g)

# test_strong = np.array(
#     [[1, 0, 0, 0],
#      [0, 0, 0, 0],
#      [0, 0, 0, 0],
#      [0, 0, 0, 1]]
# )

# test_weak = np.array(
#     [[0, 0, 0, 1],
#      [0, 1, 0, 0],
#      [1, 0, 0, 0],
#      [0, 0, 1, 0]]
# )

# test_linked = link_edges(test_strong, test_weak)

# plt.subplot(1, 3, 1)
# plt.imshow(test_strong)
# plt.title('Strong edges')

# plt.subplot(1, 3, 2)
# plt.imshow(test_weak)
# plt.title('Weak edges')

# plt.subplot(1, 3, 3)
# plt.imshow(test_linked)
# plt.title('Linked edges')
# plt.show()

edges = link_edges(strong_edges, weak_edges)

plt.imshow(edges)
plt.axis('off')
plt.show()


def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """ Implement canny edge detector by calling functions above.

    Args:
        img: binary image of shape (H, W)
        kernel_size: int of size for kernel matrix
        sigma: float for calculating kernel
        high: high threshold for strong edges
        low: low threashold for weak edges
    Returns:
        edge: numpy array of shape(H, W)
    """

    # Define 5x5 Gaussian kernel with std = sigma
    kernel = gaussian_kernel(kernel_size, sigma)

    # Convolve image with kernel to achieve smoothed effect
    smoothed = conv(img, kernel)

    G, theta = gradient(smoothed)

    nms = non_maximum_suppression(G, theta)

    strong_edges, weak_edges = double_thresholding(nms, high, low)
    edge = strong_edges + weak_edges
    pass
    # END YOUR CODE

    return edge


# sigmas = [1, 1.1, 1.2, 1.3, 1.4, 1.5]
# highs = [0.03, 0.035, 0.04, 0.045, 0.05, 0.055]
# lows = [0.02, 0.022, 0.024, 0.026, 0.028, 0.03]

# for sigma, high, low in product(sigmas, highs, lows):

#     print("sigma={}, high={}, low={}".format(sigma, high, low))
#     n_detected = 0.0
#     n_gt = 0.0
#     n_correct = 0.0

#     for img_file in listdir('images/objects'):
#         img = io.imread('images/objects/' + img_file, as_grey=True)
#         gt = io.imread('images/gt/' + img_file + '.gtf.pgm', as_grey=True)

#         mask = (gt != 5)  # 'don't' care region
#         gt = (gt == 0)  # binary image of GT edges

#         edges = canny(img, kernel_size=5, sigma=sigma, high=high, low=low)
#         edges = edges * mask

#         n_detected += np.sum(edges)
#         n_gt += np.sum(gt)
#         n_correct += np.sum(edges * gt)

#     p_total = n_correct / n_detected
#     r_total = n_correct / n_gt
#     f1 = 2 * (p_total * r_total) / (p_total + r_total)
#     print('Total precision={:.4f}, Total recall={:.4f}'.format(
#         p_total, r_total))
#     print('F1 score={:.4f}'.format(f1))


def hough_transform(img):
    """ Transform points in the input image into Hough space.

    Use the parameterization:
        rho = x * cos(theta) + y * sin(theta)
    to transform a point (x,y) to a sine-like function in Hough space.

    Args:
        img: binary image of shape (H, W)

    Returns:
        accumulator: numpy array of shape (m, n)
        rhos: numpy array of shape (m, )
        thetas: numpy array of shape (n, )
    """
    # Set rho and theta ranges
    W, H = img.shape
    diag_len = int(np.ceil(np.sqrt(W * W + H * H)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0 + 1)
    thetas = np.deg2rad(np.arange(-90.0, 90.0))
    # Cache some reusable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Initialize accumulator in the Hough space
    accumulator = np.zeros((2 * diag_len + 1, num_thetas), dtype=np.uint64)
    ys, xs = np.nonzero(img)

    # Transform each point (x, y) in image
    # Find rho corresponding to values in thetas
    # and increment the accumulator in the corresponding coordiate.
    # YOUR CODE HERE
    num_point = len(ys)
    for i in range(0, num_point):
        for m in range(0, num_thetas):
            p = cos_t[m] * xs[i] + sin_t[m] * ys[i] + 1102
            accumulator[p][m] += 1
    # END YOUR CODE

    return accumulator, rhos, thetas


# img = io.imread('road.jpg', as_grey=True)
# edges = canny(img, kernel_size=5, sigma=1.4, high=0.03, low=0.02)

# H, W = img.shape

# # Generate mask for ROI (Region of Interest)
# mask = np.zeros((H, W))
# for i in range(H):
#     for j in range(W):
#         if i > (H / W) * j and i > -(H / W) * j + H:
#             mask[i, j] = 1

# # Extract edges in ROI
# roi = edges * mask
# acc, rhos, thetas = hough_transform(roi)
# # Coordinates for right lane
# xs_right = []
# ys_right = []

# # Coordinates for left lane
# xs_left = []
# ys_left = []

# for i in range(40):
#     idx = np.argmax(acc)
#     r_idx = idx // acc.shape[1]
#     t_idx = idx % acc.shape[1]
#     acc[r_idx, t_idx] = 0  # Zero out the max value in accumulator

#     rho = rhos[r_idx]
#     theta = thetas[t_idx]
#     # Transform a point in Hough space to a line in xy-space.
#     a = - (np.cos(theta) / np.sin(theta))  # slope of the line
#     b = (rho / np.sin(theta))  # y-intersect of the line

#     # Break if both right and left lanes are detected
#     if xs_right and xs_left:
#         break

#     if a < 0:  # Left lane
#         if xs_left:
#             continue
#         xs = xs_left
#         ys = ys_left
#     else:  # Right Lane
#         if xs_right:
#             continue
#         xs = xs_right
#         ys = ys_right

#     for x in range(img.shape[1]):
#         y = a * x + b
#         if y > img.shape[0] * 0.6 and y < img.shape[0]:
#             xs.append(x)
#             ys.append(int(round(y)))


# plt.imshow(img)
# plt.plot(xs_left, ys_left, linewidth=5.0)
# plt.plot(xs_right, ys_right, linewidth=5.0)
# plt.axis('off')
# plt.show()
