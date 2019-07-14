from __future__ import division
import numpy as np
# Setup
from time import time
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from skimage import io

plt.rcParams['figure.figsize'] = (15.0, 12.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def compress_image(image, num_values):
    """Compress an image using SVD and keeping the top `num_values` singular values.

    Args:
        image: numpy array of shape (H, W)
        num_values: number of singular values to keep

    Returns:
        compressed_image: numpy array of shape (H, W) containing the compressed image
        compressed_size: size of the compressed image
    """
    compressed_image = None
    compressed_size = 0

    # YOUR CODE HERE
    # Steps:
    #     1. Get SVD of the image
    #     2. Only keep the top `num_values` singular values, and compute `compressed_image`
    #     3. Compute the compressed size
    H, W = image.shape
    u, s, v = np.linalg.svd(image)
    s1 = np.zeros((H, W))
    for i in range(num_values):
        s1[i][i] = s[i]
    compressed_image = np.dot(u, s1)
    compressed_image = np.dot(compressed_image, v)
    # compressed_image = np.dot(compress_image, v)
    compressed_size = (H + W + 1) * num_values
    pass
    # END YOUR CODE

    assert compressed_image.shape == image.shape, \
        "Compressed image and original image don't have the same shape"

    assert compressed_size > 0, "Don't forget to compute compressed_size"

    return compressed_image, compressed_size


image = io.imread('pitbull.jpg', as_grey=True)
n_values = [10, 50, 100]

for n in n_values:
    # Compress the image using `n` singular values
    compressed_image, compressed_size = compress_image(image, n)
    print(n)

    compression_ratio = compressed_size / image.size

    print("Data size (original): %d" % (image.size))
    print("Data size (compressed): %d" % compressed_size)
    print("Compression ratio: %.3f" % (compression_ratio))

    plt.imshow(compressed_image, cmap='gray')
    title = "n = %s" % n
    plt.title(title)
    plt.axis('off')
    plt.show()
