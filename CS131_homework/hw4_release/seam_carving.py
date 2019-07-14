from __future__ import print_function

import numpy as np
from skimage import color
from skimage import io, util
from skimage import filters
# Setup
import matplotlib.pyplot as plt
from matplotlib import rc

from time import time
from IPython.display import HTML


# %matplotlib inline
plt.rcParams['figure.figsize'] = (15.0, 12.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# # for auto-reloading extenrnal modules
# %load_ext autoreload
# %autoreload 2

# Load image
img = io.imread('imgs/broadway_tower.jpg')
img = util.img_as_float(img)

# plt.title('Original Image')
# plt.imshow(img)
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


def energy_function(image):
    """Computes energy of the input image.

    For each pixel, we will sum the absolute value of the gradient in each direction.
    Don't forget to convert to grayscale first.

    Hint: you can use np.gradient here

    Args:
        image: numpy array of shape (H, W, 3)

    Returns:
        out: numpy array of shape (H, W)
    """
    H, W = image.shape
    out = np.zeros((H, W))

    # YOUR CODE HERE
    # image = color.rgb2gray(image)
    dx = abs(partial_x(image))

    dy = abs(partial_y(image))

    out = dx + dy
    pass
    # END YOUR CODE

    return out


def energy_function_2(image):
    """Computes energy of the input image.

    For each pixel, we will sum the absolute value of the gradient in each direction.
    Don't forget to convert to grayscale first.

    Hint: you can use np.gradient here

    Args:
        image: numpy array of shape (H, W, 3)

    Returns:
        out: numpy array of shape (H, W)
    """
    H, W = image.shape
    out = np.zeros((H, W))

    # YOUR CODE HERE
    # image = color.rgb2gray(image)
    dx = abs(partial_x(image))

    dy = abs(partial_y(image))

    out = dx + dy
    out = image
    pass
    # END YOUR CODE

    return out


def compute_cost(image, energy, axis=1):
    """Computes optimal cost map (vertical) and paths of the seams.

    Starting from the first row, compute the cost of each pixel as the sum of energy along the
    lowest energy path from the top.
    We also return the paths, which will contain at each pixel either -1, 0 or 1 depending on
    where to go up if we follow a seam at this pixel.

    Make sure your code is vectorized because this function will be called a lot.
    You should only have one loop iterating through the rows.

    Args:
        image: not used for this function
               (this is to have a common interface with compute_forward_cost)
        energy: numpy array of shape (H, W)
        axis: compute cost in width (axis=1) or height (axis=0)

    Returns:
        cost: numpy array of shape (H, W)
        paths: numpy array of shape (H, W) containing values -1, 0 or 1
    """
    energy = energy.copy()

    if axis == 0:
        energy = np.transpose(energy, (1, 0))

    H, W = energy.shape

    cost = np.zeros((H, W))
    paths = np.zeros((H, W), dtype=np.int)

    # Initialization
    cost[0] = energy[0]
    paths[0] = 0  # we don't care about the first row of paths

    # YOUR CODE HERE
    cost = energy
    for i in range(1, H):
        for j in range(0, W):
            if(j == 0):
                cost[i][j] = cost[i][j] + \
                    np.min([cost[i - 1][j], cost[i - 1][j + 1]])
                paths[i][j] = np.argmin(
                    [cost[i - 1][j], cost[i - 1][j + 1]])
            elif(j == W - 1):
                cost[i][j] = cost[i][j] + \
                    np.min([cost[i - 1][j], cost[i - 1][j - 1]])
                paths[i][j] = np.argmin(
                    [cost[i - 1][j - 1], cost[i - 1][j]]) - 1
            else:
                cost[i][j] = cost[i][j] + \
                    np.min([cost[i - 1][j - 1], cost[i - 1]
                            [j], cost[i - 1][j + 1]])
                paths[i][j] = np.argmin([cost[i - 1][j - 1], cost[i - 1]
                                         [j], cost[i - 1][j + 1]]) - 1
            pass
        pass
    pass
    # END YOUR CODE

    if axis == 0:
        cost = np.transpose(cost, (1, 0))
        paths = np.transpose(paths, (1, 0))

    # Check that paths only contains -1, 0 or 1
    assert np.all(np.any([paths == 1, paths == 0, paths == -1], axis=0)), \
        "paths contains other values than -1, 0 or 1"

    return cost, paths


# # Vertical Cost Map
# start = time()
# # don't need the first argument for compute_cost
# vcost, vpaths = compute_cost(img, energy, axis=1)
# end = time()

# print("Computing vertical cost map: %f seconds." % (end - start))


# plt.title('Vertical Cost Map')
# plt.axis('off')
# plt.imshow(vcost, cmap='inferno')
# plt.show()


def backtrack_seam(paths, end):
    """Backtracks the paths map to find the seam ending at (H-1, end)

    To do that, we start at the bottom of the image on position (H-1, end), and we
    go up row by row by following the direction indicated by paths:
        - left (value -1)
        - middle (value 0)
        - right (value 1)

    Args:
        paths: numpy array of shape (H, W) containing values -1, 0 or 1
        end: the seam ends at pixel (H, end)

    Returns:
        seam: np.array of indices of shape (H,). The path pixels are the (i, seam[i])
    """
    H, W = paths.shape
    seam = np.zeros(H, dtype=np.int)

    # Initialization
    seam[H - 1] = end

    # YOUR CODE HERE
    for i in range(H - 2, -1, -1):
        if (paths[i + 1][seam[i + 1]] == 0):
            seam[i] = seam[i + 1]
        elif(paths[i + 1][seam[i + 1]] == -1):
            seam[i] = seam[i + 1] - 1
        else:
            seam[i] = seam[i + 1] + 1
            pass
        pass
    pass
    # END YOUR CODE

    # Check that seam only contains values in [0, W-1]
    assert np.all(np.all([seam >= 0, seam < W], axis=0)
                  ), "seam contains values out of bounds"

    return seam


def remove_seam(image, seam):
    """Remove a seam from the image.

    This function will be helpful for functions reduce and reduce_forward.

    Args:
        image: numpy array of shape (H, W, C) or shape (H, W)
        seam: numpy array of shape (H,) containing indices of the seam to remove

    Returns:
        out: numpy array of shape (H, W-1, C) or shape (H, W-1)
    """

    # Add extra dimension if 2D input
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)

    H, W, C = image.shape
    out = np.zeros((H, W - 1, C))
    for i in range(0, H):
        out[i] = np.delete(image[i], seam[i], axis=0)

    # END YOUR CODE
    out = np.squeeze(out)  # remove last dimension if C == 1

    return out


def reduce(image, size, axis=1, efunc=energy_function, cfunc=compute_cost):
    """Reduces the size of the image using the seam carving process.

    At each step, we remove the lowest energy seam from the image. We repeat the process
    until we obtain an output of desired size.
    Use functions:
        - efunc
        - cfunc
        - backtrack_seam
        - remove_seam

    Args:
        image: numpy array of shape (H, W, 3)
        size: size to reduce height or width to (depending on axis)
        axis: reduce in width (axis=1) or height (axis=0)
        efunc: energy function to use
        cfunc: cost function to use

    Returns:
        out: numpy array of shape (size, W, 3) if axis=0, or (H, size, 3) if axis=1
    """

    out = np.copy(image)
    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    H = out.shape[0]
    W = out.shape[1]

    assert W > size, "Size must be smaller than %d" % W

    assert size > 0, "Size must be greater than zero"

    # YOUR CODE HERE
    if (axis == 1):
        size_reduce = W - size
    elif(axis == 0):
        size_reduce = H - size
        pass
    for i in range(0, size_reduce):
        energy = efunc(image)
        vcost, vpaths = cfunc(image, energy)
        end = np.argmin(vcost[-1])
        seam = backtrack_seam(vpaths, end)
        out = remove_seam(image, seam)
        image = out
    # END YOUR CODE

    assert out.shape[1] == size, "Output doesn't have the right shape"

    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    return out


# Reduce image height
# H, W, _ = img.shape
# H_new = 300

# start = time()
# out = reduce(img, H_new, axis=0)
# end = time()

# print("Reducing height from %d to %d: %f seconds." % (H, H_new, end - start))

# plt.subplot(1, 2, 1)
# plt.title('Original')
# plt.imshow(img)

# plt.subplot(1, 2, 2)
# plt.title('Resized')
# plt.imshow(out)

# plt.show()


def duplicate_seam(image, seam):
    """Duplicates pixels of the seam, making the pixels on the seam path "twice larger".

    This function will be helpful in functions enlarge_naive and enlarge.

    Args:
        image: numpy array of shape (H, W, C)
        seam: numpy array of shape (H,) of indices

    Returns:
        out: numpy array of shape (H, W+1, C)
    """

    H, W = image.shape
    out = np.zeros((H, W + 1))
    # YOUR CODE HERE
    for i in range(0, H):
        # out[i] = image[i].insert(seam[i], image[i][seam[i]])
        out[i] = np.insert(image[i], seam[i], image[i][seam[i]], axis=0)
    pass
    # END YOUR CODE

    return out


# test_img = np.arange(9, dtype=np.float64).reshape((3, 3))
# test_img = np.stack([test_img, test_img, test_img], axis=2)
# assert test_img.shape == (3, 3, 3)

# cost = np.array([[1.0, 2.0, 1.5],
#                  [4.0, 2.0, 3.5],
#                  [6.0, 2.5, 5.0]])

# paths = np.array([[0, 0, 0],
#                   [0, -1, 0],
#                   [1, 0, -1]])

# # Increase image width
# W_new = 4

# # We force the cost and paths to our values
# end = np.argmin(cost[-1])
# seam = backtrack_seam(paths, end)
# out = duplicate_seam(test_img, seam)
# print(out[:,:,0])

def enlarge_naive(image, size, axis=1, efunc=energy_function, cfunc=compute_cost):
    """Increases the size of the image using the seam duplication process.

    At each step, we duplicate the lowest energy seam from the image. We repeat the process
    until we obtain an output of desired size.
    Use functions:
        - efunc
        - cfunc
        - backtrack_seam
        - duplicate_seam

    Args:
        image: numpy array of shape (H, W, C)
        size: size to increase height or width to (depending on axis)
        axis: increase in width (axis=1) or height (axis=0)
        efunc: energy function to use
        cfunc: cost function to use

    Returns:
        out: numpy array of shape (size, W, C) if axis=0, or (H, size, C) if axis=1
    """

    out = np.copy(image)
    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    H = out.shape[0]
    W = out.shape[1]

    assert size > W, "size must be greather than %d" % W

    # YOUR CODE HERE
    if (axis == 1):
        size_reduce = size - W
    elif(axis == 0):
        size_reduce = size - H
        pass
    for i in range(0, size_reduce):
        energy = efunc(image)
        vcost, vpaths = cfunc(image, energy)
        end = np.argmin(vcost[-1])
        seam = backtrack_seam(vpaths, end)
        out = duplicate_seam(image, seam)
        image = out
    pass
    # END YOUR CODE

    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    return out


def find_seams(image, k, axis=1, efunc=energy_function, cfunc=compute_cost):
    """Find the top k seams (with lowest energy) in the image.

    We act like if we remove k seams from the image iteratively, but we need to store their
    position to be able to duplicate them in function enlarge.

    We keep track of where the seams are in the original image with the array seams, which
    is the output of find_seams.
    We also keep an indices array to map current pixels to their original position in the image.

    Use functions:
        - efunc
        - cfunc
        - backtrack_seam
        - remove_seam

    Args:
        image: numpy array of shape (H, W, C)
        k: number of seams to find
        axis: find seams in width (axis=1) or height (axis=0)
        efunc: energy function to use
        cfunc: cost function to use

    Returns:
        seams: numpy array of shape (H, W)
    """

    image = np.copy(image)
    if axis == 0:
        image = np.transpose(image, (1, 0, 2))

    H, W = image.shape
    assert W > k, "k must be smaller than %d" % W

    # Create a map to remember original pixel indices
    # At each step, indices[row, col] will be the original column of current pixel
    # The position in the original image of this pixel is: (row, indices[row, col])
    # We initialize `indices` with an array like (for shape (2, 4)):
    #     [[1, 2, 3, 4],
    #      [1, 2, 3, 4]]
    indices = np.tile(range(W), (H, 1))  # shape (H, W)

    # We keep track here of the seams removed in our process
    # At the end of the process, seam number i will be stored as the path of value i+1 in `seams`
    # An example output for `seams` for two seams in a (3, 4) image can be:
    #    [[0, 1, 0, 2],
    #     [1, 0, 2, 0],
    #     [1, 0, 0, 2]]
    seams = np.zeros((H, W), dtype=np.int)

    # Iteratively find k seams for removal
    for i in range(k):
        # Get the current optimal seam
        energy = efunc(image)
        cost, paths = cfunc(image, energy)
        end = np.argmin(cost[H - 1])
        seam = backtrack_seam(paths, end)

        # Remove that seam from the image
        image = remove_seam(image, seam)

        # Store the new seam with value i+1 in the image
        # We can assert here that we are only writing on zeros (not overwriting existing seams)
        assert np.all(seams[np.arange(H), indices[np.arange(H), seam].astype(int)] == 0), \
            "we are overwriting seams"
        seams[np.arange(H), indices[np.arange(H), seam].astype(int)] = i + 1

        # We remove the indices used by the seam, so that `indices` keep the same shape as `image`
        indices = remove_seam(indices, seam)

    if axis == 0:
        seams = np.transpose(seams, (1, 0))

    return seams


def enlarge(image, size, axis=1, efunc=energy_function, cfunc=compute_cost):
    """Enlarges the size of the image by duplicating the low energy seams.

    We start by getting the k seams to duplicate through function find_seams.
    We iterate through these seams and duplicate each one iteratively.

    Use functions:
        - find_seams
        - duplicate_seam

    Args:
        image: numpy array of shape (H, W, C)
        size: size to reduce height or width to (depending on axis)
        axis: enlarge in width (axis=1) or height (axis=0)
        efunc: energy function to use
        cfunc: cost function to use

    Returns:
        out: numpy array of shape (size, W, C) if axis=0, or (H, size, C) if axis=1
    """

    out = np.copy(image)
    # Transpose for height resizing
    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    H, W = out.shape

    assert size > W, "size must be greather than %d" % W

    assert size <= 2 * W, "size must be smaller than %d" % (2 * W)

    # YOUR CODE HERE
    k = size - W
    seams = find_seams(image, k)
    indices = np.tile(range(W), (H, 1))
    for i in range(0, k):
        a = np.argwhere(seams == i + 1)
        seam = a[np.arange(H), 1]
        out = duplicate_seam(out, seam)
    pass
    # END YOUR CODE

    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    return out


def compute_forward_cost(image, energy):
    """Computes forward cost map (vertical) and paths of the seams.

    Starting from the first row, compute the cost of each pixel as the sum of energy along the
    lowest energy path from the top.
    Make sure to add the forward cost introduced when we remove the pixel of the seam.

    We also return the paths, which will contain at each pixel either -1, 0 or 1 depending on
    where to go up if we follow a seam at this pixel.

    Args:
        image: numpy array of shape (H, W, 3) or (H, W)
        energy: numpy array of shape (H, W)

    Returns:
        cost: numpy array of shape (H, W)
        paths: numpy array of shape (H, W) containing values -1, 0 or 1
    """

    image = color.rgb2gray(image)
    H, W = image.shape

    cost = np.zeros((H, W))
    paths = np.zeros((H, W), dtype=np.int)

    # Initialization
    cost[0] = energy[0]
    for j in range(W):
        if j > 0 and j < W - 1:
            cost[0, j] += np.abs(image[0, j + 1] - image[0, j - 1])
            print(cost)
    paths[0] = 0  # we don't care about the first row of paths

    # YOUR CODE HERE
    for i in range(1, H):
        for j in range(0, W):
            if(j == 0):
                cost[i][j] = cost[i][j] + \
                    np.min([cost[i - 1][j], cost[i - 1][j + 1]])
                paths[i][j] = np.argmin(
                    [cost[i - 1][j], cost[i - 1][j + 1]])
            elif(j == W - 1):
                cost[i][j] = cost[i][j] + \
                    np.min([cost[i - 1][j], cost[i - 1][j - 1]])
                paths[i][j] = np.argmin(
                    [cost[i - 1][j - 1], cost[i - 1][j]]) - 1
            else:
                cost[i][j] = cost[i][j] + np.min(cost[i - 1][j - 1] + abs(image[i][j + 1] - image[i][j - 1]) + abs(image[i - 1][j] - image[i][j - 1]), cost[i - 1][j] + abs(
                    image[i][j + 1] - image[i][j - 1]), cost[i - 1][j + 1] + abs(image[i][j + 1] - image[i][j - 1]) + abs(image[i - 1][j] - image[i][j + 1]))
                paths[i][j] = np.argmin([cost[i - 1][j - 1], cost[i - 1]
                                         [j], cost[i - 1][j + 1]]) - 1
            pass
        pass
    pass
    # END YOUR CODE

    # Check that paths only contains -1, 0 or 1
    assert np.all(np.any([paths == 1, paths == 0, paths == -1], axis=0)), \
        "paths contains other values than -1, 0 or 1"

    return cost, paths


def reduce_fast(image, size, axis=1, efunc=energy_function, cfunc=compute_cost):
    """Reduces the size of the image using the seam carving process. Faster than `reduce`.

    Use your own implementation (you can use auxiliary functions if it helps like `energy_fast`)
    to implement a faster version of `reduce`.

    Hint: do we really need to compute the whole cost map again at each iteration?

    Args:
        image: numpy array of shape (H, W, C)
        size: size to reduce height or width to (depending on axis)
        axis: reduce in width (axis=1) or height (axis=0)
        efunc: energy function to use
        cfunc: cost function to use

    Returns:
        out: numpy array of shape (size, W, C) if axis=0, or (H, size, C) if axis=1
    """

    out = np.copy(image)
    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    H = out.shape[0]
    W = out.shape[1]

    assert W > size, "Size must be smaller than %d" % W

    assert size > 0, "Size must be greater than zero"

    # YOUR CODE HERE
    if (axis == 1):
        size_reduce = W - size
    elif(axis == 0):
        size_reduce = H - size
        pass
    for i in range(0, size_reduce):
        energy = efunc(image)
        vcost, vpaths = cfunc(image, energy)
        end = np.argmin(vcost[-1])
        seam = backtrack_seam(vpaths, end)
        out = remove_seam(image, seam)
        image = out
    # END YOUR CODE

    assert out.shape[1] == size, "Output doesn't have the right shape"

    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    return out


def remove_object(image, mask):
    """Remove the object present in the mask.

    Returns an output image with same shape as the input image, but without the object in the mask.

    Args:
        image: numpy array of shape (H, W, 3)
        mask: numpy boolean array of shape (H, W)

    Returns:
        out: numpy array of shape (H, W, 3)
    """
    out = np.copy(image)

    # YOUR CODE HERE
    image = color.rgb2gray(image)
    H, W = image.shape
    image = (image) * (1 - mask)
    out = reduce(image, 300)
    out = enlarge(image, W)
    out = color.gray2rgb(out)
    pass
    # END YOUR CODE

    return out


image = io.imread('imgs/h.jpg')
img = io.imread('imgs/wyeth.jpg')
img = util.img_as_float(img)
print(img.shape)

mask = io.imread('imgs/wyeth_mask.jpg', as_grey=True)
# mask = util.img_as_bool(mask)


out = remove_object(image, mask)

plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(img)

plt.subplot(2, 2, 2)
plt.title('Mask of the object to remove')
plt.imshow(mask)

plt.subplot(2, 2, 3)
plt.title('Image with object removed')
plt.imshow(out)

plt.show()
