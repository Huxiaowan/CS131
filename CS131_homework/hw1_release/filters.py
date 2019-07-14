import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from time import time


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # YOUR CODE HERE
    out1 = np.zeros((Hi + Hk - 1, Wi + Wk - 1))

    for i in range(0, Hi + Hk - 1):
        for j in range(0, Wi + Wk - 1):
            temp = 0.0
            for m in range(0, Hi):
                for n in range(0, Wi):
                    if (((i - m) >= 0)and ((i - m) < Hk) and ((j - n) >= 0)and((j - n) < Wk)):
                        temp += image[m][n] * kernel[i - m][j - n]
            out1[i][j] = temp

    for i in range(0, Hi):
        for j in range(0, Wi):
            out[i][j] = out1[i + (Hk - 1) / 2][j + (Wk - 1) / 2]
            pass
        pass
    # END YOUR CODE

    return out


kernel = np.array(
    [
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ])
img = io.imread('dog.jpg', as_grey=True)
out = conv_nested(img, kernel)

# Plot original image
plt.subplot(2, 2, 1)
plt.imshow(img)
plt.title('Original')
plt.axis('off')

# Plot your convolved image
plt.subplot(2, 2, 3)
plt.imshow(out)
plt.title('Convolution')
plt.axis('off')

# Plot what you should get
solution_img = io.imread('convoluted_dog.jpg', as_grey=True)
plt.subplot(2, 2, 4)
plt.imshow(solution_img)
plt.title('What you should get')
plt.axis('off')


plt.show()


def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W)
        pad_width: width of the zero padding (left and right padding)
        pad_height: height of the zero padding (bottom and top padding)

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width)
    """

    H, W = image.shape
    out = None

    # YOUR CODE HERE
    out = np.pad(image, ((pad_height, pad_height), (pad_width,
                                                    pad_width)), 'constant', constant_values=(0, 0))
    pass
    # END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # YOUR CODE HERE
    image = zero_pad(image, Hk - 1, Wk - 1)
    kernel = np.fliplr(kernel)
    kernel = np.flipud(kernel)
    out1 = np.zeros((Hi + Hk - 1, Wi + Wk - 1))
    for i in range(0, Hi + Hk - 1):
        for j in range(0, Wi + Wk - 1):
            temp = 0.0
            for m in range(0, Hk):
                for n in range(0, Wk):
                    temp += kernel[m][n] * image[m + i][n + j]
            out1[i][j] = temp

    for i in range(0, Hi):
        for j in range(0, Wi):
            out[i][j] = out1[i + (Hk - 1) / 2][j + (Wk - 1) / 2]
            pass
        pass
    pass
    # END YOUR CODE

    return out


# kernel = np.array(
#     [
#         [1, 0, -1],
#         [2, 0, -2],
#         [1, 0, -1]
#     ])
# test_img = np.zeros((3, 3))
# test_img[0:3, 0:3] = 1
# # print(conv_nested(test_img, kernel))
# # print(1)
# # print(conv_fast(test_img, kernel))

# img = io.imread('./dog.jpg', as_grey=True)
# print(img.shape)
# t0 = time()
# out_fast = conv_fast(test_img, kernel)
# t1 = time()
# out_nested = conv_nested(test_img, kernel)
# t2 = time()

# # Compare the running time of the two implementations
# print("conv_nested: took %f seconds." % (t2 - t1))
# print("conv_fast: took %f seconds." % (t1 - t0))

# # Plot conv_nested output
# plt.subplot(1, 2, 1)
# plt.imshow(out_nested)
# plt.title('conv_nested')
# plt.axis('off')
# print("OK")
# # Plot conv_fast output
# plt.subplot(1, 2, 2)
# plt.imshow(out_fast)
# plt.title('conv_fast')
# plt.axis('off')
# plt.show()
# print("hu")
# # Make sure that the two outputs are the same
# if not (np.max(out_fast - out_nested) < 1e-10):
#     print("Different outputs! Check your implementation.")


def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # YOUR CODE HERE

    pass
    # END YOUR CODE

    return out


def cross_correlation(f, g):
    """ Cross-correlation of f and g

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    # YOUR CODE HERE
    out = conv_fast(f, g)
    pass
    # END YOUR CODE

    return out


# img = io.imread('shelf.jpg')
# img_grey = io.imread('shelf.jpg', as_grey=True)
# temp = io.imread('template.jpg')
# temp_grey = io.imread('template.jpg', as_grey=True)

# Perform cross-correlation between the image and the template
# out = cross_correlation(img_grey, temp_grey)

# # Find the location with maximum similarity
# y, x = (np.unravel_index(out.argmax(), out.shape))
# Display product template
# plt.figure(figsize=(25, 20))
# plt.subplot(3, 1, 1)
# plt.imshow(img_grey)
# plt.title('Template')
# plt.axis('off')

# # Display cross-correlation output
# plt.subplot(3, 1, 2)
# plt.imshow(out)
# plt.title('Cross-correlation (white means more correlated)')
# plt.axis('off')

# # Display image
# plt.subplot(3, 1, 3)
# plt.imshow(img)
# plt.title('Result (blue marker on the detected location)')
# plt.axis('off')

# # Draw marker at detected location
# plt.plot(x, y, 'bx', ms=40, mew=10)
# plt.show()


def zero_mean_cross_correlation(f, g):

    out = None
    # YOUR CODE HERE
    Hi, Wi = f.shape
    Hk, Wk = g.shape
    out = np.zeros((Hi, Wi))
    f = zero_pad(f, Hk - 1, Wk - 1)
    f_sum = 0.0
    g_sum = 0.0
    out1 = np.zeros((Hi + Hk - 1, Wi + Wk - 1))

    for m in range(0, Hi):
        for n in range(0, Wi):
            f_sum += f[m][n]

    f_av = f_sum / (Hi * Wi)

    for m in range(0, Hk):
        for n in range(0, Wk):
            g_sum += g[m][n]

    g_av = g_sum / (Hk * Wk)

    for i in range(0, Hi + Hk - 1):
        for j in range(0, Wi + Wk - 1):
            temp = 0.0
            for m in range(0, Hk):
                for n in range(0, Wk):
                    temp += (g[m][n] - g_av) * (f[m + i][n + j] - f_av)
            out1[i][j] = temp

    for i in range(0, Hi):
        for j in range(0, Wi):
            out[i][j] = out1[i + (Hk - 1) / 2][j + (Wk - 1) / 2]
            pass
        pass
    # END YOUR CODE

    return out


# out = zero_mean_cross_correlation(img_grey, temp_grey)

# # Find the location with maximum similarity
# y, x = (np.unravel_index(out.argmax(), out.shape))

# # Display product template
# plt.figure(figsize=(30, 20))
# plt.subplot(3, 1, 1)
# plt.imshow(temp)
# plt.title('Template')
# plt.axis('off')

# # Display cross-correlation output
# plt.subplot(3, 1, 2)
# plt.imshow(out)
# plt.title('Cross-correlation (white means more correlated)')
# plt.axis('off')

# # Display image
# plt.subplot(3, 1, 3)
# plt.imshow(img)
# plt.title('Result (blue marker on the detected location)')
# plt.axis('off')

# # Draw marker at detcted location
# plt.plot(x, y, 'bx', ms=40, mew=10)
# plt.show()


def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    # YOUR CODE HERE
    Hi, Wi = f.shape
    Hk, Wk = g.shape
    out = np.zeros((Hi, Wi))
    f = zero_pad(f, Hk - 1, Wk - 1)
    f_sum = 0.0
    g_sum = 0.0
    out1 = np.zeros((Hi + Hk - 1, Wi + Wk - 1))

    for m in range(0, Hi):
        for n in range(0, Wi):
            f_sum += f[m][n]

    f_av = f_sum / (Hi * Wi)

    for m in range(0, Hk):
        for n in range(0, Wk):
            g_sum += g[m][n]

    g_av = g_sum / (Hk * Wk)

    for i in range(0, Hi + Hk - 1):
        for j in range(0, Wi + Wk - 1):
            temp = 0.0
            f_stan = 0.0
            g_stan = 0.0
            for m in range(0, Hk):
                for n in range(0, Wk):
                    temp += (g[m][n] - g_av) * (f[m + i][n + j] - f_av)
                    f_stan += pow(f[m + i][n + j], 2)
                    g_stan += pow(g[m][n], 2)
            out1[i][j] = temp / pow((f_stan * g_stan), 0.5)

    for i in range(0, Hi):
        for j in range(0, Wi):
            out[i][j] = out1[i + (Hk - 1) / 2][j + (Wk - 1) / 2]
            pass
        pass
    pass
    # END YOUR CODE

    return out


# img = io.imread('shelf_dark.jpg')
# img_grey = io.imread('shelf_dark.jpg', as_grey=True)

# out = normalized_cross_correlation(img_grey, temp_grey)

# # Find the location with maximum similarity
# y, x = (np.unravel_index(out.argmax(), out.shape))

# # Display product template
# plt.figure(figsize=(30, 20))
# plt.subplot(3, 1, 1)
# plt.imshow(temp)
# plt.title('Template')
# plt.axis('off')

# # Display cross-correlation output
# plt.subplot(3, 1, 2)
# plt.imshow(out)
# plt.title('Cross-correlation (white means more correlated)')
# plt.axis('off')

# # Display image
# plt.subplot(3, 1, 3)
# plt.imshow(img)
# plt.title('Result (blue marker on the detected location)')
# plt.axis('off')

# # Draw marker at detcted location
# plt.plot(x, y, 'rx', ms=25, mew=5)
# plt.show()


def check_product_on_shelf(shelf, product):
    out = zero_mean_cross_correlation(shelf, product)

    # Scale output by the size of the template
    out = out / float(product.shape[0] * product.shape[1])

    # Threshold output (this is arbitrary, you would need to tune the threshold for a real application)
    out = out > 0.025

    if np.sum(out) > 0:
        print('The product is on the shelf')
    else:
        print('The product is not on the shelf')


# Load image of the shelf without the product
# img2 = io.imread('shelf_soldout.jpg')
# img2_grey = io.imread('shelf_soldout.jpg', as_grey=True)

# plt.imshow(img)
# plt.axis('off')
# plt.show()
# check_product_on_shelf(img_grey, temp_grey)

# plt.imshow(img2)
# plt.axis('off')
# plt.show()

# check_product_on_shelf(img2_grey, temp_grey)
