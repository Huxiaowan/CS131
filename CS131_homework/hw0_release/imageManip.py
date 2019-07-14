import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
from skimage import color
from skimage import io


def load(image_path):
    """ Loads an image from a file path

    Args:
        image_path: file path to the image

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """
    out = None

    # YOUR CODE HERE
    # Use skimage io.imread
    out = io.imread(image_path)
    pass
    # END YOUR CODE

    return out


def display(img):
    # Show image
    plt.imshow(img)
    #plt.imshow(img, cmap=plt.get_cmap('img'))
    #plt.imshow(img, cmap='Greys_r')
    #plt.imshow(img, plt.cm.gray)
    plt.axis('off')
    plt.show()


image1_path = './image1.jpg'
image2_path = './image2.jpg'
image1 = load(image1_path)
image2 = load(image2_path)
# display(image1)
# display(image2)
# print(image1.shape)


def change_value(image):
    """ Change the value of every pixel by following x_n = 0.5*x_p^2 
        where x_n is the new value and x_p is the original value

    Args:
        image: numpy array of shape(image_height, image_width, 3)

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    out = None

    # YOUR CODE HERE
    image = np.array(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i, j] = 0.5 * pow(image[i, j], 2)
    out = image
    pass
    # END YOUR CODE

    return out


new_image = change_value(image1)
# display(new_image)


def convert_to_grey_scale(image):
    """ Change image to gray scale

    Args:
        image: numpy array of shape(image_height, image_width, 3)

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """
    out = None

    # YOUR CODE HERE
    out = np.dot(image[..., :3], [0.299, 0.587, 0.114])
    pass
    # END YOUR CODE

    return out


grey_image = convert_to_grey_scale(image1)
# display(grey_image)


def rgb_decomposition(image, channel):
    """ Return image **excluding** the rgb channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: str specifying the channel

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    out = None

    # YOUR CODE HERE
    image = np.array(image)
    if channel == 'R':
        image[:, :, 0] = 0
    elif channel == 'G':
        image[:, :, 1] = 0
    else:
        image[:, :, 2] = 0

    out = image
    # END YOUR CODE

    return out


without_red = rgb_decomposition(image1, 'R')
without_blue = rgb_decomposition(image1, 'B')
without_green = rgb_decomposition(image1, 'G')

# display(without_red)
# display(without_blue)
# display(without_green)


def lab_decomposition(image, channel):
    """ Return image decomposed to just the lab channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: str specifying the channel

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    lab = color.rgb2lab(image)
    out = None

    # YOUR CODE HERE
    lab = np.array(lab)
    if channel == 'L':
        lab[:, :, 1] = 0
        lab[:, :, 2] = 0
    elif channel == 'A':
        lab[:, :, 0] = 0
        lab[:, :, 2] = 0
    else:
        lab[:, :, 0] = 0
        lab[:, :, 1] = 0

    out = lab
    # END YOUR CODE

    return out


image_l = lab_decomposition(image1, 'L')
image_a = lab_decomposition(image1, 'A')
image_b = lab_decomposition(image1, 'B')

# display(image_l)
# display(image_a)
# display(image_b)


def hsv_decomposition(image, channel='H'):
    """ Return image decomposed to just the hsv channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: str specifying the channel

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    hsv = color.rgb2hsv(image)
    out = None

    # YOUR CODE HERE
    hsv = np.array(hsv)
    if channel == 'H':
        hsv[:, :, 1] = 0
        hsv[:, :, 2] = 0
    elif channel == 'S':
        hsv[:, :, 0] = 0
        hsv[:, :, 2] = 0
    else:
        hsv[:, :, 0] = 0
        hsv[:, :, 1] = 0

    out = hsv
    # END YOUR CODE

    return out


image_h = hsv_decomposition(image1, 'H')
image_s = hsv_decomposition(image1, 'S')
image_v = hsv_decomposition(image1, 'V')

display(image_h)
display(image_s)
display(image_v)


def mix_images(image1, image2, channel1, channel2):
    """ Return image which is the left of image1 and right of image 2 excluding
    the specified channels for each image

    Args:
        image1: numpy array of shape(image_height, image_width, 3)
        image2: numpy array of shape(image_height, image_width, 3)
        channel1: str specifying channel used for image1
        channel2: str specifying channel used for image2

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    out = np.array(image1)
    # YOUR CODE HERE

    image1 = np.array(image1)
    image2 = np.array(image2)

    size1 = image1.shape
    size2 = image2.shape

    height1 = size1[0]
    width1 = size1[1]

    height2 = size2[0]
    width2 = size2[1]

    #out = Image.new("RGB", (height1, width1))

    #out = np.empty_like(image1)

    if channel1 == 'R':
        image1[:, :, 0] = 0
    elif channel1 == 'G':
        image1[:, :, 1] = 0
    else:
        image1[:, :, 2] = 0

    if channel2 == 'R':
        image2[:, :, 0] = 0
    elif channel2 == 'G':
        image2[:, :, 1] = 0
    else:
        image2[:, :, 2] = 0

    out[:, 0:width1 / 2, :] = image1[:, 0:width1 / 2, :]
    out[:height2, width2 / 2:width2] = image2[:height2, width2 / 2:width2]

    pass
    # END YOUR CODE

    return out


# image_mixed = mix_images(image1, image2, channel1='R', channel2='G')

# display(image_mixed)
