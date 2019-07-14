from __future__ import print_function, division
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
from skimage.feature import hog
from skimage import data, color, exposure
from skimage.transform import rescale, resize, downscale_local_mean
import glob
import os
import fnmatch
import time
import math


import numpy as np
from skimage import feature, data, color, exposure, io
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.filters import gaussian
from scipy import signal
from scipy.ndimage import interpolation
import math
# from util import *

import warnings
warnings.filterwarnings('ignore')

# from detection import *
# from util import *

plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def hog_feature(image, pixel_per_cell=8):
    '''
    Compute hog feature for a given image.

    Hint: use the hog function provided by skimage

    Args:
        image: an image with object that we want to detect
        pixel_per_cell: number of pixels in each cell, an argument for hog descriptor

    Returns:
        score: a vector of hog representation
        hogImage: an image representation of hog provided by skimage
    '''
    # YOUR CODE HERE
    (hogFeature, hogImage) = hog(image, visualise=True)
    # END YOUR CODE
    return (hogFeature, hogImage)


image_paths = fnmatch.filter(os.listdir('./face'), '*.jpg')
list.sort(image_paths)
n = len(image_paths)
face_shape = io.imread('./face/' + image_paths[0], as_grey=True).shape
avg_face = np.zeros((face_shape))
for i, image_path in enumerate(image_paths):
    image = io.imread('./face/' + image_path, as_grey=True)
    avg_face = np.asarray(image) + np.asarray(avg_face)
avg_face = avg_face / n

(face_feature, face_hog) = hog_feature(avg_face)


def sliding_window(image, base_score, stepSize, windowSize, pixel_per_cell=8):
    ''' A sliding window that checks each different location in the image,
        and finds which location has the highest hog score. The hog score is computed
        as the dot product between hog feature of the sliding window and the hog feature
        of the template. It generates a response map where each location of the
        response map is a corresponding score. And you will need to resize the response map
        so that it has the same shape as the image.

    Args:
        image: an np array of size (h,w)
        base_score: hog representation of the object you want to find, an array of size (m,)
        stepSize: an int of the step size to move the window
        windowSize: a pair of ints that is the height and width of the window
    Returns:
        max_score: float of the highest hog score
        maxr: int of row where the max_score is found
        maxc: int of column where the max_score is found
        response_map: an np array of size (h,w)
    '''
    # slide a window across the image
    (max_score, maxr, maxc) = (0, 0, 0)
    winH, winW = windowSize
    H, W = image.shape
    pad_image = np.lib.pad(image, ((winH // 2, winH - winH // 2),
                                   (winW // 2, winW - winW // 2)), mode='constant')
    response_map = np.zeros((H // stepSize + 1, W // stepSize + 1))
    O, P = pad_image.shape
    # YOUR CODE HERE
    h = 0
    response_score = np.zeros(((H // stepSize + 1) * (W // stepSize + 1), 1))
    # response_score = np.zeros(1,)
    i = range(0, O - winH, stepSize)
    j = range(0, P - winW, stepSize)
    for m in i:
        for n in j:
            response = pad_image[m:m + winH, n: n + winW]
            # response = np.reshape(response, (1, winH * winW))
            window_feature, window_image = hog(response, pixels_per_cell=(
                pixel_per_cell, pixel_per_cell), visualise=True)
            response = np.dot(window_feature.T, base_score)
            response_score[h] = response
            h += 1
            if(response > max_score):
                max_score = response
                maxr = m
                maxc = n
            pass
        pass
    pass
    response_map = np.reshape(
        response_score, (H // stepSize + 1, W // stepSize + 1))
    response_map = resize(response_map, (H, W))
    # END YOUR CODE

    return (max_score, maxr, maxc, response_map)


def pyramid(image, scale=0.9, minSize=(200, 100)):
    '''
    Generate image pyramid using the given image and scale.
    Reducing the size of the image until on of the height or
    width reaches the minimum limit. In the ith iteration,
    the image is resized to scale^i of the original image.

    Args:
        image: np array of (h,w), an image to scale
        scale: float of how much to rescale the image each time
        minSize: pair of ints showing the minimum height and width

    Returns:
        images: a list containing pair of
            (the current scale of the image, resized image)
    '''
    # yield the original image
    images = []
    current_scale = 1.0
    images.append((current_scale, image))
    # keep looping over the pyramid
    # YOUR CODE HERE
    re_scale = scale
    re_image = image
    for i in range(100):
        H, W = re_image.shape
        if((H, W) > minSize):
            re_image = rescale(re_image, re_scale)
            images.append((re_scale, re_image))
            re_scale = re_scale * scale
        else:
            break

    # END YOUR CODE
    return images


# image_path = 'image_0001.jpg'

# image = io.imread(image_path, as_grey=True)
# image = rescale(image, 1.2)

# images = pyramid(image, scale=0.9)
# sum_r = 0
# sum_c = 0
# for i, result in enumerate(images):
#     (scale, image) = result
#     if (i == 0):
#         sum_c = image.shape[1]
#     sum_r += image.shape[0]

# composite_image = np.zeros((sum_r, sum_c))

# pointer = 0
# for i, result in enumerate(images):
#     (scale, image) = result
#     composite_image[pointer:pointer + image.shape[0], :image.shape[1]] = image
#     pointer += image.shape[0]

# plt.imshow(composite_image)
# plt.axis('off')
# plt.title('image pyramid')
# plt.show()


def pyramid_score(image, base_score, shape, stepSize=20, scale=0.9, pixel_per_cell=8):
    '''
    Calculate the maximum score found in the image pyramid using slding window.

    Args:
        image: np array of (h,w)
        base_score: the hog representation of the object you want to detect
        shape: shape of window you want to use for the sliding_window

    Returns:
        max_score: float of the highest hog score
        maxr: int of row where the max_score is found
        maxc: int of column where the max_score is found
        max_scale: float of scale when the max_score is found
        max_response_map: np array of the response map when max_score is found
    '''
    max_score = 0
    maxr = 0
    maxc = 0
    max_scale = 1.0
    max_response_map = np.zeros(image.shape)
    images = pyramid(image, scale)
    # YOUR CODE HERE
    H, W = image.shape
    for i, result in enumerate(images):
        (scale, image) = result
        score, r, c, max_response_map = sliding_window(
            image, base_score, stepSize, shape, pixel_per_cell)
        if(score > max_score):
            max_score = score
            maxr = r - 60
            maxc = c - 250
            max_scale = scale
            re_response = max_response_map
    max_response_map = resize(re_response, (H, W))
    # END YOUR CODE
    return max_score, maxr, maxc, max_scale, max_response_map


def read_facial_labels(image_paths):
    label_path = "list_landmarks_align_celeba.txt"
    n_images = len(image_paths)
    f = open(label_path, "r")
    f.readline()
    f.readline()
    lefteyes = np.array([], dtype=np.int).reshape(0, 2)
    righteyes = np.array([], dtype=np.int).reshape(0, 2)
    noses = np.array([], dtype=np.int).reshape(0, 2)
    mouths = np.array([], dtype=np.int).reshape(0, 2)
    for line in f:
        if lefteyes.shape[0] > 40:
            break
        parts = line.strip().split(' ')
        parts = list(filter(None, parts))
        # print(line,parts)
        image_file = parts[0]
        if image_file in image_paths:
            lefteye_c = int(parts[1])
            lefteye_r = int(parts[2])
            righteye_c = int(parts[3])
            righteye_r = int(parts[4])
            nose_c = int(parts[5])
            nose_r = int(parts[6])
            leftmouth_c = int(parts[7])
            leftmouth_r = int(parts[8])
            rightmouth_c = int(parts[9])
            rightmouth_r = int(parts[10])
            mouth_c = int((leftmouth_c + rightmouth_c) / 2)
            mouth_r = int((leftmouth_r + rightmouth_r) / 2)

            lefteyes = np.vstack(
                (lefteyes, np.asarray([lefteye_r, lefteye_c])))
            righteyes = np.vstack(
                (righteyes, np.asarray([righteye_r, righteye_c])))
            noses = np.vstack((noses, np.asarray([nose_r, nose_c])))
            mouths = np.vstack((mouths, np.asarray([mouth_r, mouth_c])))
    parts = (lefteyes, righteyes, noses, mouths)
    return parts


def get_detector(part_h, part_w, parts, image_paths):
    n = len(image_paths)
    part_shape = (part_h, part_w)
    avg_part = np.zeros((part_shape))
    for i, image_path in enumerate(image_paths):
        image = io.imread('./face/' + image_path, as_grey=True)
        part_r = parts[i][0]
        part_c = parts[i][1]
        # print(image_path, part_r, part_w, part_r-part_h/2, part_r+part_h/2)
        part_image = image[int(part_r - part_h / 2):int(part_r + part_h / 2),
                           int(part_c - part_w / 2):int(part_c + part_w / 2)]
        avg_part = np.asarray(part_image) + np.asarray(avg_part)
    avg_part = avg_part / n
    return avg_part


(winH, winW) = face_shape

max_score, maxr, maxc, max_scale, max_response_map = pyramid_score(
    image, face_feature, face_shape, stepSize=30, scale=0.8)


image_paths = fnmatch.filter(os.listdir('./face'), '*.jpg')

parts = read_facial_labels(image_paths)
lefteyes, righteyes, noses, mouths = parts
print(lefteyes)
# Typical shape for left eye
lefteye_h = 10
lefteye_w = 20

lefteye_shape = (lefteye_h, lefteye_w)

avg_lefteye = get_detector(lefteye_h, lefteye_w, lefteyes, image_paths)
(lefteye_feature, lefteye_hog) = hog(
    avg_lefteye, pixels_per_cell=(2, 2), visualise=True)

righteye_h = 10
righteye_w = 20

righteye_shape = (righteye_h, righteye_w)

avg_righteye = get_detector(righteye_h, righteye_w, righteyes, image_paths)

(righteye_feature, righteye_hog) = hog(
    avg_righteye, pixels_per_cell=(2, 2), visualise=True)

nose_h = 30
nose_w = 26

nose_shape = (nose_h, nose_w)

avg_nose = get_detector(nose_h, nose_w, noses, image_paths)

(nose_feature, nose_hog) = hog(avg_nose, pixels_per_cell=(2, 2), visualise=True)

mouth_h = 20
mouth_w = 36

mouth_shape = (mouth_h, mouth_w)

avg_mouth = get_detector(mouth_h, mouth_w, mouths, image_paths)

(mouth_feature, mouth_hog) = hog(avg_mouth, pixels_per_cell=(2, 2), visualise=True)

detectors_list = [lefteye_feature,
                  righteye_feature, nose_feature, mouth_feature]


def compute_displacement(part_centers, face_shape):
    ''' Calculate the mu and sigma for each part. d is the array
        where each row is the main center (face center) minus the
        part center. Since in our dataset, the face is the full
        image, face center could be computed by finding the center
        of the image. Vector mu is computed by taking an average from
        the rows of d. And sigma is the standard deviation among
        among the rows. Note that the heatmap pixels will be shifted
        by an int, so mu is an int vector.

    Args:
        part_centers: np array of shape (n,2) containing centers
            of one part in each image
        face_shape: (h,w) that indicates the shape of a face
    Returns:
        mu: (1,2) vector
        sigma: (1,2) vector

    '''
    d = np.zeros((part_centers.shape[0], 2))
    # YOUR CODE HERE
    H = (face_shape[0] - 1) / 2.0
    W = (face_shape[1] - 1) / 2.0
    d = np.subtract([H, W], part_centers)
    mu = np.mean(d, axis=0)
    mu = mu.astype(int)
    sigma = np.std(mu)
    # END YOUR CODE
    return mu, sigma


lefteye_mu, lefteye_std = compute_displacement(lefteyes, face_shape)
righteye_mu, righteye_std = compute_displacement(righteyes, face_shape)
nose_mu, nose_std = compute_displacement(noses, face_shape)
mouth_mu, mouth_std = compute_displacement(mouths, face_shape)
print(lefteye_mu, righteye_mu, nose_mu, mouth_mu)
image_path = 'image_0338.jpg'
image = io.imread(image_path, as_grey=True)
image = rescale(image, 1.0)

(face_H, face_W) = face_shape
max_score, face_r, face_c, face_scale, face_response_map = pyramid_score(
    image, face_feature, face_shape, stepSize=30, scale=0.8)

# plt.imshow(face_response_map, cmap='viridis', interpolation='nearest')
# plt.axis('off')
# plt.show()

max_score, lefteye_r, lefteye_c, lefteye_scale, lefteye_response_map = \
    pyramid_score(image, lefteye_feature, lefteye_shape,
                  stepSize=20, scale=0.9, pixel_per_cell=2)

lefteye_response_map = resize(lefteye_response_map, face_response_map.shape)
print(lefteye_response_map)
# plt.imshow(lefteye_response_map, cmap='viridis', interpolation='nearest')
# plt.axis('off')
# plt.show()

max_score, righteye_r, righteye_c, righteye_scale, righteye_response_map = \
    pyramid_score(image, righteye_feature, righteye_shape,
                  stepSize=20, scale=0.9, pixel_per_cell=2)

righteye_response_map = resize(righteye_response_map, face_response_map.shape)

# plt.imshow(righteye_response_map, cmap='viridis', interpolation='nearest')
# plt.axis('off')
# plt.show()

max_score, nose_r, nose_c, nose_scale, nose_response_map = \
    pyramid_score(image, nose_feature, nose_shape,
                  stepSize=20, scale=0.9, pixel_per_cell=2)

nose_response_map = resize(nose_response_map, face_response_map.shape)

# plt.imshow(nose_response_map, cmap='viridis', interpolation='nearest')
# plt.axis('off')
# plt.show()

max_score, mouth_r, mouth_c, mouth_scale, mouth_response_map =\
    pyramid_score(image, mouth_feature, mouth_shape,
                  stepSize=20, scale=0.9, pixel_per_cell=2)

# mouth_response_map = resize(mouth_response_map, face_response_map.shape)
# plt.imshow(mouth_response_map, cmap='viridis', interpolation='nearest')
# plt.axis('off')
# plt.show()


def shift_heatmap(heatmap, mu):
    '''First normalize the heatmap to make sure that all the values
        are not larger than 1.
        Then shift the heatmap based on the vector mu.

        Args:
            heatmap: np array of (h,w)
            mu: vector array of (1,2)
        Returns:
            new_heatmap: np array of (h,w)
    '''
    # YOUR CODE HERE
    max_ = np.max(heatmap)
    min_ = np.min(heatmap)
    H, W = heatmap.shape
    new_heatmap = heatmap
    for i in range(H):
        for j in range(W):
            heatmap[i][j] = (heatmap[i][j] - min_) / (max_ - min_)
            pass
        pass
    pass
    for i in range(abs(mu[0]), H - abs(mu[0])):
        for j in range(abs(mu[1]), W - abs(mu[1])):
            new_heatmap[i][j] = heatmap[i + mu[0]][j + mu[1]]
    # END YOUR CODE
    return new_heatmap


face_heatmap_shifted = shift_heatmap(face_response_map, [0, 0])
print(face_heatmap_shifted.shape)

lefteye_heatmap_shifted = shift_heatmap(lefteye_response_map, lefteye_mu)
print(lefteye_heatmap_shifted.shape)
righteye_heatmap_shifted = shift_heatmap(righteye_response_map, righteye_mu)
nose_heatmap_shifted = shift_heatmap(nose_response_map, nose_mu)
mouth_heatmap_shifted = shift_heatmap(mouth_response_map, mouth_mu)

f, axarr = plt.subplots(2, 2)
axarr[0, 0].axis('off')
axarr[0, 1].axis('off')
axarr[1, 0].axis('off')
axarr[1, 1].axis('off')
axarr[0, 0].imshow(lefteye_heatmap_shifted,
                   cmap='viridis', interpolation='nearest')
axarr[0, 1].imshow(righteye_heatmap_shifted,
                   cmap='viridis', interpolation='nearest')
axarr[1, 0].imshow(nose_heatmap_shifted, cmap='viridis',
                   interpolation='nearest')
axarr[1, 1].imshow(mouth_heatmap_shifted, cmap='viridis',
                   interpolation='nearest')
# plt.show()


def gaussian_heatmap(heatmap_face, heatmaps, sigmas):
    '''
    Apply gaussian filter with the given sigmas to the corresponding heatmap.
    Then add the filtered heatmaps together with the face heatmap.
    Find the index where the maximum value in the heatmap is found.

    Hint: use gaussian function provided by skimage

    Args:
        image: np array of (h,w)
        sigma: sigma for the gaussian filter
    Return:
        new_image: an image np array of (h,w) after gaussian convoluted
    '''
    # YOUR CODE HERE
    O, P = heatmap_face.shape
    heatmap = np.zeros((O, P))
    for i in range(4):
        heatmap += gaussian(heatmaps[i], sigmas[i])
    heatmap += heatmap_face
    H, W = heatmap.shape
    num = np.argmax(heatmap)
    r = (num / W).astype(int)
    c = num % W
    # END YOUR CODE
    return heatmap, r, c


heatmap_face = face_heatmap_shifted

heatmaps = [lefteye_heatmap_shifted,
            righteye_heatmap_shifted,
            nose_heatmap_shifted,
            mouth_heatmap_shifted]
sigmas = [lefteye_std, righteye_std, nose_std, mouth_std]

heatmap, i, j = gaussian_heatmap(heatmap_face, heatmaps, sigmas)

fig, ax = plt.subplots(1)
rect = patches.Rectangle((j - winW // 2, i - winH // 2),
                         winW, winH, linewidth=1, edgecolor='r', facecolor='none')
ax.add_patch(rect)

plt.imshow(heatmap, cmap='viridis', interpolation='nearest')
plt.axis('off')
plt.show()

fig, ax = plt.subplots(1)
rect = patches.Rectangle((j - winW // 2, i - winH // 2),
                         winW, winH, linewidth=1, edgecolor='r', facecolor='none')
ax.add_patch(rect)

plt.imshow(resize(image, heatmap.shape))
plt.axis('off')
plt.show()


def get_heatmap(image, face_feature, face_shape, detectors_list, parts):
    _, _, _, _, face_response_map = pyramid_score(
        image, face_feature, face_shape, stepSize=30, scale=0.8)
    face_response_map = resize(face_response_map, image.shape)
    face_heatmap_shifted = shift_heatmap(face_response_map, [0, 0])
    for i, detector in enumerate(detectors_list):
        part = parts[i]
        max_score, r, c, scale, response_map = pyramid_score(
            image, face_feature, face_shape, stepSize=30, scale=0.8)
        mu, std = compute_displacement(part, face_shape)
        response_map = resize(response_map, face_response_map.shape)
        response_map_shifted = shift_heatmap(response_map, mu)
        heatmap = gaussian(response_map_shifted, std)
        face_heatmap_shifted += heatmap
    return face_heatmap_shifted


def detect_multiple(image, response_map):
    '''
    Extra credit
    '''
    # YOUR CODE HERE
    pass
    # END YOUR CODE
    return detected_faces


image_path = 'image_0002.jpg'
image = io.imread(image_path, as_grey=True)
plt.imshow(image)
plt.show()

image_path = 'image_0002.jpg'
image = io.imread(image_path, as_grey=True)
heatmap = get_heatmap(image, face_feature, face_shape, detectors_list, parts)

plt.imshow(heatmap, cmap='viridis', interpolation='nearest')
plt.show()
