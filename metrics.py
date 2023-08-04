"""
1. normaliziraj napovedi na [0, 1] (in ne [0, 255])
2. izračunaj "vse" smiselne metrike:
   - mse
   - intersection over union
   - (averege) precision
   - (average) recall
"""

import numpy as np
import cv2

# presek / unija
def intersection_over_union(img1, img2):
    intersection = np.minimum(img1, img2)
    union = np.maximum(img1, img2)
    I = np.sum(intersection)
    U = np.sum(union)
    return I / U if U != 0 else 1   # U=0 je redek primer na slovenskih cestah, ampak ga je treba upoštevat :)

# "natančnost prepoznavanja"
def precision(true, predicted, threshold):
    img1_t = true > threshold
    img2_t = predicted > threshold
    intersection = np.minimum(img1_t, img2_t)
    return np.sum(intersection) / np.sum(img1_t)

# "uspešnost pri iskanju pravih ojbektov"
def recall(true, predicted, threshold):
    """
    :param true: ground truth numpy array with values from [0, 1]
    :param predicted: predicted numpy array with values from [0, 1]
    :param threshold: threshold for binarization
    """
    true_t = true > threshold
    pred_t = predicted > threshold
    intersection = np.minimum(true_t, pred_t)
    return np.sum(intersection) / np.sum(pred_t)

# returns mean square error
def mse(im1, im2) :
    h, w, rgb = im1.shape
    diff = cv2.subtract(im1, im2)
    err = np.sum(diff**2)
    mse = err/(float(h*w))
    return mse