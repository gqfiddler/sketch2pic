import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import skimage
import glob
from skimage import io
from skimage.transform import resize
from time import time
from skimage.feature import canny
from skimage.color import rgb2gray

def safe_to_crop(img):
    # only checks sides, because top/bottom are never a problem in this dset
    # allows buffer of 2000 for shadows etc.
    if np.sum(img[:,:4], axis=(0,1,2)) > 136*4*3*255 - 2000 \
    and np.sum(img[:,-4:], axis=(0,1,2)) > 136*4*3*255 - 2000:
        return True
    return False

def img_to_sketch(img, sigma=2.0):
    white_outline = canny(rgb2gray(img), sigma=sigma)
    black_outline = np.invert(white_outline)
    return(black_outline)

def display_transform(filename):
    img = io.imread(filename)
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    sketch = img_to_sketch(img)
    plt.imshow(sketch, cmap='gray')
    plt.show()

def get_complexity(img):
    vertical = sum(
        np.clip(np.invert(img).sum(axis=0), 2, 8) - 2
    )
    horizontal = sum(
        np.clip(np.invert(img).sum(axis=1), 2, 8) - 2
    )
    return vertical + horizontal

def display_samples(img_pairs, num_samples=50):
    '''num_samples should be multiple of five'''
    selection = np.random.choice(range(len(img_pairs)), size=num_samples, replace=False)
    rows = []
    for i in range(num_samples//5):
        for j in range(5):
            idx = 5*i + j
            if j == 0:
                row = img_pairs[selection[idx]][1]
            else:
                row = np.concatenate((row, img_pairs[selection[idx]][1]), axis=1)
        if i == 0:
            rows = row
        else:
            rows = np.concatenate((rows, row), axis=0)

    plt.figure(figsize=(10,num_samples))
    plt.imshow(rows, cmap='gray')
    plt.show()

def side_by_side(img1, img2):
    plt.subplot(121)
    plt.imshow(img1)
    plt.subplot(122)
    plt.imshow(img2)
    plt.show()


def load_data(img_filenames, cap=-1, min_complexity=450, \
              max_complexity=750, verbose=True):

    error_count = 0
    img_pairs = []
    start = time()
    for i, filename in enumerate(img_filenames[:cap]):
        if verbose:
            print('\rProcessing image {} of {}'.format(i, len(img_filenames[:cap])), end='')
        img = io.imread(filename)
        if img.shape != (136, 136, 3):
            error_count += 1
            continue
        if safe_to_crop(img):
            img = img[4:-4, 4:-4]
            img = img/255 # converts to (0, 1) floats
        else:
            img = resize(img, (128,128,3), mode='reflect', anti_aliasing=True)
            # resizing automatically converts to (0,1) floats
        sigma = 2.0
        sketch = img_to_sketch(img, sigma)
        # check for appropriate complexity; adjust as necessary
        adjustments = 0
        while adjustments < 5:
            complexity = get_complexity(sketch)
            if complexity < min_complexity:
                sigma -= 0.25
                sketch = img_to_sketch(img, sigma=sigma)
            elif complexity > max_complexity:
                sigma += 0.25
                sketch = img_to_sketch(img, sigma=sigma)
            else:
                break
            adjustments += 1
        img_pairs.append(
            (img, sketch)
        )
    if verbose:
        print('\nElapsed time: ', round((time()-start)/60, 1), 'minutes')
        print('Image dimension errors: ', error_count)
        print(len(img_pairs), 'total images processed')

    return img_pairs
