import scipy.misc
import numpy as np


def save_merged_images(images, size, path):
    """ This function concatenate multiple images and saves them as a single image.

    Args:
        images: images to concatenate
        size: number of columns and rows of images to be concatenated
        path: location to save merged image

    Returns:
        saves merged image in path
    """
    h, w = images.shape[1], images.shape[2]

    merge_img = np.zeros((h * size[0], w * size[1]))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = int(idx / size[1])
        merge_img[j * h:j * h + h, i * w:i * w + w] = image

    scipy.misc.imsave(path, merge_img)
