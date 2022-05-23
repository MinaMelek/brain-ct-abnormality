# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 14:24:51 2022

@author: Mina
"""
import os.path
import cupy as cp
import pydicom
from cupyx.scipy import ndimage
from cucim.skimage.transform import resize  # , rotate
from cucim.skimage import morphology
from skimage.io import imread as sk_imread
# import cv2
from prepare_dicom import prep_pipeline


def remove_noise(brain_image, create_mask=False):
    """
    Removes noise from the CT brain image like artifacts, pillow, etc... 
    
    Parameters:
        
        **brain_image** -- a 2d (numpy array) image representing a slice of a brain CT scan.
        
        **create_mask** -- a bool variable for whether to use a threshold-based mask or not (used in specific cases),
        default = False.
    
    Notes:
      * "morphology.dilation" creates a segmentation of the image
        If one pixel is between the origin and the edge of a square of size
        5x5, the pixel belongs to the same class

      * We can instead use a circule using: morphology.disk(2)
        In this case the pixel belongs to the same class if it's between the origin
        and the radius
    """
    
    # Deprecated
    if create_mask:  
        # img = cv2.merge([brain_image, brain_image, brain_image])
        # mask = cv2.threshold(img, 210, 255, cv2.THRESH_BINARY)[1][:, :, 0]
        mask = brain_image
    else:
        mask = brain_image

    brain_image = cp.array(brain_image)  # convert to cupy array type
    segmentation = morphology.dilation(brain_image, cp.ones((5, 5)))
    labels, label_nb = ndimage.label(segmentation)
    label_count = cp.bincount(labels.ravel().astype(int))
    # The size of label_count is the number of classes/segmentations found
    # We don't use the first class since it's the background
    label_count[0] = 0
    # We create a mask with the class with more pixels
    # In this case should be the brain
    mask = labels == label_count.argmax()
    # Improve the brain mask
    mask = morphology.dilation(mask, cp.ones((5, 5)))
    mask = ndimage.morphology.binary_fill_holes(mask)
    mask = morphology.dilation(mask, cp.ones((3, 3)))
    # Since the pixels in the mask are zero's and one's
    # We can multiply the original image to only keep the brain region
    masked_image = mask * brain_image

    return masked_image


def crop_image(image):
    """
    Crops the relevant part of the image from the background.
    
    Parameters:
        **image** -- a 2d (numpy array) image representing a slice of a brain CT scan (preferable to be de-noised)
    """
    # Create a mask with the background pixels
    mask = image == 0
    # Find the brain area
    coords = cp.array(cp.nonzero(~mask))
    if coords.size == 0:
        return image
    top_left = cp.min(coords, axis=1)
    bottom_right = cp.max(coords, axis=1)
    # Remove the background
    cropped_image = image[top_left[0]:bottom_right[0],
                          top_left[1]:bottom_right[1]]
    return cropped_image


def resize_to_scale(image, new_height=512, new_width=512):
    height, width = image.shape  # original size
    # Calculate size preserving the aspect ratio
    resized_height = min(new_height, height*new_width//width)
    resized_width = min(new_width, width*new_height//height)
    image = resize(image, output_shape=(resized_height, resized_width))
    return image


def add_pad(image, new_height=512, new_width=512, to_scale=True):
    """
    Pad the image with zeros to fill the new shape.
    
    Parameters:
        **image** -- a 2d (numpy array) cropped image.
        **new_height** -- the image new height after resize, default: 512.
        **new_width** -- the image new width after resize, default: 512.
        **to_scale** -- a bool variable; If true, resize the image to fill the new size
                        instead of filling it with zeros. Center and fill with zeros if not, default: True.
    """
    if to_scale: 
        image = resize_to_scale(image, new_height, new_width)
    # Pad the rest to fill the new size
    height, width = image.shape
    final_image = cp.zeros((new_height, new_width))
    pad_left = int((new_width - width) / 2)
    pad_top = int((new_height - height) / 2)
    # Replace the pixels with the image's pixels
    final_image[pad_top:pad_top + height, pad_left:pad_left + width] = image
    return final_image


def zoom_on_image(image, im_size=(512, 512), zoom=True):
    # if zoom is False, only center the image
    image = crop_image(image)
    image = add_pad(image, *im_size, zoom) 
    return image


def window(slice_s, w_level=40, w_width=120):
    # Apply brain window
    slice_s = cp.asarray(slice_s)
    w_min = w_level - w_width / 2
    w_max = w_level + w_width / 2
    slice_s = (slice_s - w_min)*(255/(w_max-w_min))  # or slice_s = (slice_s - (w_level-(w_width/2)))*(255/(w_width))
    slice_s[slice_s < 0] = 0
    slice_s[slice_s > 255] = 255
    return slice_s


def window_slice(slice_s, w_level=40, w_width=120, rotate=False, size=None):
    # Filter and enhance the brain tissue to the brain window
    size = slice_s.shape if size is None else size
    slice_s = window(slice_s, w_level, w_width)
    if rotate:
        slice_s = cp.rot90(slice_s)  # rotate for horizontal scans
    slice_s = remove_noise(slice_s)  # remove artifacts
    slice_s = zoom_on_image(slice_s, size)  # center and pad to size

    return slice_s


def standardize(im):
    """
    rescale image to normal distribution
    :im: original image
    :return: standardized (normal) image
    """
    im_norm = (im - im.mean()) / im.std()  # Standardize image
    im_norm = im_norm.transpose(2, 0, 1)  # Move channel first
    return im_norm


def load_batch(im_files):
    """
    load images (png) from a list of paths
    :im_files: a list of image paths (a directory files)
    :return: a list of image arrays
    """
    im_list = []
    for im_file in im_files:
        im = sk_imread(str(im_file))[:, :, :3]  # read RGB image
        # Get image into shape
        im_norm = standardize(im)
        im_list.append(im_norm)
    im_list = cp.array(im_list)
    return im_list


def download_data(path):  # TODO
    # Placeholder
    assert os.path.isdir(path), f"{path} is not a directory!"
    return path


def slice_preprocess(file_name, new_size):
    """
    Apply all preprocessing steps on image (Dicom slice) before prediction
    :param file_name: Dicom file name
    :param new_size: The size of the output image (input shape to model)
    :return: A preprocessed image ready for prediction
    """
    # Read Dicom file.
    img_dcm = pydicom.dcmread(file_name)

    # Note: Make sure the selected study is for a brain.
    # Apply window and slope information from Dicom metadata.
    try:
        image = prep_pipeline(img=img_dcm)
    except AttributeError as e:
        # If Dicom windowing information were unavailable.
        if 'window' in str(e).split()[-1].lower():
            print("didn't find window attributes")
            image = window(img_dcm.pixel_array, w_level=40, w_width=120)
        else:
            raise e

    # Remove noise from the CT slice
    image = remove_noise(image)
    # Center and zoom into the brain.
    image = zoom_on_image(image, new_size)
    # Convert to rgb image
    image = cp.concatenate([image[:, :, cp.newaxis], image[:, :, cp.newaxis], image[:, :, cp.newaxis]], 2)
    # standardize images
    image = standardize(image)

    return image


