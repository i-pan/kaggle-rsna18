###########
# IMPORTS #
###########

import numpy as np 
import scipy.misc 
import subprocess 
import sys
import os, json 

WDIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(WDIR, "../../SETTINGS.json")) as f: 
    SETTINGS_JSON = json.load(f)  

from scipy.ndimage.interpolation import zoom, rotate
from scipy.ndimage.filters import gaussian_filter
from skimage import exposure, io 

#############
# FUNCTIONS #
#############

def resize_image(img, size, smooth=None):
  """
  Resizes image to new_length x new_length and pads with black. 
  Only works with grayscale right now. 

  Arguments:
    - smooth (float/None) : sigma value for Gaussian smoothing
  """
  resize_factor = float(size) / np.max(img.shape)
  if resize_factor > 1: 
    # Cubic spline interpolation
    resized_img = zoom(img, resize_factor)
  else:
    # Linear interpolation 
    resized_img = zoom(img, resize_factor, order=1, prefilter=False)
  if smooth is not None: 
    resized_img = gaussian_filter(resized_img, sigma=smooth) 
  l = resized_img.shape[0] ; w = resized_img.shape[1] 
  if l != w: 
    ldiff = (size-l) / 2 
    wdiff = (size-w) / 2
    pad_list = [(ldiff, size-l-ldiff), (wdiff, size-w-wdiff)] 
    resized_img = np.pad(resized_img, pad_list, "constant", 
                         constant_values=0)
  assert size == resized_img.shape[0] == resized_img.shape[1]
  return resized_img.astype("uint8")

def resize_images_and_save_as_nparray(list_of_images, in_dir, out_dir, new_size=256):
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    num_images = len(list_of_images)  
    for index, img in enumerate(list_of_images): 
        sys.stdout.write("Resizing {}/{} ...\r".format(index+1, num_images))
        sys.stdout.flush()
        loaded_img = scipy.misc.imread(os.path.join(in_dir, img), mode="L")
        resized_img = resize_image(loaded_img, new_size) 
        np.save(os.path.join(out_dir, img.replace("png", "npy")), resized_img) 

def pad_image(img, size, smooth=None):
  """
  Pads image to new_length x new_length and pads with black. 
  Only works with grayscale right now. 

  Arguments:
    - smooth (float/None) : sigma value for Gaussian smoothing
  """
  if np.max(img.shape) > size: 
    resize_factor = float(size) / np.max(img.shape)
    # Linear interpolation 
    resized_img = zoom(img, resize_factor, order=1, prefilter=False)
  else: 
    resized_img = img.copy()
  if smooth is not None: 
    resized_img = gaussian_filter(resized_img, sigma=smooth) 
  l = resized_img.shape[0] ; w = resized_img.shape[1]   
  ldiff = (size-l) / 2 
  wdiff = (size-w) / 2
  pad_list = [(ldiff, size-l-ldiff), (wdiff, size-w-wdiff)] 
  resized_img = np.pad(resized_img, pad_list, "constant", 
                       constant_values=0)
  assert size == resized_img.shape[0] == resized_img.shape[1]
  return resized_img.astype("uint8")

def pad_images_and_save_as_nparray(list_of_images, in_dir, out_dir, new_size=256):
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    num_images = len(list_of_images)  
    for index, img in enumerate(list_of_images): 
        sys.stdout.write("Resizing {}/{} ...\r".format(index+1, num_images))
        sys.stdout.flush()
        loaded_img = scipy.misc.imread(os.path.join(in_dir, img), mode="L")
        resized_img = pad_image(loaded_img, new_size) 
        np.save(os.path.join(out_dir, img.split(".")[0]+".npy"), resized_img) 

##########
# SCRIPT #
##########

in_dir  = os.path.join(WDIR, "../../", SETTINGS_JSON["TRAIN_IMAGES_CLEAN_DIR"], "orig")

out_dir = os.path.join(WDIR, "../../", SETTINGS_JSON["TRAIN_IMAGES_CLEAN_DIR"], "resized/i256/")
list_of_images = subprocess.check_output("ls " + in_dir, shell=True).split() 
resize_images_and_save_as_nparray(list_of_images, in_dir, out_dir, 256)

out_dir = os.path.join(WDIR, "../../", SETTINGS_JSON["TRAIN_IMAGES_CLEAN_DIR"], "resized/i320/")
list_of_images = subprocess.check_output("ls " + in_dir, shell=True).split() 
resize_images_and_save_as_nparray(list_of_images, in_dir, out_dir, 320)

out_dir = os.path.join(WDIR, "../../", SETTINGS_JSON["TRAIN_IMAGES_CLEAN_DIR"], "resized/i384/")
list_of_images = subprocess.check_output("ls " + in_dir, shell=True).split() 
resize_images_and_save_as_nparray(list_of_images, in_dir, out_dir, 384)

out_dir = os.path.join(WDIR, "../../", SETTINGS_JSON["TRAIN_IMAGES_CLEAN_DIR"], "resized/i448/")
list_of_images = subprocess.check_output("ls " + in_dir, shell=True).split() 
resize_images_and_save_as_nparray(list_of_images, in_dir, out_dir, 448)

out_dir = os.path.join(WDIR, "../../", SETTINGS_JSON["TRAIN_IMAGES_CLEAN_DIR"], "resized/i512/")
list_of_images = subprocess.check_output("ls " + in_dir, shell=True).split() 
resize_images_and_save_as_nparray(list_of_images, in_dir, out_dir, 512)

