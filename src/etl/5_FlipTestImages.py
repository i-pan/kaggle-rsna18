import scipy.misc 
import numpy as np
import glob, sys
import os, json 

WDIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(WDIR, "../../SETTINGS.json")) as f: 
    SETTINGS_JSON = json.load(f) 

TEST_IMAGES_DIR = os.path.join(WDIR, "../..", SETTINGS_JSON["TEST_IMAGES_CLEAN_DIR"], "orig")
FLIP_IMAGES_DIR = os.path.join(WDIR, "../../", SETTINGS_JSON["TEST_IMAGES_CLEAN_DIR"], "flip")

if not os.path.exists(FLIP_IMAGES_DIR): os.makedirs(FLIP_IMAGES_DIR)

test_images = glob.glob(os.path.join(TEST_IMAGES_DIR, "*"))

for index, imgfile in enumerate(test_images): 
    sys.stdout.write("Flipping {}/{} ...\r".format(index+1, len(test_images)))
    sys.stdout.flush()
    img = scipy.misc.imread(imgfile) 
    img = np.fliplr(img) 
    scipy.misc.imsave(os.path.join(FLIP_IMAGES_DIR, imgfile.split("/")[-1]), img)

