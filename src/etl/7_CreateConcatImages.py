#
# IMPORTS 
#

import pandas as pd 
import numpy as np 

import scipy.misc 
import glob 
import os, json 

WDIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(WDIR, "../../SETTINGS.json")) as f: 
    SETTINGS_JSON = json.load(f) 

from scipy.ndimage.interpolation import zoom 

#
# FX
#

def createConcatImages(folds_df, 
                       bbox_df, 
                       data_dir, 
                       concat_labels_file,
                       concat_img_save_dir):
    np.random.seed(88)
    labels_df = folds_df.merge(bbox_df, on="patientId") 
    concat_img_save_dir = os.path.join(concat_img_save_dir)
    if not os.path.exists(concat_img_save_dir): os.makedirs(concat_img_save_dir)
    images = glob.glob(os.path.join(data_dir, "*"))
    pos_images = [img for img in images if img.split("/")[-1].split(".")[0] in list(labels_df[labels_df["class"] == "Lung Opacity"].patientId)]
    neg_images = [img for img in images if img.split("/")[-1].split(".")[0] in list(labels_df[labels_df["class"] != "Lung Opacity"].patientId)]
    pos_images_by_fold = []
    for each_fold in range(10): 
        pos_images_by_fold.append([img for img in pos_images if img.split("/")[-1].split(".")[0] in list(labels_df[labels_df.fold == each_fold].patientId)])
    # TRAIN IMAGES #
    train_labels_df = pd.DataFrame() 
    train_file_paths = []
    for i, neg_img in enumerate(neg_images):
        print ("[TRAIN] Processing : {}/{} ...".format(i+1, len(neg_images))) 
        neg_img_id = neg_img.split("/")[-1].split(".")[0]
        neg_img_fold = labels_df[labels_df.patientId == neg_img_id].fold.iloc[0]
        neg_img = scipy.misc.imread(neg_img)
        # Randomly sample 1 image from same fold 
        pos_img  = np.random.choice(pos_images_by_fold[neg_img_fold])  
        pos_img_id = pos_img.split("/")[-1].split(".")[0]
        pos_img = scipy.misc.imread(pos_img)
        # Randomly pick position of negative image
        position = np.random.choice([1,2])
        if position == 1:
            # Left 
            new_img = np.concatenate((neg_img, pos_img), axis=1)
            tmp_df = labels_df[labels_df.patientId == pos_img_id]
            tmp_df["x"] = tmp_df["x"] + 1024 
            tmp_df["patientId"] = "ConcatenatedImage{}".format(str(i).zfill(5))
        elif position == 2:
            # Right
            new_img = np.concatenate((pos_img, neg_img), axis=1)
            tmp_df = labels_df[labels_df.patientId == pos_img_id]
            tmp_df["patientId"] = "ConcatenatedImage{}".format(str(i).zfill(5))
        assert new_img.shape == (1024, 2048) 
        tmp_file_path = os.path.join(concat_img_save_dir, "{}.png".format(tmp_df.patientId.iloc[0]))
        train_file_paths.append(tmp_file_path)
        # Resize
        # new_img = zoom(new_img, [0.5, 0.5], order=1, prefilter=False) 
        # assert new_img.shape == (1024, 1024) 
        scipy.misc.imsave(tmp_file_path, new_img) 
        train_labels_df = train_labels_df.append(tmp_df) 
    train_labels_df.to_csv(concat_labels_file, index=False) 
    # VALID IMAGES #
    # for i, img in enumerate(images): 
    #     print ("Processing : {}/{} ...".format(i+1, len(images)))
    #     img_id = img.split("/")[-1].split(".")[0]
    #     img = scipy.misc.imread(img) 
    #     new_img = np.concatenate((img, np.zeros(img.shape)), axis=1)
    #     assert new_img.shape == (1024, 2048) 
    #     scipy.misc.imsave(os.path.join(concat_img_save_dir, "{}.png".format(img_id)), new_img)

#
# SCRIPT 
#

data_dir = os.path.join(WDIR, "../..", SETTINGS_JSON["TRAIN_IMAGES_CLEAN_DIR"], "orig")

folds_df = pd.read_csv(os.path.join(WDIR, "../..", SETTINGS_JSON["TRAIN_INFO_DIR"], "stratified_folds_df.csv"))
bbox_df  = pd.read_csv(os.path.join(WDIR, "../..", SETTINGS_JSON["TRAIN_INFO_DIR"], "train_labels.csv"))

createConcatImages(folds_df, bbox_df, data_dir, 
                   os.path.join(WDIR, "../..", SETTINGS_JSON["TRAIN_INFO_DIR"], "concat_train_labels.csv"),
                   os.path.join(WDIR, "../..", SETTINGS_JSON["TRAIN_IMAGES_CLEAN_DIR"], "concat"))  




