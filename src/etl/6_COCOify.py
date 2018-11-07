import argparse 
import os, json 

WDIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(WDIR, "../../SETTINGS.json")) as f: 
    SETTINGS_JSON = json.load(f)  

parser = argparse.ArgumentParser()

parser.add_argument("subset", type=str) 
parser.add_argument("TRAIN_LABELS_PATH", nargs="?", type=str,
                    const=1,
                    default=os.path.join(WDIR, "../../", SETTINGS_JSON["TRAIN_INFO_DIR"], "train_labels.csv"))
parser.add_argument("TRAIN_IMAGES_DIR",  nargs="?", type=str,
                    const=1, 
                    default=os.path.join(WDIR, "../../", SETTINGS_JSON["TRAIN_IMAGES_CLEAN_DIR"], "orig"))
parser.add_argument("TEST_IMAGES_DIR",   nargs="?", type=str,
                    const=1, 
                    default=os.path.join(WDIR, "../../", SETTINGS_JSON["TEST_IMAGES_CLEAN_DIR"], "orig"))
parser.add_argument("FOLDS_DF_PATH",     nargs="?", type=str,
                    const=1, 
                    default=os.path.join(WDIR, "../../", SETTINGS_JSON["TRAIN_INFO_DIR"], "stratified_folds_df.csv")) 

args = parser.parse_args() 

subset = args.subset

TRAIN_LABELS_PATH = args.TRAIN_LABELS_PATH
TRAIN_IMAGES_DIR  = args.TRAIN_IMAGES_DIR
TEST_IMAGES_DIR   = args.TEST_IMAGES_DIR
FOLDS_DF_PATH     = args.FOLDS_DF_PATH

###########
# IMPORTS #
###########

import pandas as pd 
import numpy as np 
import scipy.misc 
import json 
import glob
import sys

#############
# FUNCTIONS #
#############

def COCOify(folds_df, 
            fold, 
            labels_df, 
            save_data_dir, 
            save_mapping_dir,
            save_instances_dir,
            pos_only=True,
            valid=True):
    train_df = folds_df[folds_df.fold != fold] 
    valid_df = folds_df[folds_df.fold == fold]
    #
    if pos_only:
        train_prefix = "train{}-pos".format(fold)
    else:
        train_prefix = "train{}".format(fold) 
    save_train_data_dir = os.path.join(save_data_dir, train_prefix)
    save_valid_data_dir = os.path.join(save_data_dir, "valid{}".format(fold))
    if not os.path.exists(save_train_data_dir): os.makedirs(save_train_data_dir)
    if not os.path.exists(save_valid_data_dir): os.makedirs(save_valid_data_dir)
    if not os.path.exists(save_mapping_dir): os.makedirs(save_mapping_dir)
    if not os.path.exists(save_instances_dir): os.makedirs(save_instances_dir)
    #
    if pos_only: 
        train_df = train_df[train_df["class"] == "Lung Opacity"]
    all_train_images = [os.path.join(TRAIN_IMAGES_DIR, "{}.png".format(_)) for _ in train_df.patientId]
    all_valid_images = [os.path.join(TRAIN_IMAGES_DIR, "{}.png".format(_)) for _ in valid_df.patientId]
    #
    # == TRAIN == # 
    rsna_train = {} 
    train_image_to_coco_dict = {} 
    # Images 
    images = [] 
    for i, imgfile in enumerate(all_train_images):
        sys.stdout.write("Processing images: {}/{} ...\r".format(i+1, len(all_train_images)))
        sys.stdout.flush() 
        tmp_dict = {}
        tmp_dict["file_name"] = "COCO_{}_{}.png".format(train_prefix, str(i).zfill(12))
        train_image_to_coco_dict[tmp_dict["file_name"]] = imgfile.split("/")[-1]
        os.system("cp {} {}".format(imgfile, os.path.join(save_train_data_dir, tmp_dict["file_name"])))
        tmp_dict["height"] = 1024 
        tmp_dict["width"]  = 1024 
        tmp_dict["id"] = i
        images.append(tmp_dict) 
    # Save mapping
    with open(os.path.join(save_mapping_dir, "{}_image_to_coco.json".format(train_prefix)), "w") as f: 
        json.dump(train_image_to_coco_dict, f)
    # Annotations
    annots = [] 
    # Take only positive cases
    counter = 0 
    labels_df = labels_df[labels_df.Target == 1]
    print ("\nProcessing annotations ...")
    for i, pid in enumerate(all_train_images): 
        pid = pid.split("/")[-1].split(".")[0] 
        tmp_df = labels_df[labels_df.patientId == pid] 
        if tmp_df.shape[0] == 0:
            continue
        for rownum, row in tmp_df.iterrows(): 
            tmp_dict = {} 
            tmp_dict["bbox"] = [row["x"], row["y"], row["width"], row["height"]]
            tmp_dict["image_id"] = i 
            tmp_dict["category_id"] = 1 
            tmp_dict["id"] = counter
            tmp_dict["area"] = row["width"] * row["height"] 
            tmp_dict["iscrowd"] = 0 
            counter += 1
            annots.append(tmp_dict)
    # Assemble into dictionary
    rsna_train["images"] = images 
    rsna_train["annotations"] = annots 
    rsna_train["categories"] = [{"supercategory": "opacity", "id": 1, "name": "opacity"}]
    # 
    with open(os.path.join(save_instances_dir, "instances_{}.json".format(train_prefix)), "w") as f: 
        json.dump(rsna_train, f) 
    if valid: 
        # == VALID == # 
        rsna_valid = {} 
        valid_image_to_coco_dict = {} 
        # Images 
        images = [] 
        for i, imgfile in enumerate(all_valid_images):
            sys.stdout.write("Processing images: {}/{} ...\r".format(i+1, len(all_valid_images)))
            sys.stdout.flush() 
            tmp_dict = {}
            tmp_dict["file_name"] = "COCO_valid{}_{}.png".format(fold, str(i).zfill(12))
            valid_image_to_coco_dict[tmp_dict["file_name"]] = imgfile.split("/")[-1]
            os.system("cp {} {}".format(imgfile, os.path.join(save_valid_data_dir, tmp_dict["file_name"])))
            tmp_dict["height"] = 1024 
            tmp_dict["width"]  = 1024 
            tmp_dict["id"] = i
            images.append(tmp_dict) 
        # Save mapping
        with open(os.path.join(save_mapping_dir, "valid{}_image_to_coco.json".format(fold)), "w") as f: 
            json.dump(valid_image_to_coco_dict, f)
        # Annotations
        annots = [] 
        # Take only positive cases
        counter = 0 
        labels_df = labels_df[labels_df.Target == 1]
        print ("\nProcessing annotations ...")
        for i, pid in enumerate(all_valid_images): 
            pid = pid.split("/")[-1].split(".")[0] 
            tmp_df = labels_df[labels_df.patientId == pid] 
            if tmp_df.shape[0] == 0:
                continue
            for rownum, row in tmp_df.iterrows(): 
                tmp_dict = {} 
                tmp_dict["bbox"] = [row["x"], row["y"], row["width"], row["height"]]
                tmp_dict["image_id"] = i 
                tmp_dict["category_id"] = 1 
                tmp_dict["id"] = counter
                tmp_dict["area"] = row["width"] * row["height"] 
                tmp_dict["iscrowd"] = 0 
                counter += 1
                annots.append(tmp_dict)
        # Assemble into dictionary
        rsna_valid["images"] = images 
        rsna_valid["annotations"] = annots 
        rsna_valid["categories"] = [{"supercategory": "opacity", "id": 1, "name": "opacity"}]
        # 
        with open(os.path.join(save_instances_dir, "instances_valid{}.json".format(fold)), "w") as f: 
            json.dump(rsna_valid, f) 

def COCOifyTest(test_data_dir,
                save_data_dir, 
                save_mapping_dir,
                save_instances_dir,
                prefix="test"):
    """
    """
    all_test_images = glob.glob(os.path.join(test_data_dir, "*"))
    save_data_dir = os.path.join(save_data_dir, prefix)
    if not os.path.exists(save_data_dir): os.makedirs(save_data_dir)
    if not os.path.exists(save_mapping_dir): os.makedirs(save_mapping_dir)
    if not os.path.exists(save_instances_dir): os.makedirs(save_instances_dir)
    # == TEST == # 
    rsna_test = {} 
    test_image_to_coco_dict = {} 
    # Images 
    images = [] 
    for i, imgfile in enumerate(all_test_images):
        sys.stdout.write("Processing images: {}/{} ...\r".format(i+1, len(all_test_images)))
        sys.stdout.flush() 
        tmp_dict = {}
        tmp_dict["file_name"] = "COCO_{}_{}.png".format(prefix, str(i).zfill(12))
        test_image_to_coco_dict[tmp_dict["file_name"]] = imgfile.split("/")[-1]
        os.system("cp {} {}".format(imgfile, os.path.join(save_data_dir, tmp_dict["file_name"])))
        tmp_dict["height"] = 1024 
        tmp_dict["width"]  = 1024 
        tmp_dict["id"] = i
        images.append(tmp_dict) 
    # Save mapping
    with open(os.path.join(save_mapping_dir, "{}_image_to_coco.json".format(prefix)), "w") as f: 
        json.dump(test_image_to_coco_dict, f)
    # Annotations
    annots = [] 
    # Take only positive cases
    print ("\nProcessing annotations ...")
    # Assemble into dictionary
    rsna_test["images"] = images 
    rsna_test["annotations"] = annots 
    rsna_test["categories"] = [{"supercategory": "opacity", "id": 1, "name": "opacity"}]
    # 
    with open(os.path.join(save_instances_dir, "image_info_{}.json".format(prefix)), "w") as f: 
        json.dump(rsna_test, f) 

##########
# SCRIPT #
##########

labels_df = pd.read_csv(TRAIN_LABELS_PATH)
folds_df  = pd.read_csv(FOLDS_DF_PATH)

if subset == "test": 
    COCOifyTest(TEST_IMAGES_DIR,
                os.path.join(WDIR, "../../data/coco/rsna/images/"),
                os.path.join(WDIR, "../../data/mappings/"),
                os.path.join(WDIR, "../../data/coco/rsna/annotations/"),
                "test")

    COCOifyTest(TEST_IMAGES_DIR.replace("orig", "flip"),
                os.path.join(WDIR, "../../data/coco/rsna/images/"),
                os.path.join(WDIR, "../../data/mappings/"),
                os.path.join(WDIR, "../../data/coco/rsna/annotations/"),
                "test_flip")

elif subset == "train": 
    COCOify(folds_df, 0, labels_df, 
            os.path.join(WDIR, "../../data/coco/rsna/images/"),
            os.path.join(WDIR, "../../data/mappings/"),
            os.path.join(WDIR, "../../data/coco/rsna/annotations/"))

    COCOify(folds_df, 1, labels_df, 
            os.path.join(WDIR, "../../data/coco/rsna/images/"),
            os.path.join(WDIR, "../../data/mappings/"),
            os.path.join(WDIR, "../../data/coco/rsna/annotations/"))

    COCOify(folds_df, 2, labels_df, 
            os.path.join(WDIR, "../../data/coco/rsna/images/"),
            os.path.join(WDIR, "../../data/mappings/"),
            os.path.join(WDIR, "../../data/coco/rsna/annotations/"))

    COCOify(folds_df, 3, labels_df, 
            os.path.join(WDIR, "../../data/coco/rsna/images/"),
            os.path.join(WDIR, "../../data/mappings/"),
            os.path.join(WDIR, "../../data/coco/rsna/annotations/"))

    COCOify(folds_df, 4, labels_df, 
            os.path.join(WDIR, "../../data/coco/rsna/images/"),
            os.path.join(WDIR, "../../data/mappings/"),
            os.path.join(WDIR, "../../data/coco/rsna/annotations/"))

    COCOify(folds_df, 5, labels_df, 
            os.path.join(WDIR, "../../data/coco/rsna/images/"),
            os.path.join(WDIR, "../../data/mappings/"),
            os.path.join(WDIR, "../../data/coco/rsna/annotations/"))

    COCOify(folds_df, 6, labels_df, 
            os.path.join(WDIR, "../../data/coco/rsna/images/"),
            os.path.join(WDIR, "../../data/mappings/"),
            os.path.join(WDIR, "../../data/coco/rsna/annotations/"))

    COCOify(folds_df, 7, labels_df, 
            os.path.join(WDIR, "../../data/coco/rsna/images/"),
            os.path.join(WDIR, "../../data/mappings/"),
            os.path.join(WDIR, "../../data/coco/rsna/annotations/"))

    COCOify(folds_df, 8, labels_df, 
            os.path.join(WDIR, "../../data/coco/rsna/images/"),
            os.path.join(WDIR, "../../data/mappings/"),
            os.path.join(WDIR, "../../data/coco/rsna/annotations/"))

    COCOify(folds_df, 9, labels_df, 
            os.path.join(WDIR, "../../data/coco/rsna/images/"),
            os.path.join(WDIR, "../../data/mappings/"),
            os.path.join(WDIR, "../../data/coco/rsna/annotations/"))

