import pandas as pd 
import numpy as np
import os, json 

WDIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(WDIR, "../../SETTINGS.json")) as f: 
    SETTINGS_JSON = json.load(f) 

CONCAT_IMAGES_DIR = os.path.join(WDIR, "../..", SETTINGS_JSON["TRAIN_IMAGES_CLEAN_DIR"], "concat") 

labels_df = pd.read_csv(os.path.join(WDIR, "../..", SETTINGS_JSON["TRAIN_INFO_DIR"], "concat_train_labels.csv"))
save_annotations_dir = os.path.join(WDIR, "../../models/RetinaNet/annotations/cloud/")

def generate_annotations(fold, labels_df=labels_df, concat_images_dir=CONCAT_IMAGES_DIR, save_annotations_dir=save_annotations_dir):
    df = labels_df[labels_df != fold] 
    df["x2"] = df.x + df.width ; df["x2"] = df.x2.astype("int32") 
    df["y2"] = df.y + df.height ; df["y2"] = df.y2.astype("int32") 
    df["x"] = df.x.astype("int32")  
    df["y"] = df.y.astype("int32")
    df["class"] = "opacity"
    df["filepath"] = [os.path.join(concat_images_dir, "{}.png".format(row.patientId)) for rowNum, row in df.iterrows()]
    df = df[["filepath", "x", "y", "x2", "y2", "class"]]
    if not os.path.exists(save_annotations_dir): os.makedirs(save_annotations_dir)
    df.to_csv(os.path.join(save_annotations_dir, "fold{}_train_concat_annotations.csv".format(fold)), index=False, header=False)

generate_annotations(0)
generate_annotations(1)
generate_annotations(2)
generate_annotations(3)
generate_annotations(4)
generate_annotations(5)
generate_annotations(6)
generate_annotations(7)
generate_annotations(8)
generate_annotations(9)




