###########
# IMPORTS #
###########

import pandas as pd 
import numpy as np 
import os, json 

WDIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(WDIR, "../../SETTINGS.json")) as f: 
    SETTINGS_JSON = json.load(f) 


with open(os.path.join(WDIR, "stage_1_patientIds.txt")) as f: 
    stage_1_patientIds = f.readlines()
    stage_1_patientIds = [line.strip() for line in stage_1_patientIds]

path_to_labels_file = os.path.join(WDIR, "../../", SETTINGS_JSON["TRAIN_INFO_DIR"], "detailed_class_info.csv")
labels_df = pd.read_csv(path_to_labels_file) 
labels_df = labels_df[labels_df.patientId.isin(stage_1_patientIds)]
labels_df.to_csv(path_to_labels_file, index=False) 

path_to_boxes_file = os.path.join(WDIR, "../../", SETTINGS_JSON["TRAIN_INFO_DIR"], "train_labels.csv")
boxes_df = pd.read_csv(path_to_boxes_file) 
boxes_df = boxes_df[boxes_df.patientId.isin(stage_1_patientIds)]
boxes_df.to_csv(path_to_boxes_file, index=False) 
