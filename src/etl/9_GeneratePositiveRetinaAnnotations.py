###########
# IMPORTS #
###########
import pandas as pd 
import numpy as np 
import os, json 

WDIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(WDIR, "../../SETTINGS.json")) as f: 
    SETTINGS_JSON = json.load(f) 

#############
# FUNCTIONS #
#############

def generate_annotations(folds_df, 
                         data_dir, 
                         save_annotations_dir, 
                         percent_no_opacity, 
                         percent_normal, 
                         fold=None):
    # Generate annotations for keras-retinanet for each fold
    if fold is not None:
        folds = fold
        if type(folds) != list: folds = [folds]
    else: 
        folds = np.unique(folds_df.fold)
    for each_fold in folds: 
        tmp_train_df = folds_df[folds_df.fold != each_fold] 
        tmp_valid_df = folds_df[folds_df.fold == each_fold] 
        tmp_train_df["filepath"] = [os.path.join(data_dir, "{}.png".format(pid)) for pid in tmp_train_df.patientId]
        tmp_valid_df["filepath"] = [os.path.join(data_dir, "{}.png".format(pid)) for pid in tmp_valid_df.patientId]
        tmp_train_df["Target"] = ["opacity" if _ == 1 else None for _ in tmp_train_df.Target] 
        tmp_train_df_pos = tmp_train_df[tmp_train_df.Target == "opacity"] 
        pos_frac = 1. - percent_no_opacity - percent_normal
        num_unique_pos = len(np.unique(tmp_train_df_pos.patientId))
        # No Lung Opacity / Not Normal
        tmp_train_df_neg0 = tmp_train_df[tmp_train_df["class"] == "No Lung Opacity / Not Normal"] 
        tmp_train_df_neg0 = tmp_train_df_neg0.sample(n=int(percent_no_opacity*num_unique_pos/pos_frac))
        # Normal
        tmp_train_df_neg1 = tmp_train_df[tmp_train_df["class"] == "Normal"] 
        tmp_train_df_neg1 = tmp_train_df_neg1.sample(n=int(percent_normal*num_unique_pos/pos_frac))
        tmp_train_df_neg = tmp_train_df_neg0.append(tmp_train_df_neg1)
        tmp_train_df = tmp_train_df_pos.append(tmp_train_df_neg) 
        tmp_train_df = tmp_train_df.sample(frac=1)
        tmp_train_df = tmp_train_df[["filepath", "x1", "y1", "x2", "y2", "Target"]]
        tmp_valid_df = tmp_valid_df[["filepath", "x1", "y1", "x2", "y2", "Target"]]
        # Leave validation as positives only
        # Evaluation script will use the full validation set 
        tmp_valid_df["Target"] = ["opacity" if _ == 1 else None for _ in tmp_valid_df.Target] 
        tmp_train_df.to_csv(os.path.join(save_annotations_dir, 
                                         "fold{}_train_{}_{}_{}_annotations.csv".format(each_fold, 
                                                                                        int(pos_frac*100), 
                                                                                        int(percent_no_opacity*100), 
                                                                                        int(percent_normal*100))), 
                            header=False, index=False)
        tmp_valid_df.to_csv(os.path.join(save_annotations_dir, "fold{}_pos_valid_annotations.csv".format(each_fold)), 
                            header=False, index=False)


##########
# SCRIPT #
##########

# Add bbox labels to folds
labels_df = pd.read_csv(os.path.join(WDIR, "../..", SETTINGS_JSON["TRAIN_INFO_DIR"], "train_labels.csv"))
folds_df  = pd.read_csv(os.path.join(WDIR, "../..", SETTINGS_JSON["TRAIN_INFO_DIR"], "stratified_folds_df.csv"))
folds_df  = folds_df.merge(labels_df, on="patientId")

folds_df["x1"] = folds_df["x"] 
folds_df["y1"] = folds_df["y"]
folds_df["x2"] = folds_df["x"] + folds_df["width"]
folds_df["y2"] = folds_df["y"] + folds_df["height"]

folds_df = folds_df.fillna(8888888) 

folds_df["x1"] = folds_df.x1.astype("int32").astype("str") 
folds_df["y1"] = folds_df.y1.astype("int32").astype("str") 
folds_df["x2"] = folds_df.x2.astype("int32").astype("str") 
folds_df["y2"] = folds_df.y2.astype("int32").astype("str") 

folds_df = folds_df.replace({"8888888":None}) 

del folds_df["x"], folds_df["y"] 

data_dir = os.path.join(WDIR, "../../", SETTINGS_JSON["TRAIN_IMAGES_CLEAN_DIR"], "orig")  
save_annotations_dir = os.path.join(WDIR, "../../models/RetinaNet/annotations/skippy/")

if not os.path.exists(save_annotations_dir): os.makedirs(save_annotations_dir)

generate_annotations(folds_df, data_dir, save_annotations_dir, 0, 0, None)

os.system("cp {}/*valid* {}/../cloud/".format(save_annotations_dir, save_annotations_dir))

