###########
# IMPORTS #
###########

import pandas as pd 
import numpy as np 
import os, json 

WDIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(WDIR, "../../SETTINGS.json")) as f: 
    SETTINGS_JSON = json.load(f) 

from sklearn.model_selection import StratifiedKFold

#############
# FUNCTIONS #
#############

def assign_folds(orig_df, num_folds, val_frac=0.10, seed=88):
    # Stratified splits
    np.random.seed(seed) 
    df = orig_df.copy() 
    df["fold"] = None  
    skf = StratifiedKFold(n_splits=num_folds, random_state=0, shuffle=True) 
    fold_counter = 0 
    for train_index, test_index in skf.split(df.patientId, df.combined_cat):
        df["fold"].iloc[test_index] = fold_counter
        fold_counter += 1 
    # for each_fold in np.unique(df.fold): 
    #     train_df = df[df.fold != each_fold] 
    #     val_counter = 0
    #     train_df["val{}".format(each_fold)] = None 
    #     for train_index, test_index in skf.split(train_df.patientId, train_df.combined_cat): 
    #         train_df["val{}".format(each_fold)].iloc[test_index] = val_counter
    #         val_counter += 1
    #     df = df.merge(train_df[["patientId", "val{}".format(each_fold)]], on="patientId", how="left")
    return df

##########
# SCRIPT #
##########

path_to_labels_file = os.path.join(WDIR, "../../", SETTINGS_JSON["TRAIN_INFO_DIR"], "detailed_class_info.csv")
labels_df = pd.read_csv(path_to_labels_file).drop_duplicates()

# Get metadata for training set
path_to_metadata = os.path.join(WDIR, "../../", SETTINGS_JSON["TRAIN_INFO_DIR"], "train_metadata.csv")
metadata = pd.read_csv(path_to_metadata)

# Divide age into categories based on decade
metadata["age_category"] = np.ceil(metadata.age / 10.)
metadata["age_category"][metadata.age_category > 9] = 9.

df = labels_df.merge(metadata, on="patientId") 
df["combined_cat"] = ["{}_{}_{}_{}".format(row["class"], row.age_category, row.sex, row["view"]) for rownum, row in df.iterrows()]
folds_df = assign_folds(df, 10)

folds_df.to_csv(os.path.join(WDIR, "../../", SETTINGS_JSON["TRAIN_INFO_DIR"], "stratified_folds_df.csv"))


