import os, json

WDIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(WDIR, "../../SETTINGS.json")) as f: 
    SETTINGS_JSON = json.load(f) 

STRATIFIED_FOLDS_DF_PATH = os.path.join(WDIR, "../..", SETTINGS_JSON["TRAIN_INFO_DIR"], "stage_1_stratified_folds_df.csv")

def train(fold, backbone, 
          gpu=0, 
          stratified_folds=STRATIFIED_FOLDS_DF_PATH, 
          batch_size=1, 
          data_dir=os.path.join(WDIR, "../../", SETTINGS_JSON["TRAIN_IMAGES_CLEAN_DIR"], "orig"), 
          steps=8000,
          epochs=8,
          snapshot_path=os.path.join(WDIR, "../../models/RetinaNet/output/skippy/fold{}_384"),
          image_min_side=384,
          image_max_side=384,
          annotations_folder=os.path.join(WDIR, "../../models/RetinaNet/annotations/skippy/fold{}_train_100_0_0_annotations.csv"), 
          classes_file_path=os.path.join(WDIR, "../../models/RetinaNet/classes.csv"), 
          val_annotations=os.path.join(WDIR, "../../models/RetinaNet/annotations/skippy/fold{}_pos_valid_annotations.csv")): 
  TRAIN_KAGGLE_PATH = os.path.join(WDIR, "../../models/RetinaNet/keras_retinanet/bin/train_kaggle.py")
  snapshot_path = snapshot_path.format(fold)
  annotations_folder = annotations_folder.format(fold) 
  val_annotations = val_annotations.format(fold) 
  command  = "python {} --backbone {} --batch-size {}".format(TRAIN_KAGGLE_PATH, backbone, batch_size) 
  command += " --gpu {} --stratified_folds {} --fold {} --data_dir {}".format(gpu, stratified_folds, fold, data_dir) 
  command += " --steps {} --epochs {} --snapshot-path {}".format(steps, epochs, snapshot_path)
  command += " --image_min_side {} --image_max_side {} csv {}".format(image_min_side, image_max_side, annotations_folder)
  command += " {} --val-annotations {}".format(classes_file_path, val_annotations)
  os.system(command) 


train(0, "resnet101") 
train(1, "resnet152") 
train(2, "resnet101") 
train(3, "resnet152") 
train(4, "resnet101") 
train(5, "resnet152") 
train(6, "resnet101") 
train(7, "resnet152") 
train(8, "resnet101") 
train(9, "resnet152") 


