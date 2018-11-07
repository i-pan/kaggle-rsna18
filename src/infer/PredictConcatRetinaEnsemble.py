# Specify GPU 
import os, json 

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

WDIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(WDIR, "../../SETTINGS.json")) as f: 
    SETTINGS_JSON = json.load(f) 

TEST_IMAGES_DIR = os.path.join(WDIR, "../../", SETTINGS_JSON["TEST_IMAGES_CLEAN_DIR"], "orig")
MODELS_DIR      = os.path.join(WDIR, "../../models/RetinaNet/output/cloud/")

import sys
sys.path.append(os.path.join(WDIR, "../../models/RetinaNet/"))
from keras_retinanet.models import load_model 
from scipy.ndimage.interpolation import zoom 

import numpy as np 
import scipy.misc 
import glob
import os 
import re

# Specify GPU (if needed)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Get functions for ensembling object detections
execfile(os.path.join(WDIR, "DetectionEnsemble.py"))

MODEL0_PATH = os.path.join(MODELS_DIR, "fold0_384_resnet101_csv_06.h5")
MODEL1_PATH = os.path.join(MODELS_DIR, "fold1_384_resnet152_csv_03.h5")
MODEL2_PATH = os.path.join(MODELS_DIR, "fold2_384_resnet101_csv_05.h5")
MODEL3_PATH = os.path.join(MODELS_DIR, "fold3_384_resnet152_csv_02.h5")
MODEL4_PATH = os.path.join(MODELS_DIR, "fold4_384_resnet101_csv_04.h5")
MODEL5_PATH = os.path.join(MODELS_DIR, "fold5_384_resnet152_csv_07.h5")
MODEL6_PATH = os.path.join(MODELS_DIR, "fold6_384_resnet101_csv_06.h5")
MODEL7_PATH = os.path.join(MODELS_DIR, "fold7_384_resnet152_csv_04.h5")
MODEL8_PATH = os.path.join(MODELS_DIR, "fold8_384_resnet101_csv_04.h5")
MODEL9_PATH = os.path.join(MODELS_DIR, "fold9_384_resnet152_csv_04.h5")

model0 = load_model(MODEL0_PATH, backbone_name="resnet101", convert=True)
model1 = load_model(MODEL1_PATH, backbone_name="resnet152", convert=True)
model2 = load_model(MODEL2_PATH, backbone_name="resnet101", convert=True)
model3 = load_model(MODEL3_PATH, backbone_name="resnet152", convert=True)
model4 = load_model(MODEL4_PATH, backbone_name="resnet101", convert=True)
model5 = load_model(MODEL5_PATH, backbone_name="resnet152", convert=True)
model6 = load_model(MODEL6_PATH, backbone_name="resnet101", convert=True)
model7 = load_model(MODEL7_PATH, backbone_name="resnet152", convert=True)
model8 = load_model(MODEL8_PATH, backbone_name="resnet101", convert=True)
model9 = load_model(MODEL9_PATH, backbone_name="resnet152", convert=True)

model_list = [model0, model1, model2, model3, model4, model5, model6, model7, model8, model9]

def preprocess_input(x, model):
    x = x.astype("float32")
    if re.search("inception|xception|mobilenet", model): 
        x /= 255.
        x -= 0.5
        x *= 2.
    elif re.search("densenet", model): 
        x /= 255.
        if x.shape[-1] == 3:
            x[..., 0] -= 0.485
            x[..., 1] -= 0.456
            x[..., 2] -= 0.406 
            x[..., 0] /= 0.229 
            x[..., 1] /= 0.224
            x[..., 2] /= 0.225 
        elif x.shape[-1] == 1: 
            x[..., 0] -= 0.449
            x[..., 0] /= 0.226
    elif re.search("resnet|vgg", model):
        if x.shape[-1] == 3:
            x[..., 0] -= 103.939
            x[..., 1] -= 116.779
            x[..., 2] -= 123.680
        elif x.shape[-1] == 1: 
            x[..., 0] -= 115.799
    return x

def flip_box(box):
    """
    box (list, length 4): [x1, y1, x2, y2]
    """
    # Get top right corner of prediction
    w = box[2] - box[0] 
    h = box[3] - box[1]
    topRight = (box[2], box[1])
    # Top left corner of flipped box is:
    newTopLeft = (1024. - topRight[0], topRight[1])
    return [newTopLeft[0], newTopLeft[1], newTopLeft[0]+w, newTopLeft[1]+h]

test_images = glob.glob(os.path.join(TEST_IMAGES_DIR, "*"))

TTA_sizes = [320, 352, 384, 416, 448]
test_predictions = [] 
test_pids = [] 
for imgIndex, imgfile in enumerate(test_images): 
    sys.stdout.write("{}/{} ...\r".format(imgIndex+1, len(test_images)))
    sys.stdout.flush() 
    img = scipy.misc.imread(imgfile, mode="RGB") 
    individual_preds = [] 
    for index, model in enumerate(model_list):
        TTA_predictions = [] 
        for imsize in TTA_sizes: 
            scale =  float(imsize) / img.shape[0]
            resized_img = zoom(img.copy(), [scale, scale, 1.], prefilter=False, order=1)
            resized_img = preprocess_input(resized_img, "resnet")
            prediction0 = model.predict_on_batch(np.expand_dims(resized_img, axis=0))
            # Flip image
            prediction1 = model.predict_on_batch(np.expand_dims(np.fliplr(resized_img), axis=0))
            # Prepare predictions for DetectionEnsemble 
            bboxes0 = prediction0[0][0][:10] / scale 
            bboxes1 = prediction1[0][0][:10] / scale 
            bboxes1 = [flip_box(_) for _ in bboxes1]
            scores0 = prediction0[1][0][:10] ; scores0 = np.clip(scores0, 0, 1)
            scores1 = prediction1[1][0][:10] ; scores1 = np.clip(scores1, 0, 1)
            # Convert boxes to [x, y, w, h] 
            bboxes0 = [[box[0], box[1], box[2]-box[0], box[3]-box[1]] for box in bboxes0]
            bboxes1 = [[box[0], box[1], box[2]-box[0], box[3]-box[1]] for box in bboxes1]            
            final_pred0 = [list(_) + [1, scores0[i]] for i, _ in enumerate(bboxes0)]
            final_pred1 = [list(_) + [1, scores1[i]] for i, _ in enumerate(bboxes1)]
            TTA_predictions.extend([final_pred0, final_pred1]) 
        # Ensemble predictions 
        individual_preds.append(GeneralEnsemble(TTA_predictions, iou_thresh=0.4))
    test_predictions.append(individual_preds) 
    test_pids.append(imgfile.split("/")[-1].split(".")[0])

import pandas as pd 

# Parse the predictions
det_df = pd.DataFrame() 
for patientIndex, eachDet in enumerate(test_predictions): 
    for modelIndex, eachModelPred in enumerate(eachDet): 
        tmp_df = pd.DataFrame(np.asarray(eachModelPred)) 
        tmp_df.columns = ["x", "y", "w", "h", "TYPE", "rawScore", "votes"]
        tmp_df["pid"] = test_pids[patientIndex]
        tmp_df["modelId"] = modelIndex
        det_df = det_df.append(tmp_df)

det_df["adjScore"] = det_df.votes * det_df.rawScore

# Get list of lists of detections
detections_list = [] 
for each_pid in test_pids: 
    tmp_df = det_df[det_df.pid == each_pid] 
    individual_pid_dets = []
    for each_model in range(10): 
        individual_model_dets = [] 
        tmp_model_df = tmp_df[tmp_df.modelId == each_model] 
        for rowNum, row in tmp_model_df.iterrows(): 
            individual_model_dets.append([row.x, row.y, row.w, row.h, row.TYPE, row.adjScore])
        individual_pid_dets.append(individual_model_dets) 
    detections_list.append(individual_pid_dets)

# Apply ensemble

ensemble_dets = [] 
for each_det in detections_list: 
    ensemble_dets.append(GeneralEnsemble(each_det))


# Assemble DataFrame
import pandas as pd 
df = pd.DataFrame() 
for index, each_det in enumerate(ensemble_dets): 
    tmp_df = pd.DataFrame({"patientId": test_pids[index],
                           "x": [box[0] for box in each_det],
                           "y": [box[1] for box in each_det],
                           "w": [box[2] for box in each_det], 
                           "h": [box[3] for box in each_det], 
                           "score": [box[5] for box in each_det], 
                           "votes": [box[6] for box in each_det]})
    df = df.append(tmp_df) 

df["adjustedScore"] = df.score * df.votes 
df["adjustedScore"][df.adjustedScore < 0] = 0

df.to_csv(os.path.join(WDIR, "../../ConcatRetinaEnsemblePredictions.csv"), index=False)


