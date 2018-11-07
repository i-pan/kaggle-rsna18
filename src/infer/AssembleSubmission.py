import os, json 

WDIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(WDIR, "../../SETTINGS.json")) as f: 
    SETTINGS_JSON = json.load(f) 

CLASSIFIER_SCORES_PATH = os.path.join(WDIR, "../../ClassifierEnsembleScores.csv")
METADATA_PATH          = os.path.join(WDIR, "../../", SETTINGS_JSON["TEST_INFO_DIR"], "test_metadata.csv") 
DCN_ENSEMBLE_PRED_PATH = os.path.join(WDIR, "../../DCNEnsemblePredictions.csv") 
CONCATRETINA_PRED_PATH = os.path.join(WDIR, "../../ConcatRetinaEnsemblePredictions.csv")
NORMALRETINA_PRED_PATH = os.path.join(WDIR, "../../RetinaEnsemblePredictions.csv") 

def reduce_bbox_size(bbox, width_frac=0.875, height_frac=0.875): 
    """
    bbox (list): 
    -- [x, y, w, h] 
    frac_of_original (float): 
    -- new box area will be frac_of_original times original area
    """
    x, y, w, h = bbox
    bbox_center = (x + w / 2., y + h / 2.) 
    new_w = width_frac * w
    new_h = height_frac * h
    new_x = bbox_center[0] - new_w / 2.
    new_y = bbox_center[1] - new_h / 2. 
    resized_bbox = [new_x, new_y, new_w, new_h]
    resized_bbox = [int(round(_)) for _ in resized_bbox]
    return resized_bbox

import pandas as pd 
import numpy as np 
import scipy.stats 

classify = pd.read_csv(CLASSIFIER_SCORES_PATH)
metadata = pd.read_csv(METADATA_PATH) 

classify = classify.merge(metadata, on="patientId") 
all_pids = classify.patientId 

###################
# DCN PREDICTIONS #
###################
detect_dcn = pd.read_csv(DCN_ENSEMBLE_PRED_PATH) 
detect_dcn = detect_dcn.merge(classify[["patientId","ensembleScore"]], on="patientId")
detect_dcn["adjustedScore"] = detect_dcn.adjustedScore * detect_dcn.ensembleScore
box_score_threshold = 0.225
predict_box_dcn = detect_dcn[detect_dcn.adjustedScore >= box_score_threshold]
# len(np.unique(predict_box_dcn.patientId))
# print (predict_box_dcn.patientId.value_counts().head(n=10))

resized_bboxes = []
for rownum, row in predict_box_dcn.iterrows():
    new_bbox = reduce_bbox_size([row.x, row.y, row.w, row.h], 0.875, 0.875) 
    resized_bboxes.append(new_bbox) 

predict_box_dcn["x"] = [box[0] for box in resized_bboxes]
predict_box_dcn["y"] = [box[1] for box in resized_bboxes]
predict_box_dcn["w"] = [box[2] for box in resized_bboxes]
predict_box_dcn["h"] = [box[3] for box in resized_bboxes]


######################
# RETINA PREDICTIONS #
######################
detect_ret0 = pd.read_csv(CONCATRETINA_PRED_PATH)
detect_ret0 = detect_ret0.merge(classify, on="patientId")
detect_ret0 = detect_ret0[detect_ret0.ensembleScore >= 0.2] 
box_score_threshold = 0.3
predict_box_ret0 = detect_ret0[detect_ret0.adjustedScore >= box_score_threshold]
# len(np.unique(predict_box_ret0.patientId))
# print(predict_box_ret0.patientId.value_counts().head(n=10))

resized_bboxes = []
for rownum, row in predict_box_ret0.iterrows():
    new_bbox = reduce_bbox_size([row.x, row.y, row.w, row.h], 0.875, 0.875) 
    resized_bboxes.append(new_bbox) 

predict_box_ret0["x"] = [box[0] for box in resized_bboxes]
predict_box_ret0["y"] = [box[1] for box in resized_bboxes]
predict_box_ret0["w"] = [box[2] for box in resized_bboxes]
predict_box_ret0["h"] = [box[3] for box in resized_bboxes]

detect_ret1  = pd.read_csv(NORMALRETINA_PRED_PATH) 
detect_ret1 = detect_ret1.merge(classify, on="patientId")
detect_ret1 = detect_ret1[detect_ret1.ensembleScore >= 0.325] 
box_score_threshold = 0.35
predict_box_ret1 = detect_ret1[detect_ret1.adjustedScore >= box_score_threshold]
# len(np.unique(predict_box_ret1.patientId))
# print(predict_box_ret1.patientId.value_counts().head(n=10))

resized_bboxes = []
for rownum, row in predict_box_ret1.iterrows():
    new_bbox = reduce_bbox_size([row.x, row.y, row.w, row.h], 0.875, 0.875) 
    resized_bboxes.append(new_bbox) 

predict_box_ret1["x"] = [box[0] for box in resized_bboxes]
predict_box_ret1["y"] = [box[1] for box in resized_bboxes]
predict_box_ret1["w"] = [box[2] for box in resized_bboxes]
predict_box_ret1["h"] = [box[3] for box in resized_bboxes]

# Ensemble the two RetinaNets
execfile(os.path.join(WDIR, "DetectionEnsemble.py"))

list_of_dfs = [predict_box_ret0, predict_box_ret1]

list_of_pids = []
list_of_ensemble_bboxes = [] 
unique_pids = list(predict_box_ret0.patientId)
unique_pids.extend(list(predict_box_ret1.patientId))
unique_pids = np.unique(unique_pids)
for pid in unique_pids:
    list_of_tmp_dfs = []
    list_of_detections = [] 
    for each_df in list_of_dfs: 
        tmp_df = each_df[each_df.patientId == pid]
        list_of_bboxes = []
        for rownum, row in tmp_df.iterrows(): 
            bbox = list(row[["x", "y", "w", "h"]])
            bbox.append(1) 
            bbox.append(row.adjustedScore) 
            list_of_bboxes.append(bbox) 
        list_of_detections.append(list_of_bboxes) 
    list_of_ensemble_bboxes.append(GeneralEnsemble(list_of_detections, iou_thresh=0.4, weights=[1.2,0.8]))
    list_of_pids.append(pid) 

list_of_new_pids = []
list_of_bboxes = [] 
for i, ensemble_bboxes in enumerate(list_of_ensemble_bboxes): 
    for bbox in ensemble_bboxes: 
        list_of_new_pids.append(list_of_pids[i]) 
        list_of_bboxes.append(bbox) 

predict_box_ret = pd.DataFrame({"patientId": list_of_new_pids,
                            "x": [box[0] for box in list_of_bboxes],
                            "y": [box[1] for box in list_of_bboxes],
                            "w": [box[2] for box in list_of_bboxes], 
                            "h": [box[3] for box in list_of_bboxes],
                            "score": [box[5] for box in list_of_bboxes],
                            "votes": [box[-1]*len(list_of_dfs) for box in list_of_bboxes]})
predict_box_ret = predict_box_ret.merge(metadata[["patientId", "view"]])
predict_box_ret = predict_box_ret[["patientId", "x", "y", "w", "h", "score", "votes", "view"]]
predict_box_ret["doubleScore"] = predict_box_ret.score * (predict_box_ret.votes / len(list_of_dfs))
predict_box_ret = predict_box_ret[predict_box_ret.doubleScore >= 0.2]
predict_box_ret["adjustedScore"] = predict_box_ret.doubleScore

########################
# COMBINE DCN + RETINA #
########################
list_of_dfs = [predict_box_dcn, predict_box_ret]

list_of_pids = []
list_of_ensemble_bboxes = [] 
unique_pids = list(predict_box_dcn.patientId)
unique_pids.extend(list(predict_box_ret.patientId))
unique_pids = np.unique(unique_pids)
for pid in unique_pids:
    list_of_tmp_dfs = []
    list_of_detections = [] 
    for each_df in list_of_dfs: 
        tmp_df = each_df[each_df.patientId == pid]
        list_of_bboxes = []
        for rownum, row in tmp_df.iterrows(): 
            bbox = list(row[["x", "y", "w", "h"]])
            bbox.append(1) 
            bbox.append(row.adjustedScore) 
            list_of_bboxes.append(bbox) 
        list_of_detections.append(list_of_bboxes) 
    list_of_ensemble_bboxes.append(GeneralEnsemble(list_of_detections, iou_thresh=0.4, weights=[1.,1.,]))
    list_of_pids.append(pid) 

list_of_new_pids = []
list_of_bboxes = [] 
for i, ensemble_bboxes in enumerate(list_of_ensemble_bboxes): 
    for bbox in ensemble_bboxes: 
        list_of_new_pids.append(list_of_pids[i]) 
        list_of_bboxes.append(bbox) 

predict_box = pd.DataFrame({"patientId": list_of_new_pids,
                            "x": [box[0] for box in list_of_bboxes],
                            "y": [box[1] for box in list_of_bboxes],
                            "w": [box[2] for box in list_of_bboxes], 
                            "h": [box[3] for box in list_of_bboxes],
                            "score": [box[5] for box in list_of_bboxes],
                            "votes": [box[-1]*len(list_of_dfs) for box in list_of_bboxes]})
predict_box = predict_box.merge(metadata[["patientId", "view"]])
predict_box = predict_box[["patientId", "x", "y", "w", "h", "score", "votes", "view"]]
predict_box["doubleScore"] = predict_box.score * (predict_box.votes / len(list_of_dfs))
for thres in np.linspace(0.05, 0.95, 37): 
    print "{0:.3f} : {1}p // {2}b".format(thres, 
        len(np.unique(predict_box.patientId[predict_box.doubleScore >= thres])),
            np.sum(predict_box.doubleScore >= thres))

predict_box = predict_box[predict_box.doubleScore >= 0.15]

list_of_pids = [] 
list_of_preds = [] 
for pid in np.unique(predict_box.patientId): 
    tmp_df = predict_box[predict_box.patientId == pid] 
    predictionString = " ".join(["{} {} {} {} {}".format(row.doubleScore, row.x, row.y, row.w, row.h) for rownum, row in tmp_df.iterrows()])
    list_of_preds.append(predictionString)
    list_of_pids.append(pid) 

positives = pd.DataFrame({"patientId": list_of_pids, 
                          "PredictionString": list_of_preds}) 

negatives = pd.DataFrame({"patientId": list(set(all_pids) - set(list_of_pids)), 
                          "PredictionString": [""] * (len(all_pids)-len(list_of_pids))})

submission = positives.append(negatives) 
submission.to_csv(os.path.join(WDIR, "../../", SETTINGS_JSON["SUBMISSION_DIR"], "FinalSubmission.csv"), index=False)


