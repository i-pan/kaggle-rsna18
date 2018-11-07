import os, json

WDIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(WDIR, "../../SETTINGS.json")) as f:
    SETTINGS_JSON = json.load(f)

CLASSIFIER_SCORES_PATH = os.path.join(WDIR, "../../OneClassifierScores.csv")
METADATA_PATH          = os.path.join(WDIR, "../../", SETTINGS_JSON["TEST_INFO_DIR"], "test_metadata.csv")
DCN_ENSEMBLE_PRED_PATH = os.path.join(WDIR, "../../SimpleDCNPredictions.csv")
CONCATRETINA_PRED_PATH = os.path.join(WDIR, "../../ConcatRetinaEnsemblePredictions.csv")
NORMALRETINA_PRED_PATH = os.path.join(WDIR, "../../RetinaEnsemblePredictions.csv")

def reduce_bbox_size(bbox, width_frac=1., height_frac=1.): 
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
del classify["ensembleScore"]
classify["ensembleScore"] = classify.InceptionResNetV2_Fold0Px256_e005
detect_dcn = pd.read_csv(DCN_ENSEMBLE_PRED_PATH) 
detect_dcn = detect_dcn.merge(classify[["patientId","ensembleScore"]], on="patientId")
detect_dcn["adjustedScore"] = detect_dcn.adjustedScore * detect_dcn.ensembleScore
box_score_threshold = 0.2
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


predict_box = predict_box_dcn
predict_box = predict_box.merge(metadata[["patientId", "view"]])
predict_box["doubleScore"] = predict_box.adjustedScore
for thres in np.linspace(0.05, 0.95, 37): 
    print "{0:.3f} : {1}p // {2}b".format(thres, 
        len(np.unique(predict_box.patientId[predict_box.doubleScore >= thres])),
            np.sum(predict_box.doubleScore >= thres))

predict_box = predict_box[predict_box.doubleScore >= 0.275]

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
submission.to_csv(os.path.join(WDIR, "../..", SETTINGS_JSON["SUBMISSION_DIR"], "SimpleSubmission.csv"), index=False) 


