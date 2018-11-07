###########
# IMPORTS #
###########

from keras_retinanet.models import load_model 
from sklearn.metrics import roc_auc_score 

from scipy.ndimage.interpolation import zoom 

import pandas as pd 
import numpy as np 
import scipy.misc 
import glob
import sys 
import os 
import re 

WDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(WDIR, "../../"))

#############
# FUNCTIONS #
#############

def preprocess_input(x, model):
    x = x.astype("float32")
    if re.search(r"inception|xception|mobilenet", model): 
        x /= 255.
        x -= 0.5
        x *= 2.
    elif re.search(r"densenet", model): 
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
    elif re.search(r"resnet|vgg", model):
        if x.shape[-1] == 3:
            x[..., 0] -= 103.939
            x[..., 1] -= 116.779
            x[..., 2] -= 123.680
        elif x.shape[-1] == 1: 
            x[..., 0] -= 115.799
    return x

def IoU(box1, box2):
    x11, y11, w1, h1 = box1
    x21, y21, w2, h2 = box2
    assert w1 * h1 > 0
    assert w2 * h2 > 0
    x12, y12 = x11 + w1, y11 + h1
    x22, y22 = x21 + w2, y21 + h2
    area1, area2 = w1 * h1, w2 * h2
    xi1, yi1, xi2, yi2 = max([x11, x21]), max([y11, y21]), min([x12, x22]), min([y12, y22])
    if xi2 <= xi1 or yi2 <= yi1:
        return 0
    else:
        intersect = (xi2-xi1) * (yi2-yi1)
        union = area1 + area2 - intersect
        return intersect / union

def mAP_IoU(boxes_true, boxes_pred, scores, thresholds = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]):
    """
    Mean average precision at different intersection over union (IoU) thresholds
    
    input:
        boxes_true: Mx4 numpy array of ground true bounding boxes of one image. 
                    bbox format: (x1, y1, w, h)
        boxes_pred: Nx4 numpy array of predicted bounding boxes of one image. 
                    bbox format: (x1, y1, w, h)
        scores:     length N numpy array of scores associated with predicted bboxes
        thresholds: IoU thresholds to evaluate mean average precision on
    output: 
        mAP: mean average precision of the image
    """
    # According to the introduction, images with no ground truth bboxes will not be 
    # included in the mAP score unless there is a false positive detection (?)
    # return None if both are empty, don't count the image in final evaluation (?)
    if len(boxes_true) == 0 and len(boxes_pred) == 0:
        return None
    assert boxes_true.shape[1] == 4 or boxes_pred.shape[1] == 4, "boxes should be 2D arrays with shape[1]=4"
    if len(boxes_pred):
        assert len(scores) == len(boxes_pred), "boxes_pred and scores should be same length"
        # sort boxes_pred by scores in decreasing order
        boxes_pred = boxes_pred[np.argsort(scores)[::-1], :]
    map_total = 0
    # loop over thresholds
    for t in thresholds:
        matched_bt = set()
        tp, fn = 0, 0
        for i, bt in enumerate(boxes_true):
            matched = False
            for j, bp in enumerate(boxes_pred):
                miou = IoU(bt, bp)
                if miou >= t and not matched and j not in matched_bt:
                    matched = True
                    tp += 1 # bt is matched for the first time, count as TP
                    matched_bt.add(j)
            if not matched:
                fn += 1 # bt has no match, count as FN               
        fp = len(boxes_pred) - len(matched_bt) # FP is the bp that not matched to any bt
        m = float(tp) / (tp + fn + fp)
        map_total += m  
    return map_total / float(len(thresholds))

##########
# SCRIPT #
##########

def KaggleEvaluate(model_save_path, 
                   results_save_path,
                   backbone_name, 
                   val_annotations,
                   stratified_folds,
                   fold,
                   image_input_size,
                   data_dir,
                   max_detections=5, 
                   score_threshold=np.linspace(0.05, 0.95, 19)):
    model = load_model(model_save_path,
                      backbone_name=backbone_name,
                      convert=True)
    val_annotations_df = pd.read_csv(val_annotations, header=None)
    val_annotations_df.columns = ["filepath", "x1", "y1", "x2", "y2", "class"]
    val_annotations_df["patientId"] = [_.split("/")[-1].split(".")[0] for _ in val_annotations_df.filepath]
    folds_df = pd.read_csv(stratified_folds) 
    folds_df = folds_df[folds_df.fold == fold] 
    folds_df = folds_df.merge(val_annotations_df, on="patientId", how="left") 
    folds_df["filepath"] = [os.path.join(data_dir, "{}.png".format(_)) for _ in folds_df.patientId]
    folds_df["class_y"]  = [1 if _ == "opacity" else 0 for _ in folds_df.class_y]
    # Lists for Kaggle mAP over all images
    list_of_metrics = []
    list_of_threshs = []
    list_of_imagids = [] 
    # Lists for Kaggle mAP over positive images
    list_of_metrics_positives_only = [] 
    list_of_threshs_positives_only = [] 
    list_of_imagids_positives_only = []  
    # Lists for evaluating RetinaNet as classifier
    list_of_scores = [] 
    list_of_labels = [] 
    list_of_views  = []
    list_of_pids   = [] 
    num_images = len(np.unique(folds_df.filepath))
    for index, each_img in enumerate(np.unique(folds_df.filepath)): 
        sys.stdout.write("Predicting: {}/{} ...\r".format(index+1, num_images))
        sys.stdout.flush() 
        test_id = each_img.split("/")[-1].split(".")[0] 
        # Read as BGR
        tmp_img = scipy.misc.imread(each_img, mode="RGB")  
        tmp_img = tmp_img[..., ::-1]
        # Image will always be square 
        scale = float(image_input_size) / tmp_img.shape[0]  
        if scale != 1.: 
            tmp_img = zoom(tmp_img, [scale, scale, 1.], order=1, prefilter=False)
        tmp_img = preprocess_input(tmp_img, backbone_name) 
        prediction = model.predict_on_batch(np.expand_dims(tmp_img, axis=0))
        # Get ground truth for image
        tmp_df = folds_df[folds_df.filepath == each_img]
        if tmp_df.class_y.iloc[0] == 0: 
            gt_bboxes = np.empty((0, 4))
        else:
            gt_bboxes = [np.asarray((row.x1, row.y1, row.x2-row.x1, row.y2-row.y1)) for rownum, row in tmp_df.iterrows()]
            gt_bboxes = np.asarray(gt_bboxes) 
        # Save values for classifier evaluation 
        list_of_scores.append(np.max(prediction[1][0]))
        list_of_labels.append(tmp_df.class_y.iloc[0])
        list_of_views.append(tmp_df.view.iloc[0])
        list_of_pids.append(each_img.split("/")[-1].split(".")[0])
        for each_thres in score_threshold:
            scores = prediction[1][0] 
            bboxes = prediction[0][0] 
            # Ensure that scores are sorted in descending order
            sorted_indices = np.argsort(scores)[::-1]
            scores = np.asarray(scores[sorted_indices])
            bboxes = np.asarray(bboxes[sorted_indices]) 
            # Get boxes greater than threshold
            detected = scores >= each_thres  
            # Limit number of boxes to max_detections
            if np.sum(detected) > max_detections:
                detected[max_detections:] = False 
            scores = np.asarray(scores[detected])
            bboxes = np.asarray(bboxes[detected])
            list_of_bboxes = []
            # Rescale boxes 
            bboxes = [box / scale for box in bboxes] 
            for each_box in bboxes: 
                x1 = each_box[0] 
                y1 = each_box[1] 
                ww = each_box[2] - each_box[0] 
                hh = each_box[3] - each_box[1] 
                list_of_bboxes.append((np.asarray((x1, y1, ww, hh))))
            box_array = np.asarray(list_of_bboxes) 
            # Calculate metric 
            mapiou = mAP_IoU(gt_bboxes, box_array, scores)
            # Positives only 
            if len(gt_bboxes) != 0:
                list_of_metrics_positives_only.append(mapiou) 
                list_of_threshs_positives_only.append(each_thres) 
                list_of_imagids_positives_only.append(each_img.split("/")[-1].split(".")[0])
            # All images 
            list_of_metrics.append(mapiou) 
            list_of_threshs.append(each_thres) 
            list_of_imagids.append(each_img.split("/")[-1].split(".")[0])
    results_df = pd.DataFrame({"patientId": list_of_imagids,
                               "mAP":       list_of_metrics,
                               "threshold": list_of_threshs})
    results_df[["mAP", "threshold"]].groupby("threshold").mean().reset_index().to_csv(results_save_path, index=False) 
    results_df_positives_only = pd.DataFrame({"patientId": list_of_imagids_positives_only,
                                              "mAP":       list_of_metrics_positives_only,
                                              "threshold": list_of_threshs_positives_only})
    results_df_positives_only[["mAP", "threshold"]].groupby("threshold").mean().reset_index().to_csv("{}-positives-only.csv".format(results_save_path.replace(".csv", "")), index=False) 
    #
    scores_df = pd.DataFrame({"patientId": list_of_pids,
                              "y_score":   list_of_scores,
                              "y_true":    list_of_labels,
                              "view":      list_of_views})  
    auroc = roc_auc_score(scores_df.y_true, scores_df.y_score) 
    auroc_AP = roc_auc_score(scores_df.y_true[scores_df.view == "AP"], 
                             scores_df.y_score[scores_df.view == "AP"])
    auroc_PA = roc_auc_score(scores_df.y_true[scores_df.view == "PA"],
                             scores_df.y_score[scores_df.view == "PA"])
    scores_df.to_csv("{}-scores-auc{}-AP{}-PA{}.csv".format(results_save_path.replace(".csv", ""), 
                                                            round(auroc, 3),
                                                            round(auroc_AP, 3),
                                                            round(auroc_PA, 3)))
    return np.max(results_df.mAP), np.max(results_df_positives_only.mAP), auroc

