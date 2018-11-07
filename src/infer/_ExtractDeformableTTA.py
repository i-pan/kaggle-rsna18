import pandas as pd 
import numpy as np 
import json 
import os 

WDIR = os.path.dirname(os.path.abspath(__file__))

def get_results(det_folder, test_set, suffix): 
    filepath = os.path.join(det_folder, test_set, "results/detections_{}_results_{}.json".format(test_set, suffix))
    with open(filepath) as f: 
        return json.load(f) 

def flip_box(box):
    """
    box (list, length 4): [x1, y1, w, h]
    """
    # Get top right corner of prediction
    x1 = box[0]
    y1 = box[1]
    w  = box[2] 
    h  = box[3]
    topRight = (x1 + w, y1)
    # Top left corner of flipped box is:
    newTopLeft = (1024. - topRight[0], topRight[1])
    return [newTopLeft[0], newTopLeft[1], w, h]

def convert_dict_to_df(results, mapping, metadata, test_set, flip=False, threshold=0.):
    list_of_image_ids = [] 
    list_of_scores    = [] 
    list_of_bboxes    = [] 
    for res in results: 
        coco_image_id = res["image_id"] 
        coco_img_file = "COCO_{}_{}.png".format(test_set, str(coco_image_id).zfill(12))
        list_of_image_ids.append(mapping[coco_img_file]) 
        list_of_scores.append(res["score"]) 
        list_of_bboxes.append(res["bbox"])
    if flip: 
        list_of_bboxes = [flip_box(_) for _ in list_of_bboxes]
    results_df = pd.DataFrame({"patientId": [pid.split(".")[0] for pid in list_of_image_ids],
                               "score": list_of_scores, 
                               "x": [box[0] for box in list_of_bboxes],
                               "y": [box[1] for box in list_of_bboxes],
                               "w": [box[2] for box in list_of_bboxes],
                               "h": [box[3] for box in list_of_bboxes],
                               "bbox": list_of_bboxes})
    results_df = results_df.sort_values(["patientId", "score"], ascending=False)
    results_df = results_df[results_df.score >= threshold] 
    results_df = results_df.merge(metadata, on="patientId", how="left")  
    return results_df[["patientId", "score", "x", "y", "w", "h", "bbox", "view"]]


with open(MAPPINGS_PATH) as f: 
    mapping = json.load(f) 

with open(MAPPINGS_PATH.replace(test_image_set, "{}_flip".format(test_image_set))) as f: 
    flip_mapping = json.load(f) 

metadata = pd.read_csv(METADATA_PATH) 

def get_TTA_results(fold_imsize, test_image_set, MAIN_DIR):
    TTAs = [] 
    for test_set in [test_image_set, "{}_flip".format(test_image_set)]:
        for suffix in ["original", "scale080", "scale120"]:
            tmp_results = get_results(os.path.join(MAIN_DIR, "peepin_{}".format(fold_imsize, fold_imsize)),
                                      test_set=test_set, suffix=suffix) 
            if test_set == "stage_2_test_flip":
                tmp_df = convert_dict_to_df(tmp_results, 
                                            flip_mapping, 
                                            metadata,
                                            test_set=test_set,
                                            flip=True, 
                                            threshold=0.01)
            elif test_set == "stage_2_test":
                tmp_df = convert_dict_to_df(tmp_results, 
                                            mapping, 
                                            metadata,
                                            test_set=test_set,
                                            flip=False, 
                                            threshold=0.01)
            TTAs.append(tmp_df) 
    return TTAs

execfile(os.path.join(WDIR, "DetectionEnsemble.py"))
def run_ensemble(list_of_dfs, metadata, adjust_score=True):
    list_of_pids = []
    list_of_ensemble_bboxes = [] 
    for pid in np.unique(metadata.patientId): 
        list_of_tmp_dfs = []
        list_of_detections = [] 
        view = metadata[metadata.patientId == pid]["view"].iloc[0]
        for df_index, each_df in enumerate(list_of_dfs): 
            tmp_df = each_df[each_df.patientId == pid]
            list_of_bboxes = []
            for rownum, row in tmp_df.iterrows(): 
                bbox = row.bbox 
                bbox.append(1) 
                bbox.append(row.score) 
                list_of_bboxes.append(bbox) 
            list_of_detections.append(list_of_bboxes) 
        list_of_ensemble_bboxes.append(GeneralEnsemble(list_of_detections, iou_thresh=0.4))
        list_of_pids.append(pid) 
    # Create new DataFrame 
    list_of_new_pids = []
    list_of_bboxes = [] 
    for i, ensemble_bboxes in enumerate(list_of_ensemble_bboxes): 
        for bbox in ensemble_bboxes: 
            list_of_new_pids.append(list_of_pids[i]) 
            list_of_bboxes.append(bbox) 
    ensemble_bbox_df = pd.DataFrame({"patientId": list_of_new_pids,
                                     "x": [box[0] for box in list_of_bboxes],
                                     "y": [box[1] for box in list_of_bboxes],
                                     "w": [box[2] for box in list_of_bboxes], 
                                     "h": [box[3] for box in list_of_bboxes],
                                     "score": [box[5] for box in list_of_bboxes],
                                     "votes": [box[-1] for box in list_of_bboxes],
                                     "bbox":  list_of_bboxes})
    if adjust_score:
        ensemble_bbox_df["score"] = ensemble_bbox_df.score * ensemble_bbox_df.votes
    return ensemble_bbox_df 

imsizes = [224, 256, 288, 320, 352, 384, 416, 448, 480, 512]

fold0_nom = "fold{}_{}".format(0, imsizes[0])
fold1_nom = "fold{}_{}".format(1, imsizes[1])
fold2_nom = "fold{}_{}".format(2, imsizes[2])
fold3_nom = "fold{}_{}".format(3, imsizes[3])
fold4_nom = "fold{}_{}".format(4, imsizes[4])
fold5_nom = "fold{}_{}".format(5, imsizes[5])
fold6_nom = "fold{}_{}".format(6, imsizes[6])
fold7_nom = "fold{}_{}".format(7, imsizes[7])
fold8_nom = "fold{}_{}".format(8, imsizes[8])
fold9_nom = "fold{}_{}".format(9, imsizes[9])

fold0RFCN = run_ensemble(get_TTA_results("fold0_224", test_image_set, RFCN_DETS_DIR.format(fold0_nom)), metadata)
fold1RFCN = run_ensemble(get_TTA_results("fold1_256", test_image_set, RFCN_DETS_DIR.format(fold1_nom)), metadata)
fold2RFCN = run_ensemble(get_TTA_results("fold2_288", test_image_set, RFCN_DETS_DIR.format(fold2_nom)), metadata)
fold3RFCN = run_ensemble(get_TTA_results("fold3_320", test_image_set, RFCN_DETS_DIR.format(fold3_nom)), metadata)
fold4RFCN = run_ensemble(get_TTA_results("fold4_352", test_image_set, RFCN_DETS_DIR.format(fold4_nom)), metadata)
fold5RFCN = run_ensemble(get_TTA_results("fold5_384", test_image_set, RFCN_DETS_DIR.format(fold5_nom)), metadata)
fold6RFCN = run_ensemble(get_TTA_results("fold6_416", test_image_set, RFCN_DETS_DIR.format(fold6_nom)), metadata)
fold7RFCN = run_ensemble(get_TTA_results("fold7_448", test_image_set, RFCN_DETS_DIR.format(fold7_nom)), metadata)
fold8RFCN = run_ensemble(get_TTA_results("fold8_480", test_image_set, RFCN_DETS_DIR.format(fold8_nom)), metadata)
fold9RFCN = run_ensemble(get_TTA_results("fold9_512", test_image_set, RFCN_DETS_DIR.format(fold9_nom)), metadata)

fold0RCNN0 = run_ensemble(get_TTA_results("fold0_224", test_image_set, RCNN0_DETS_DIR.format(fold0_nom)), metadata)
fold1RCNN0 = run_ensemble(get_TTA_results("fold1_256", test_image_set, RCNN0_DETS_DIR.format(fold1_nom)), metadata)
fold2RCNN0 = run_ensemble(get_TTA_results("fold2_288", test_image_set, RCNN0_DETS_DIR.format(fold2_nom)), metadata)
fold3RCNN0 = run_ensemble(get_TTA_results("fold3_320", test_image_set, RCNN0_DETS_DIR.format(fold3_nom)), metadata)
fold4RCNN0 = run_ensemble(get_TTA_results("fold4_352", test_image_set, RCNN0_DETS_DIR.format(fold4_nom)), metadata)
fold5RCNN0 = run_ensemble(get_TTA_results("fold5_384", test_image_set, RCNN0_DETS_DIR.format(fold5_nom)), metadata)
fold6RCNN0 = run_ensemble(get_TTA_results("fold6_416", test_image_set, RCNN0_DETS_DIR.format(fold6_nom)), metadata)
fold7RCNN0 = run_ensemble(get_TTA_results("fold7_448", test_image_set, RCNN0_DETS_DIR.format(fold7_nom)), metadata)
fold8RCNN0 = run_ensemble(get_TTA_results("fold8_480", test_image_set, RCNN0_DETS_DIR.format(fold8_nom)), metadata)
fold9RCNN0 = run_ensemble(get_TTA_results("fold9_512", test_image_set, RCNN0_DETS_DIR.format(fold9_nom)), metadata)

fold0RCNN1 = run_ensemble(get_TTA_results("fold0_224", test_image_set, RCNN1_DETS_DIR.format(fold0_nom)), metadata)
fold1RCNN1 = run_ensemble(get_TTA_results("fold1_256", test_image_set, RCNN1_DETS_DIR.format(fold1_nom)), metadata)
fold2RCNN1 = run_ensemble(get_TTA_results("fold2_288", test_image_set, RCNN1_DETS_DIR.format(fold2_nom)), metadata)
fold3RCNN1 = run_ensemble(get_TTA_results("fold3_320", test_image_set, RCNN1_DETS_DIR.format(fold3_nom)), metadata)
fold4RCNN1 = run_ensemble(get_TTA_results("fold4_352", test_image_set, RCNN1_DETS_DIR.format(fold4_nom)), metadata)
fold5RCNN1 = run_ensemble(get_TTA_results("fold5_384", test_image_set, RCNN1_DETS_DIR.format(fold5_nom)), metadata)
fold6RCNN1 = run_ensemble(get_TTA_results("fold6_416", test_image_set, RCNN1_DETS_DIR.format(fold6_nom)), metadata)
fold7RCNN1 = run_ensemble(get_TTA_results("fold7_448", test_image_set, RCNN1_DETS_DIR.format(fold7_nom)), metadata)
fold8RCNN1 = run_ensemble(get_TTA_results("fold8_480", test_image_set, RCNN1_DETS_DIR.format(fold8_nom)), metadata)
fold9RCNN1 = run_ensemble(get_TTA_results("fold9_512", test_image_set, RCNN1_DETS_DIR.format(fold9_nom)), metadata)


list_of_dfs = [fold0RFCN,  fold1RFCN,  fold2RFCN,  fold3RFCN,  fold4RFCN,
               fold5RFCN,  fold6RFCN,  fold7RFCN,  fold8RFCN,  fold9RFCN,
               fold0RCNN0, fold1RCNN0, fold2RCNN0, fold3RCNN0, fold4RCNN0, 
               fold5RCNN0, fold6RCNN0, fold7RCNN0, fold8RCNN0, fold9RCNN0,
               fold0RCNN1, fold1RCNN1, fold2RCNN1, fold3RCNN1, fold4RCNN1, 
               fold5RCNN1, fold6RCNN1, fold7RCNN1, fold8RCNN1, fold9RCNN1]


final_TTA_ensemble = run_ensemble(list_of_dfs, metadata, adjust_score=False) 
final_TTA_ensemble["adjustedScore"] = final_TTA_ensemble.score * final_TTA_ensemble.votes 
final_TTA_ensemble = final_TTA_ensemble[["patientId", "x", "y", "w", "h", "score", "votes", "adjustedScore"]]
final_TTA_ensemble.to_csv(os.path.join(WDIR, "../../DCNEnsemblePredictions.csv"), index=False)



