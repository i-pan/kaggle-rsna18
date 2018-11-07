###########
# IMPORTS #
###########
import pandas as pd
import scipy.misc 
import pydicom 
import glob 
import sys
import os, json 

WDIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(WDIR, "../../SETTINGS.json")) as f: 
    SETTINGS_JSON = json.load(f) 

##########
# SCRIPT #
##########

train_dicoms_dir = os.path.join(WDIR, "../../", SETTINGS_JSON["RAW_TRAIN_DICOMS_DIR"])
test_dicoms_dir  = os.path.join(WDIR, "../../", SETTINGS_JSON["RAW_TEST_DICOMS_DIR"])

list_of_train_dicoms = glob.glob(os.path.join(train_dicoms_dir, "*"))
list_of_test_dicoms  = glob.glob(os.path.join(test_dicoms_dir, "*"))

train_metadata = pd.DataFrame() 
for i, each_train_dicom in enumerate(list_of_train_dicoms): 
    sys.stdout.write("Getting metadata: {}/{} ...\r".format(i+1, len(list_of_train_dicoms)))
    sys.stdout.flush() 
    tmp_dicom = pydicom.read_file(each_train_dicom)
    tmp_metadata = pd.DataFrame({"patientId": [each_train_dicom.split("/")[-1].split(".")[0]],
                                 "sex":  [tmp_dicom.PatientSex],
                                 "view": [tmp_dicom.ViewPosition],
                                 "age":  [tmp_dicom.PatientAge]})
    train_metadata = train_metadata.append(tmp_metadata) 

train_metadata.to_csv(os.path.join(WDIR, "../../", SETTINGS_JSON["TRAIN_INFO_DIR"], "train_metadata.csv"), index=False)


test_metadata = pd.DataFrame() 
for i, each_test_dicom in enumerate(list_of_test_dicoms): 
    sys.stdout.write("Getting metadata: {}/{} ...\r".format(i+1, len(list_of_test_dicoms)))
    sys.stdout.flush() 
    tmp_dicom = pydicom.read_file(each_test_dicom)
    tmp_metadata = pd.DataFrame({"patientId": [each_test_dicom.split("/")[-1].split(".")[0]],
                                 "sex":  [tmp_dicom.PatientSex],
                                 "view": [tmp_dicom.ViewPosition],
                                 "age":  [tmp_dicom.PatientAge]})
    test_metadata = test_metadata.append(tmp_metadata) 

test_metadata.to_csv(os.path.join(WDIR, "../../", SETTINGS_JSON["TEST_INFO_DIR"], "test_metadata.csv"), index=False)


