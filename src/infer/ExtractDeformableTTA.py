import argparse, os, json

WDIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser() 

parser.add_argument("MODE", type=str)
parser.add_argument("test_image_set", type=str)
parser.add_argument("RFCN_DETS_DIR",  type=str, nargs="?", default="../../models/DeformableConvNets/skpeep/unfreeze/output/{}/rfcn_dcn_default/") 
parser.add_argument("RCNN0_DETS_DIR", type=str, nargs="?", default="../../models/RelationNetworks/skpeep/unfreeze/output/{}/rcnn_dcn_default/")
parser.add_argument("RCNN1_DETS_DIR", type=str, nargs="?", default="../../models/RelationNetworks/skpeep/default/output/{}/rcnn_dcn_default") 
args = parser.parse_args() 

test_image_set = args.test_image_set 
RFCN_DETS_DIR  = os.path.join(WDIR, args.RFCN_DETS_DIR)
RCNN0_DETS_DIR = os.path.join(WDIR, args.RCNN0_DETS_DIR)
RCNN1_DETS_DIR = os.path.join(WDIR, args.RCNN1_DETS_DIR)

with open(os.path.join(WDIR, "../../SETTINGS.json")) as f: 
    SETTINGS_JSON = json.load(f) 

MAPPINGS_PATH = os.path.join(WDIR, "../../data/mappings/{}_image_to_coco.json".format(test_image_set)) 
METADATA_PATH = os.path.join(WDIR, "../../", SETTINGS_JSON["TEST_INFO_DIR"], "test_metadata.csv") 

WDIR = os.path.dirname(os.path.abspath(__file__))

if args.MODE == "complete": 
    execfile(os.path.join(WDIR, "_ExtractDeformableTTA.py"))
elif args.MODE == "simple":
    execfile(os.path.join(WDIR, "_ExtractSimpleDeformTTA.py"))
else:
    raise(Exception("MODE must be one of: [simple, complete]"))

