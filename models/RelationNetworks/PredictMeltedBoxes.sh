cd $(dirname $(readlink -f $0 || realpath $0))

rm -rf ../../data/coco/cache/

../../src/infer/PredictDeformableTTA.py test skpeep/unfreeze/cfgs/peepin_fold0_224.yaml skpeep/rcnn_test.py skpeep/unfreeze/test_cfgs/ skpeep/unfreeze/flip/ skpeep/unfreeze/scale/

../../src/infer/PredictDeformableTTA.py test skpeep/unfreeze/cfgs/peepin_fold1_256.yaml skpeep/rcnn_test.py skpeep/unfreeze/test_cfgs/ skpeep/unfreeze/flip/ skpeep/unfreeze/scale/

../../src/infer/PredictDeformableTTA.py test skpeep/unfreeze/cfgs/peepin_fold2_288.yaml skpeep/rcnn_test.py skpeep/unfreeze/test_cfgs/ skpeep/unfreeze/flip/ skpeep/unfreeze/scale/

../../src/infer/PredictDeformableTTA.py test skpeep/unfreeze/cfgs/peepin_fold3_320.yaml skpeep/rcnn_test.py skpeep/unfreeze/test_cfgs/ skpeep/unfreeze/flip/ skpeep/unfreeze/scale/

../../src/infer/PredictDeformableTTA.py test skpeep/unfreeze/cfgs/peepin_fold4_352.yaml skpeep/rcnn_test.py skpeep/unfreeze/test_cfgs/ skpeep/unfreeze/flip/ skpeep/unfreeze/scale/

../../src/infer/PredictDeformableTTA.py test skpeep/unfreeze/cfgs/peepin_fold5_384.yaml skpeep/rcnn_test.py skpeep/unfreeze/test_cfgs/ skpeep/unfreeze/flip/ skpeep/unfreeze/scale/

../../src/infer/PredictDeformableTTA.py test skpeep/unfreeze/cfgs/peepin_fold6_416.yaml skpeep/rcnn_test.py skpeep/unfreeze/test_cfgs/ skpeep/unfreeze/flip/ skpeep/unfreeze/scale/

../../src/infer/PredictDeformableTTA.py test skpeep/unfreeze/cfgs/peepin_fold7_448.yaml skpeep/rcnn_test.py skpeep/unfreeze/test_cfgs/ skpeep/unfreeze/flip/ skpeep/unfreeze/scale/

../../src/infer/PredictDeformableTTA.py test skpeep/unfreeze/cfgs/peepin_fold8_480.yaml skpeep/rcnn_test.py skpeep/unfreeze/test_cfgs/ skpeep/unfreeze/flip/ skpeep/unfreeze/scale/

../../src/infer/PredictDeformableTTA.py test skpeep/unfreeze/cfgs/peepin_fold9_512.yaml skpeep/rcnn_test.py skpeep/unfreeze/test_cfgs/ skpeep/unfreeze/flip/ skpeep/unfreeze/scale/

