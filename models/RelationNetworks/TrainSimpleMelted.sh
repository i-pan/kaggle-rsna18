cd $(dirname $(readlink -f $0 || realpath $0))

rm -rf ../../data/coco/cache/

python skpeep/rcnn_end2end_train_test.py --cfg skpeep/unfreeze/cfgs/peepin_fold1_256.yaml


python skpeep/rcnn_end2end_train_test.py --cfg skpeep/unfreeze/cfgs/peepin_fold3_320.yaml


python skpeep/rcnn_end2end_train_test.py --cfg skpeep/unfreeze/cfgs/peepin_fold5_384.yaml


python skpeep/rcnn_end2end_train_test.py --cfg skpeep/unfreeze/cfgs/peepin_fold7_448.yaml


python skpeep/rcnn_end2end_train_test.py --cfg skpeep/unfreeze/cfgs/peepin_fold9_512.yaml

