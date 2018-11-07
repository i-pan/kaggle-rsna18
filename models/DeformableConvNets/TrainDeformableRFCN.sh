cd $(dirname $(readlink -f $0 || realpath $0))

python skpeep/rfcn_end2end_train_test.py --cfg skpeep/unfreeze/cfgs/peepin_fold0_224.yaml

python skpeep/rfcn_end2end_train_test.py --cfg skpeep/unfreeze/cfgs/peepin_fold1_256.yaml

python skpeep/rfcn_end2end_train_test.py --cfg skpeep/unfreeze/cfgs/peepin_fold2_288.yaml

python skpeep/rfcn_end2end_train_test.py --cfg skpeep/unfreeze/cfgs/peepin_fold3_320.yaml

python skpeep/rfcn_end2end_train_test.py --cfg skpeep/unfreeze/cfgs/peepin_fold4_352.yaml

python skpeep/rfcn_end2end_train_test.py --cfg skpeep/unfreeze/cfgs/peepin_fold5_384.yaml

python skpeep/rfcn_end2end_train_test.py --cfg skpeep/unfreeze/cfgs/peepin_fold6_416.yaml

python skpeep/rfcn_end2end_train_test.py --cfg skpeep/unfreeze/cfgs/peepin_fold7_448.yaml

python skpeep/rfcn_end2end_train_test.py --cfg skpeep/unfreeze/cfgs/peepin_fold8_480.yaml

python skpeep/rfcn_end2end_train_test.py --cfg skpeep/unfreeze/cfgs/peepin_fold9_512.yaml

