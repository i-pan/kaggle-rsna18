echo "Predicting classification scores ..."
python src/infer/PredictClassifierEnsemble.py 

echo "Predicting Deformable R-FCN ..."
sh models/DeformableConvnets/PredictBoxes.sh 

echo "Predicting Deformable Relation Networks ..."
sh models/RelationNetworks/PredictMeltedBoxes.sh
sh models/RelationNetworks/PredictFrozenBoxes.sh

echo "Predicting ConcatRetinaNet ..."
python src/infer/PredictConcatRetinaEnsemble.py

echo "Predicting RetinaNet ..."
python src/infer/PredictRetinaEnsemble.py

echo "Merging predictions from Deformable R-FCN + Relation Networks ..."
python src/infer/ExtractDeformableTTA.py complete test 

echo "Creating final submission ..."
python src/infer/AssembleSubmission.py

echo "DONE !"
