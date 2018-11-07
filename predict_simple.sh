echo "Obtaining classification scores ..."
python src/infer/PredictOneClassifier.py

echo "Obtaining bounding boxes ..."
sh models/RelationNetworks/PredictSimpleMelted.sh 
python src/infer/ExtractDeformableTTA.py simple test

echo "Assembling submission ..."
python src/infer/SimpleSubmission.py
