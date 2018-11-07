echo "Training classifier ..."
python src/train/TrainOneClassifier.py

echo "Training Deformable Relation Network [unfrozen backbone] ..."
sh models/RelationNetworks/TrainSimpleMelted.sh
