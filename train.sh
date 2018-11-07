echo "Training classifier ensemble ..."
python src/train/TrainClassifierEnsemble.py 

echo "Training RetinaNet on positive images ..."
python src/train/TrainSkippy.py

echo "Training RetinaNet on concatenated images ..."
python src/train/TrainCloud.py 

echo "Training Deformable R-FCN [unfrozen backbone] ..." 
sh models/DeformableConvNets/TrainDeformableRFCN.sh 

echo "Training Deformable Relation Network [unfrozen backbone] ..."
sh models/RelationNetworks/TrainMelted.sh 

echo "Training Deformable Relation Network [frozen backbone] ..."
sh models/RelationNetworks/TrainFrozen.sh 
