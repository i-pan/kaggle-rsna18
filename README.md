# Installation 

All requirements should be detailed in `requirements.txt`. We strongly recommend using Anaconda (Python 2.7). For example:

```
conda create -n kaggle-rsna18 python=2.7 pip
pip install -r requirements.txt
```

We used CUDA 8.0/cuDNN v6.0. You may need to change the TensorFlow version for different versions of CUDA/cuDNN. Please use Keras 2.2.0. 

We will refer to the top-level directory as `$TOP`.

## Keras

We encountered the following error using Keras 2.2.0. 
`TypeError: softmax() got an unexpected keyword argument 'axis'`

This can be fixed by editing `keras/backend/tensorflow_backend.py` in your Python packages. Look for `tf.nn.softmax(x, axis=axis)` and remove the `axis` argument. 

## MXNet

Deformable R-FCN/relation networks run with the MXNet backend. Please install the version corresponding to your CUDA version. We used `pip install mxnet-cu80==1.3.0`. 

## Keras-RetinaNet

To install keras-retinanet:
`cd $TOP/models/RetinaNet/ ; pip install .` 

If you are still having issues importing from `keras-retinanet`, try `export PYTHONPATH="${PYTHONPATH}:$TOP/models/RetinaNet/"` with the absolute file path. 

# Data

We strongly recommend sticking with the directories specified in `SETTINGS.json`. There is no guarantee that the code will work as intended if changes are made. We will provide instructions for training and testing on the challenge data.  

In `$TOP`, run the following commands: 
```
mkdir data ; cd data 
# Download the challenge data here 
kaggle competitions download -c rsna-pneumonia-detection-challenge
unzip stage_2_detailed_class_info.csv.zip 
unzip stage_2_train_labels.csv.zip 
mv stage_2_detailed_class_info.csv detailed_class_info.csv 
mv stage_2_train_labels.csv train_labels.csv 
mkdir train_dicoms test_dicoms 
cd train_dicoms ; unzip ../stage_2_train_images.zip 
cd ../test_dicoms ; unzip ../stage_2_test_images.zip 
```

Go back to `$TOP`. We trained our models on the stage 1 training data and did NOT retrain on stage 2 training data. Thus to reproduce our methodology we recommend running: 

`python src/etl/0_Stage1Filter.py`

This will filter out the stage 1 training data. Then, run the following script to get all the data into the necessary formats. 

`sh prepare_data.sh`

# Training 

In `$TOP/.keras/keras.json` you will find the Keras configuration we used to train some of our models. We adapted the Keras code for existing model architectures to accept grayscale (1-channel) input. You can find this code in `src/grayscale-models/`.

To train all models in the ensemble:

`sh train.sh`

Comment out lines if you only wish to train specific models. 

## Trained Models 
To download the trained models we used for the challenge, use: 

```
wget --no-check-certificate \
                         -r \
'https://docs.google.com/uc?export=download&id=12abFXy7-FOwoKxFSJ__IbOGm9oQDU7CQ' \
-O models.tar.gz
```

You can delete the existing `models/` directory and replace it with this one. It contains all of the code in addition to the model weights and pretrained models (see below). The file is 22 GB so the download may take a while.  

## Pretrained Models
We pretrained InceptionResNetV2, Xception, and DenseNet169 on the NIH ChestXray14 dataset. The training code for the classification ensemble depends on the existence of the pretrained models. You can download them via the following command: 

```
wget --no-check-certificate \
                         -r \
'https://docs.google.com/uc?export=download&id=1rI_WSlot6ZNa_ERdLSCsGquUXEK_ikYb' \
-O pretrained.zip
```

Unzip them into `models/pretrained`.

## Multiple GPUs

Please see the training scripts and modify `CUDA_VISIBLE_DEVICES`. 

For deformable R-FCN/relation networks, you will need to edit the `gpus` option of the YAML config files located in: 

`$TOP/models/<DeformableConvNets|RelationNetworks>/skpeep/<unfreeze|default>/cfgs`

Note that there is a separate YAML config file for each training fold, and the `gpus` option must be changed in each one. For training on multiple GPUs, follow this format: `gpus: '0,1,2'`.

## Model Checkpoints

After training the classifier ensemble, you will have to move 1 model checkpoint per fold into a new folder, depending on which checkpoint you have selected (we used highest AUC+F1). 

Model checkpoints will be saved in `$TOP/models/classifiers/snapshots/<binary|multiclass>`. Create the following directories:

`mkdir $TOP/models/classifiers/binary $TOP/models/classifiers/multiclass/`

For each fold, you will need to select a model checkpoint and move it to `$TOP/models/classifiers/<binary|multiclass>`.

For example: 

`mv $TOP/models/classifiers/snapshots/binary/InceptionResNetV2/fold0/weights/weights_subepoch_005.h5 $TOP/models/classifiers/binary/InceptionResNetV2_Fold0Px256_e005.h5` 

Please follow this file name format because the inference code depends on it. Make sure you include the exact model architecture name, fold, image resolution, and a suffix for the epoch you used. Note that model checkpoints are saved within `classifiers/snapshots` but final checkpoints should be stored in `classifiers`. There are also separate folders for `binary` versus `multiclass` and it's important to retain this distinction. 

For RetinaNet models, model checkpoints will be stored within `$TOP/models/RetinaNet/output/<cloud|skippy>/fold[0-9]_384/`. Once you have selected your preferred checkpoint, please store it in `$TOP/models/RetinaNet/output/<cloud|skippy>/fold[0-9]_384_<resnet101|resnet152>_csv_<epochNum>.h5`, which is one directory above where checkpoints were originally saved. 

For deformable R-FCN/relation networks, you do not need to do anything to the model checkpoints. We just used the final checkpoint. The inference code depends on the checkpoint being saved in its original location. 

# Inference 

`sh predict.sh` 

This should produce predictions for the test data in `$TOP/FinalSubmission.csv`.

# Simple Model 

This code will train 1 InceptionResNetV2 classifier at 256x256px and 5 deformable relation networks. This model achieved 0.253 on stage 2 private LB. 

`sh train_simple.sh` 

In this case, the model checkpoints will be saved in `$TOP/models/one_classifier/snapshots/binary/InceptionResNetV2/fold0/weights/`. You will need to move your selected checkpoint to `$TOP/models/one_classifier/binary/`.

`sh predict_simple.sh`

# Acknowledgements

We used code from the following repositories: 

https://github.com/msracver/Deformable-ConvNets

https://github.com/msracver/Relation-Networks-for-Object-Detection

https://github.com/fizyr/keras-retinanet

https://github.com/ahrnbom/ensemble-objdet
