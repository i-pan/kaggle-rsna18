# Specify GPU 
import os, json

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

WDIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(WDIR, "../../SETTINGS.json")) as f: 
    SETTINGS_JSON = json.load(f) 

BINARY_MODELS_DIR = os.path.join(WDIR, "../../models/one_classifier/binary/")
TEST_IMAGES_DIR   = os.path.join(WDIR, "../../", SETTINGS_JSON["TEST_IMAGES_CLEAN_DIR"], "orig")

###########
# IMPORTS #
###########

import sys
sys.path.insert(0, os.path.join(WDIR, "../grayscale-models"))
from inception_resnet_v2_gray import InceptionResNetV2 
from mobilenet_v2_gray import MobileNetV2 
from densenet_gray import DenseNet121, DenseNet169, DenseNet201 
from resnet50_gray import ResNet50 
from xception_gray import Xception 

from keras.layers import Dropout, Flatten, Dense, Input, Concatenate
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.engine import Model 
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras import backend as K 
from keras import optimizers, layers

import pandas as pd
import numpy as np 
import scipy.misc 
import glob
import os

from scipy.ndimage.interpolation import zoom, rotate
from scipy.ndimage.filters import gaussian_filter 

from skimage import exposure 

from sklearn.metrics import roc_auc_score, cohen_kappa_score, accuracy_score, f1_score 

################
# KERAS MODELS #
################
   
def get_model(base_model, 
              layer, 
              lr=1e-3, 
              input_shape=(224,224,1), 
              classes=2,
              activation="softmax",
              dropout=None, 
              pooling="avg", 
              weights=None,
              pretrained=None): 
    base = base_model(input_shape=input_shape,
                      include_top=False,
                      weights=pretrained, 
                      channels="gray") 
    if pooling == "avg": 
        x = GlobalAveragePooling2D()(base.output) 
    elif pooling == "max": 
        x = GlobalMaxPooling2D()(base.output) 
    elif pooling is None: 
        x = Flatten()(base.output) 
    if dropout is not None: 
        x = Dropout(dropout)(x) 
    x = Dense(classes, activation=activation)(x) 
    model = Model(inputs=base.input, outputs=x) 
    if weights is not None: 
        model.load_weights(weights) 
    for l in model.layers[:layer]:
        l.trainable = False 
    model.compile(loss="binary_crossentropy", metrics=["accuracy"], 
                  optimizer=optimizers.Adam(lr)) 
    return model

##########
## DATA ##
##########

# == PREPROCESSING == #
def preprocess_input(x, model):
    x = x.astype("float32")
    if model in ("inception","xception","mobilenet"): 
        x /= 255.
        x -= 0.5
        x *= 2.
    if model in ("densenet"): 
        x /= 255.
        if x.shape[-1] == 3:
            x[..., 0] -= 0.485
            x[..., 1] -= 0.456
            x[..., 2] -= 0.406 
            x[..., 0] /= 0.229 
            x[..., 1] /= 0.224
            x[..., 2] /= 0.225 
        elif x.shape[-1] == 1: 
            x[..., 0] -= 0.449
            x[..., 0] /= 0.226
    elif model in ("resnet","vgg"):
        if x.shape[-1] == 3:
            x[..., 0] -= 103.939
            x[..., 1] -= 116.779
            x[..., 2] -= 123.680
        elif x.shape[-1] == 1: 
            x[..., 0] -= 115.799
    return x

# == AUGMENTATION == #
def crop_center(img, cropx, cropy):
    y,x = img.shape[:2]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx,:]

def data_augmentation(image):
    # Input should be ONE image with shape: (L, W, CH)
    options = ["gaussian_smooth", "vertical_flip", "rotate", "zoom", "adjust_gamma"] 
    # Probabilities for each augmentation were arbitrarily assigned 
    which_option = np.random.choice(options)
    if which_option == "vertical_flip":
        image = np.fliplr(image)
    if which_option == "horizontal_flip": 
        image = np.flipud(image) 
    elif which_option == "gaussian_smooth": 
        sigma = np.random.uniform(0.2, 1.0)
        image = gaussian_filter(image, sigma)
    elif which_option == "zoom": 
      # Assumes image is square
        min_crop = int(image.shape[0]*0.85)
        max_crop = int(image.shape[0]*0.95)
        crop_size = np.random.randint(min_crop, max_crop) 
        crop = crop_center(image, crop_size, crop_size)
        if crop.shape[-1] == 1: crop = crop[:,:,0]
        image = scipy.misc.imresize(crop, image.shape) 
    elif which_option == "rotate":
        angle = np.random.uniform(-15, 15)
        image = rotate(image, angle, reshape=False)
    elif which_option == "adjust_gamma": 
        image = image / 255. 
        image = exposure.adjust_gamma(image, np.random.uniform(0.75,1.25))
        image = image * 255. 
    if len(image.shape) == 2: image = np.expand_dims(image, axis=2)
    return image 

def TTA(img, model, model_name, seed=88, niter=4):
    np.random.seed(seed)
    original_img = img.copy() 
    inverted_img = np.invert(img.copy())
    hflipped_img = np.fliplr(img.copy())
    original_img_array = np.empty((niter+1, img.shape[0], img.shape[1], img.shape[2]))
    inverted_img_array = original_img_array.copy() 
    hflipped_img_array = original_img_array.copy() 
    original_img_array[0] = original_img 
    inverted_img_array[0] = inverted_img 
    hflipped_img_array[0] = hflipped_img 
    for each_iter in range(niter): 
        original_img_array[each_iter+1] = data_augmentation(original_img) 
        inverted_img_array[each_iter+1] = data_augmentation(inverted_img) 
        hflipped_img_array[each_iter+1] = data_augmentation(hflipped_img)
    tmp_array = np.vstack((original_img_array, inverted_img_array, hflipped_img_array))
    tmp_array = preprocess_input(tmp_array, model_name)
    if int(model.get_output_at(-1).get_shape()[1]) == 1: 
        prediction = np.mean(model.predict(tmp_array)[:,0])
    else: 
        prediction = np.mean(model.predict(tmp_array)[:,-1])
    return prediction

##########
# SCRIPT #
##########

model_ensemble_weights = glob.glob(os.path.join(BINARY_MODELS_DIR, "*"))

model_name_dict = {"InceptionResNetV2": "inception"}

# Load each model 
model_architectures = [] 
model_ensemble = [] 
for model_weights in model_ensemble_weights:
    arch = model_weights.split("/")[-1].split("_")[0]
    classifier_type = model_weights.split("/")[-2]
    imsize = model_weights.split("/")[-1].split("_")[1].split("Px")[-1]
    imsize = int(imsize)
    model_architectures.append(arch) 
    if classifier_type == "binary": 
        num_classes = 2
    elif classifier_type == "multiclass": 
        num_classes = 3 
    model_ensemble.append(get_model(eval(arch), 
                                    layer=0, 
                                    classes=num_classes, 
                                    input_shape=(imsize, imsize, 1),
                                    weights=model_weights))

# Predict on test images 
path_to_test_images = os.path.join(TEST_IMAGES_DIR) 
test_images = glob.glob(os.path.join(path_to_test_images, "*"))

list_of_test_predictions = [] 
list_of_test_pids = [] 
for img_index, imgfile in enumerate(test_images): 
    sys.stdout.write("Predicting: {}/{} ...\r".format(img_index+1, len(test_images)))
    sys.stdout.flush() 
    list_of_test_pids.append(imgfile.split("/")[-1].split(".")[0])
    img = scipy.misc.imread(imgfile) 
    img = np.expand_dims(img, axis=-1) 
    list_of_individual_model_preds = [] 
    for model_index, model in enumerate(model_ensemble): 
        model_name = model_name_dict[model_architectures[model_index]]
        # Get the input shape directly from the model
        input_size = int(model.get_input_at(0).get_shape()[1])
        # We assume image is already square 
        resize_factor = float(input_size) / img.shape[0] 
        resized_img = zoom(img.copy(), [resize_factor, resize_factor, 1.], order=1, prefilter=False) 
        list_of_individual_model_preds.append(TTA(resized_img, model, model_name))
    list_of_test_predictions.append(list_of_individual_model_preds)

pred_df = pd.DataFrame(np.vstack(list_of_test_predictions))
pred_df.columns = [_.split("/")[-1].replace(".h5", "") for _ in model_ensemble_weights]
pred_df["patientId"] = list_of_test_pids 
pred_df["ensembleScore"] = np.mean(np.vstack(list_of_test_predictions), axis=1) 

pred_df.to_csv(os.path.join(WDIR, "../../OneClassifierScores.csv"), index=False)


