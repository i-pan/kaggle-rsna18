###########
# IMPORTS #
###########

import os

WDIR = os.path.dirname(os.path.abspath(__file__))

import sys
sys.path.insert(0, os.path.join(WDIR, "gradient-checkpointing"))
import memory_saving_gradients
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
from keras import optimizers, layers, utils
K.__dict__["gradients"] = memory_saving_gradients.gradients_memory

import pandas as pd
import numpy as np 
import scipy.misc 
import glob
import json

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
              pretrained="imagenet"): 
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

def apply_clahe(img): 
    img = img / 255. 
    img = exposure.equalize_adapthist(img) 
    img = img * 255. 
    return img 

# == AUGMENTATION == #
def crop_center(img, cropx, cropy):
    y,x = img.shape[:2]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx,:]

def data_augmentation(image):
    # Input should be ONE image with shape: (L, W, CH)
    options = ["gaussian_smooth", "rotate", "zoom", "adjust_gamma"]  
    # Probabilities for each augmentation were arbitrarily assigned 
    which_option = np.random.choice(options)
    if which_option == "gaussian_smooth": 
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

# == I/O == #
def load_sample(train_images, num_train_samples, z): 
    if len(train_images) < num_train_samples: 
        train_sample_images = np.random.choice(train_images, num_train_samples, replace=True)
        train_sample_array  = np.asarray([np.load(arr) for arr in train_sample_images])
        return train_sample_array, []
    train_sample_images = list(set(train_images) - set(z))
    if len(train_sample_images) < num_train_samples: 
        sample_diff = num_train_samples - len(train_sample_images)
        not_sampled = list(set(train_images) - set(train_sample_images))
        train_sample_images.extend(np.random.choice(not_sampled, sample_diff, replace=False)) 
        z = []
    else:
        train_sample_images = np.random.choice(train_sample_images, num_train_samples, replace=False) 
    z.extend(train_sample_images) 
    train_sample_array = np.asarray([np.load(arr) for arr in train_sample_images])
    return train_sample_array, z 

def load_sample_and_labels(df, train_images, num_train_samples, z): 
    if len(train_images) < num_train_samples: 
        train_sample_images = np.random.choice(train_images, num_train_samples, replace=True)
        train_sample_array  = np.asarray([np.load(arr) for arr in train_sample_images])
        return train_sample_array, []
    train_sample_images = list(set(train_images) - set(z))
    if len(train_sample_images) < num_train_samples: 
        sample_diff = num_train_samples - len(train_sample_images)
        not_sampled = list(set(train_images) - set(train_sample_images))
        train_sample_images.extend(np.random.choice(not_sampled, sample_diff, replace=False)) 
        z = []
    else:
        train_sample_images = np.random.choice(train_sample_images, num_train_samples, replace=False) 
    z.extend(train_sample_images) 
    train_sample_ids = [_.split("/")[-1].split(".")[0] for _ in train_sample_images] 
    train_sample_df = df[(df.patientId.isin(train_sample_ids))]
    train_sample_df.index = train_sample_df.patientId 
    train_sample_df = train_sample_df.reindex(train_sample_ids) 
    train_sample_labels = np.asarray(train_sample_df["label"]) 
    train_sample_array = np.asarray([np.load(arr) for arr in train_sample_images])
    return train_sample_array, train_sample_labels, z 

def TTA(img, model, model_name, seed=88, niter=0):
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

############
# VALIDATE #
############

def reduce_learning_rate_or_not(metric_list, direction="max", patience=2):
    # **NOTE: metric_list should have CURRENT metric as last element
    if len(metric_list) < patience + 1: 
        return False 
    else: 
        if direction == "max": 
            if metric_list[-1] <= metric_list[(-1-patience)]:
                return True 
            else: 
                return False 
        elif direction == "min": 
            if metric_list[-1] >= metric_list[(-1-patience)]:
                return True 
            else: 
                return False

def competitionMetric(y_true, y_pred): 
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return float(TP) / (float(FP) + float(FN) + float(TP))

def calculate_metrics(val_results_dict, y_pred, y_val, suffix=""): 
    tmp_kappa_list = []
    tmp_accur_list = [] 
    tmp_f1_list = [] 
    tmp_cm_list = []
    y_val = utils.to_categorical(y_val)[:,-1]
    for each_threshold in np.linspace(0.1, 0.9, 17): 
        tmp_pred = [1 if _ >= each_threshold else 0 for _ in y_pred]
        tmp_kappa_list.append(cohen_kappa_score(tmp_pred, y_val))
        tmp_accur_list.append(accuracy_score(tmp_pred, y_val)) 
        tmp_f1_list.append(f1_score(tmp_pred, y_val))
        tmp_cm_list.append(competitionMetric(tmp_pred, y_val))
    auroc = round(roc_auc_score(y_val, y_pred), 3)
    kappa = round(np.max(tmp_kappa_list), 3)
    accur = round(np.max(tmp_accur_list), 3) 
    cm = round(np.max(tmp_cm_list), 3)
    f1 = round(np.max(tmp_f1_list), 3) 
    val_results_dict["auc{}".format(suffix)].append(auroc)
    val_results_dict["kap{}".format(suffix)].append(kappa)
    val_results_dict["acc{}".format(suffix)].append(accur) 
    val_results_dict["f1{}".format(suffix)].append(f1) 
    val_results_dict["cm{}".format(suffix)].append(cm)
    kappa_threshold = np.linspace(0.1,0.9,17)[tmp_kappa_list.index(np.max(tmp_kappa_list))]
    accur_threshold = np.linspace(0.1,0.9,17)[tmp_accur_list.index(np.max(tmp_accur_list))]
    f1_threshold = np.linspace(0.1,0.9,17)[tmp_f1_list.index(np.max(tmp_f1_list))]
    cm_threshold = np.linspace(0.1,0.9,17)[tmp_cm_list.index(np.max(tmp_cm_list))]
    val_results_dict["threshold_kap{}".format(suffix)].append(round(kappa_threshold, 2))
    val_results_dict["threshold_acc{}".format(suffix)].append(round(accur_threshold, 2))
    val_results_dict["threshold_f1{}".format(suffix)].append(round(f1_threshold, 2))
    val_results_dict["threshold_cm{}".format(suffix)].append(round(cm_threshold, 2))
    return val_results_dict 

def validate(val_results_dict, model_name,
             model, y_val, X_val, valid_ids, valid_views,
             save_weights_path, val_results_path, 
             subepoch,
             batch_size):
    y_pred = np.asarray([TTA(img, model, model_name) for img in X_val])
    val_results_dict = calculate_metrics(val_results_dict, y_pred, y_val) 
    val_results_dict = calculate_metrics(val_results_dict, y_pred[valid_views == "AP"], y_val[valid_views == "AP"], "_AP")
    val_results_dict = calculate_metrics(val_results_dict, y_pred[valid_views == "PA"], y_val[valid_views == "PA"], "_PA")
    val_results_dict["subepoch"].append(subepoch) 
    out_df = pd.DataFrame(val_results_dict)
    out_df.to_csv(os.path.join(val_results_path, "results.csv"), index=False) 
    predictions_df = pd.DataFrame({"patientId": valid_ids, "y_pred": y_pred})
    predictions_df.to_csv(os.path.join(val_results_path, "predictions.csv"), index=False)  
    model.save_weights(os.path.join(save_weights_path, "weights_subepoch_{}.h5".format(str(subepoch).zfill(3))))
    return val_results_dict

def load_and_validate(val_results_dict, 
                      model, model_name,
                      clahe,
                      valid_df, data_dir, 
                      save_weights_path, val_results_path, 
                      subepoch,
                      batch_size):
    # Memory requirements may prevent all validation data from being
    # loaded at once 
    # NOTE: data is NOT preprocessed
    print ">>VALIDATING<<\n"
    X_val = np.asarray([np.load(os.path.join(data_dir, "{}.npy".format(_))) for _ in valid_df.patientId])
    if clahe:
        X_val = np.asarray([apply_clahe(_) for _ in X_val])
    X_val = np.expand_dims(X_val, axis=-1)
    #X_val = preprocess_input(X_val, model_name)
    valid_ids = np.asarray(list(valid_df["patientId"]))
    y_val = np.asarray(list(valid_df["label"]))
    valid_views = np.asarray(list(valid_df["view"]))
    val_results_dict = validate(val_results_dict, model_name,
                                model, y_val, X_val, valid_ids, valid_views,
                                save_weights_path, 
                                val_results_path, 
                                subepoch, batch_size) 
    return val_results_dict

def train(df, fold, 
          model, model_name, 
          subepochs, batch_size, base_lr, augment_p, 
          save_weights_path, save_logs_path, val_results_path,
          data_dir, 
          mode="weighted_loss",
          clahe=False, 
          lr_schedule=[20,10,2],
          load_validation_data=True, 
          validate_every_nth_epoch=5,
          resume=0, 
          num_train_samples=16000):
    # lr_schedule : list of 3 integers OR list of 1 string and 2 integer
    #   - index 0: subepoch for first annealing
    #   - index 1: subepoch interval for annealing after first annealing
    #   - index 2: annealing_factor 
    #   OR
    #   - index 0: "ReduceLROnPlateau"
    #   - index 1: annealing_factor 
    #   - index 2: patience
    if not os.path.exists(save_weights_path): 
        os.makedirs(save_weights_path)
    if not os.path.exists(save_logs_path): 
        os.makedirs(save_logs_path)
    if not os.path.exists(val_results_path):
        os.makedirs(val_results_path)
    train_df = df[(df.fold != fold)] 
    valid_df = df[(df.fold == fold)] 
    # Load the validation data if specified
    if load_validation_data: 
        print "Loading validation data ..."
        X_val = np.asarray([np.load(os.path.join(data_dir, "{}.npy".format(_))) for _ in valid_df.patientId])
        if clahe: 
            X_val = np.asarray([apply_clahe(_) for _ in X_val])
        X_val = np.expand_dims(X_val, axis=-1)
        #X_val = preprocess_input(X_val, model_name)
        print "DONE !" 
    valid_ids = np.asarray(list(valid_df["patientId"]))
    y_val = np.asarray(list(valid_df["label"]))
    valid_views = np.asarray(list(valid_df["view"]))
    if mode == "weighted_loss": 
        train_images = [os.path.join(data_dir, "{}.npy".format(_)) for _ in train_df.patientId]
        z = []
    elif mode == "sample_equally": 
        pos_train_df = train_df[train_df["label"] == 1] 
        neg_train_df = train_df[train_df["label"] == 0] 
        pos_train_images = [os.path.join(data_dir, "{}.npy".format(_)) for _ in pos_train_df.patientId]
        neg_train_images = [os.path.join(data_dir, "{}.npy".format(_)) for _ in neg_train_df.patientId]   
        z_pos = [] 
        z_neg = []
    val_results_dict = {"auc": [], 
                        "kap": [],
                        "acc": [],
                        "f1":  [],
                        "cm":  [],
                        "threshold_kap": [],
                        "threshold_acc": [],
                        "threshold_f1":  [],
                        "threshold_cm":  [],
                        "subepoch": [],
                        "auc_AP": [], 
                        "kap_AP": [],
                        "acc_AP": [],
                        "f1_AP":  [],
                        "cm_AP":  [], 
                        "threshold_kap_AP": [],
                        "threshold_acc_AP": [],
                        "threshold_f1_AP":  [],
                        "threshold_cm_AP":  [],
                        "auc_PA": [], 
                        "kap_PA": [],
                        "acc_PA": [],
                        "f1_PA":  [],
                        "cm_PA":  [],
                        "threshold_kap_PA": [],
                        "threshold_acc_PA": [],
                        "threshold_f1_PA":  [],
                        "threshold_cm_PA":  []}
    lr_annealing_counter = 0 
    for each_subepoch in range(resume, subepochs): 
        suffix = str(each_subepoch).zfill(3) 
        logs_path = os.path.join(save_logs_path, "log_subepoch_{}.csv".format(suffix))
        csvlogger = CSVLogger(logs_path) 
        print "Loading training sample ..."
        if mode == "weighted_loss": 
            X_train, y_train, z = load_sample_and_labels(train_df, train_images, num_train_samples, z) 
            class_weight_dict = {} 
            class_freq_list = []
            y_train = utils.to_categorical(y_train)
            for each_class in range(y_train.shape[1]):
                class_freq_list.append(np.sum(y_train[:,each_class]) / float(y_train.shape[0]))
            for each_class in range(y_train.shape[1]):
                class_weight_dict[each_class] = np.max(class_freq_list) / class_freq_list[each_class]
        elif mode == "sample_equally": 
            X_pos_train, z_pos = load_sample(pos_train_images, num_train_samples / 2, z_pos) 
            X_neg_train, z_neg = load_sample(neg_train_images, num_train_samples / 2, z_neg) 
            X_train = np.vstack((X_pos_train, X_neg_train))
            y_train = np.concatenate((np.repeat(1, len(X_pos_train)),
                                      np.repeat(0, len(X_neg_train))))
            del X_pos_train, X_neg_train
        if clahe: 
            X_train = np.asarray([apply_clahe(_) for _ in X_train]) 
        X_train = np.expand_dims(X_train, axis=-1) 
        print "Augmenting training data ..."
        for index, each_image in enumerate(X_train):
            sys.stdout.write("{}/{} ...\r".format(index+1, len(X_train)))
            sys.stdout.flush() 
            if np.random.binomial(1, 0.5):
                each_image = np.invert(each_image)
            if np.random.binomial(1, 0.5): 
                each_image = np.fliplr(each_image) 
            if np.random.binomial(1, augment_p): 
                X_train[index] = data_augmentation(each_image) 
        X_train = preprocess_input(X_train, model_name) 
        print ("\nDONE !")
        if mode == "weighted_loss": 
            model.fit(X_train, y_train, 
                  batch_size=batch_size, epochs=1,
                  shuffle=True, callbacks=[csvlogger],
                  class_weight=class_weight_dict)
        elif mode == "sample_equally":
            model.fit(X_train, y_train, 
                      batch_size=batch_size, epochs=1,
                      shuffle=True, callbacks=[csvlogger])
        ##### VALIDATE #####
        if (each_subepoch + 1) % validate_every_nth_epoch == 0:
            if load_validation_data:
                val_results_dict = validate(val_results_dict, model_name,
                                            model, y_val, X_val, valid_ids, valid_views,
                                            save_weights_path, val_results_path,
                                            each_subepoch,
                                            batch_size)
            else: 
                val_results_dict = load_and_validate(val_results_dict,
                                            model, model_name,
                                            clahe,
                                            valid_df, data_dir, 
                                            save_weights_path, val_results_path,
                                            each_subepoch,
                                            batch_size)
        ##### LEARNING RATE SCHEDULE #####
        if lr_schedule[0] != "ReduceLROnPlateau":
            if (each_subepoch + 1) >= lr_schedule[0] and (each_subepoch + 1) % lr_schedule[1] == 0:
                lr_annealing_counter += 1.
                # Step-wise learning rate annealing schedule 
                new_lr = base_lr / (lr_schedule[2] ** lr_annealing_counter)
                K.set_value(model.optimizer.lr, new_lr)
        else:
            if (each_subepoch + 1) % validate_every_nth_epoch == 0:
                if reduce_learning_rate_or_not(val_results_dict["acc"], "max", lr_schedule[2]):
                    lr_annealing_counter += 1. 
                    new_lr = base_lr / (lr_schedule[1] ** lr_annealing_counter)
                    K.set_value(model.optimizer.lr, new_lr)


##########
# SCRIPT #
##########

import json 

# Specify GPU
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

with open(os.path.join(WDIR, "../../SETTINGS.json")) as f: 
    SETTINGS_JSON = json.load(f) 

df = pd.read_csv(os.path.join(WDIR, "../..", SETTINGS_JSON["TRAIN_INFO_DIR"], "stratified_folds_df.csv"))
df["label"] = [1 if _ == "Lung Opacity" else 0 for _ in df["class"]]

#####################
# InceptionResNetV2 #
#####################
fold = 0
input_size = 256
fold_save_dir = os.path.join(WDIR, "../../models/one_classifier/snapshots/binary/InceptionResNetV2/fold{}".format(fold)) 
model = get_model(InceptionResNetV2, 0, 5e-5, dropout=None, input_shape=(input_size,input_size,1),
    pretrained=os.path.join(WDIR, "../../models/pretrained/InceptionResNetV2_NIH15_Px256.h5"))
model_name = "inception"
train(df, fold, model, model_name, 15, 16, 5e-5, 0.5,
      os.path.join(fold_save_dir, "l0/weights/"),
      os.path.join(fold_save_dir, "l0/logs/"),
      os.path.join(fold_save_dir, "l0/val-results/"),
      os.path.join(WDIR, "../../", SETTINGS_JSON["TRAIN_IMAGES_CLEAN_DIR"], "resized/i{}/".format(input_size)),
      mode="weighted_loss",
      lr_schedule=[6,3,2.],
      validate_every_nth_epoch=2,
      load_validation_data=False,
      num_train_samples=8000)


