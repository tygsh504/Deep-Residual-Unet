## tune.py
import os
import random
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau
import keras_tuner as kt

## Seeding 
seed = 2019
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# ==============================================================================
# Configuration: Directories & Paths
# ==============================================================================
train_dir = "dataset/train/"
val_dir = "dataset/val/"

img_w = 480
img_h = 640
fixed_batch_size = 4 

def get_file_ids(directory):
    img_folder = os.path.join(directory, "images")
    if not os.path.exists(img_folder):
        return []
    return [os.path.splitext(f)[0] for f in os.listdir(img_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# ==============================================================================
# Data Generator
# ==============================================================================
class DataGen(keras.utils.Sequence):
    def __init__(self, ids, path, batch_size=4, img_h=640, img_w=480):
        self.ids = ids
        self.path = path
        self.batch_size = batch_size
        self.img_h = img_h
        self.img_w = img_w
        self.on_epoch_end()
        
    def __load__(self, id_name):
        image_path = os.path.join(self.path, "images", id_name) + ".png"
        mask_path = os.path.join(self.path, "masks", id_name) + ".png"
        
        # cv2.resize expects (width, height)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (self.img_w, self.img_h))
        
        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (self.img_w, self.img_h))
        mask = np.expand_dims(mask, axis=-1)
        
        return image / 255.0, mask / 255.0
    
    def __getitem__(self, index):
        if (index + 1) * self.batch_size > len(self.ids):
            self.batch_size = len(self.ids) - index * self.batch_size
        
        files_batch = self.ids[index * self.batch_size : (index + 1) * self.batch_size]
        image, mask = [], []
        
        for id_name in files_batch:
            _img, _mask = self.__load__(id_name)
            image.append(_img)
            mask.append(_mask)
            
        return np.array(image, dtype=np.float32), np.array(mask, dtype=np.float32)
    
    def on_epoch_end(self): pass
    
    def __len__(self):
        return int(np.ceil(len(self.ids) / float(self.batch_size)))

# ==============================================================================
# Custom Architecture (From ori_training.py)
# ==============================================================================
def bn_act(x, act=True):
    x = keras.layers.BatchNormalization()(x)
    return keras.layers.Activation("relu")(x) if act else x

def conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = bn_act(x)
    return keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)

def stem(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    conv = conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    shortcut = keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    return keras.layers.Add()([conv, shortcut])

def residual_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)
    shortcut = keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    return keras.layers.Add()([shortcut, res])

def upsample_concat_block(x, xskip):
    u = keras.layers.UpSampling2D((2, 2))(x)
    return keras.layers.Concatenate()([u, xskip])

smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

# ==============================================================================
# Keras Tuner Setup
# ==============================================================================
def build_model(hp):
    f = [16, 32, 64, 128, 256]
    inputs = keras.layers.Input((img_h, img_w, 3))
    
    e0 = inputs
    e1 = stem(e0, f[0])
    e2 = residual_block(e1, f[1], strides=2)
    e3 = residual_block(e2, f[2], strides=2)
    e4 = residual_block(e3, f[3], strides=2)
    e5 = residual_block(e4, f[4], strides=2)
    
    b0 = conv_block(e5, f[4], strides=1)
    b1 = conv_block(b0, f[4], strides=1)
    
    u1 = upsample_concat_block(b1, e4)
    d1 = residual_block(u1, f[4])
    u2 = upsample_concat_block(d1, e3)
    d2 = residual_block(u2, f[3])
    u3 = upsample_concat_block(d2, e2)
    d3 = residual_block(u3, f[2])
    u4 = upsample_concat_block(d3, e1)
    d4 = residual_block(u4, f[1])
    
    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d4)
    model = keras.models.Model(inputs, outputs)
    
    # Define Hyperparameters to Tune
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4, 5e-5])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), 
        loss='binary_crossentropy', 
        metrics=[dice_coef]
    )
    return model

# ==============================================================================
# Run Tuner
# ==============================================================================
train_ids = get_file_ids(train_dir)
valid_ids = get_file_ids(val_dir)

train_gen = DataGen(train_ids, train_dir, batch_size=fixed_batch_size, img_h=img_h, img_w=img_w)
valid_gen = DataGen(valid_ids, val_dir, batch_size=fixed_batch_size, img_h=img_h, img_w=img_w)

tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=4,           
    executions_per_trial=1, 
    directory='tuning_outputs',
    project_name='custom_resunet_tuning'
)

reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)

print("Starting Hyperparameter tuning...")
tuner.search(train_gen, validation_data=valid_gen, epochs=5, callbacks=[reduce_lr])

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"\n==========================================")
print(f"Optimal Learning Rate found: {best_hps.get('learning_rate')}")
print(f"==========================================")