## Imports
import os
import random

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

## Seeding 
seed = 2019
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# ======================================================================
# CONFIGURATION: DIRECTORIES & PATHS
# ======================================================================
train_dir = "dataset/train/"       # Path for Training Dataset
val_dir = "dataset/val/"           # Path for Validation Dataset
output_dir = "training_outputs/"   # Path to store weights, excel, and plots

# Automatically create the output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Data Generator
class DataGen(keras.utils.Sequence):
    def __init__(self, ids, path, batch_size=4, img_w=640, img_h=480):
        self.ids = ids
        self.path = path
        self.batch_size = batch_size
        self.img_w = img_w
        self.img_h = img_h
        self.on_epoch_end()
        
    def __load__(self, id_name):
        ## Path
        image_path = os.path.join(self.path, "images", id_name) + ".png"
        mask_path = os.path.join(self.path, "masks", id_name) + ".png"
        
        ## Reading Image
        image = cv2.imread(image_path)
        image = cv2.resize(image, (self.img_w, self.img_h))
        
        ##Reading Mask
        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (self.img_w, self.img_h))
        mask = np.expand_dims(mask, axis=-1)
        
        ## Normalizing 
        image = image/255.0
        mask = mask/255.0
        
        return image, mask
    
    def __getitem__(self, index):
        if(index+1)*self.batch_size > len(self.ids):
            self.batch_size = len(self.ids) - index*self.batch_size
        
        files_batch = self.ids[index*self.batch_size : (index+1)*self.batch_size]
        
        image = []
        mask  = []
        
        for id_name in files_batch:
            _img, _mask = self.__load__(id_name)
            image.append(_img)
            mask.append(_mask)
            
        image = np.array(image)
        mask  = np.array(mask)
        
        return image, mask
    
    def on_epoch_end(self):
        pass
    
    def __len__(self):
        return int(np.ceil(len(self.ids)/float(self.batch_size)))

# Different Blocks
def bn_act(x, act=True):
    x = keras.layers.BatchNormalization()(x)
    if act == True:
        x = keras.layers.Activation("relu")(x)
    return x

def conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = bn_act(x)
    conv = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
    return conv

def stem(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    conv = conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    
    shortcut = keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    
    output = keras.layers.Add()([conv, shortcut])
    return output

def residual_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)
    
    shortcut = keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    
    output = keras.layers.Add()([shortcut, res])
    return output

def upsample_concat_block(x, xskip):
    u = keras.layers.UpSampling2D((2, 2))(x)
    c = keras.layers.Concatenate()([u, xskip])
    return c

def Pretrained_ResUNet(img_h, img_w):
    # 1. Input Layer
    inputs = keras.layers.Input((img_h, img_w, 3))
    
    # 2. Load the Pre-trained ResNet50 Encoder (Fully Trainable)
    encoder = keras.applications.ResNet50(weights="imagenet", include_top=False, input_tensor=inputs)
    
    # 3. Extract Skip Connections from the pre-trained encoder
    s1 = encoder.input            # Shape: (480, 640)
    s2 = encoder.get_layer("conv1_relu").output          # Shape: (240, 320)
    s3 = encoder.get_layer("conv2_block3_out").output    # Shape: (120, 160)
    s4 = encoder.get_layer("conv3_block4_out").output    # Shape: (60, 80)
    
    # 4. Bridge
    b1 = encoder.get_layer("conv4_block6_out").output    # Shape: (30, 40)
    
    # 5. Decoder (Using your existing custom residual blocks)
    f = [16, 32, 64, 128, 256]
    
    u1 = upsample_concat_block(b1, s4)
    d1 = residual_block(u1, f[3])
    
    u2 = upsample_concat_block(d1, s3)
    d2 = residual_block(u2, f[2])
    
    u3 = upsample_concat_block(d2, s2)
    d3 = residual_block(u3, f[1])
    
    u4 = upsample_concat_block(d3, s1)
    d4 = residual_block(u4, f[0])
    
    # Final classification layer
    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d4)
    
    model = keras.models.Model(inputs, outputs)
    
    return model

# Metrics and Loss
smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


# ======================================================================
# TRAINING SETUP WITH TRANSFER LEARNING (SINGLE PHASE)
# ======================================================================

# Function to dynamically fetch IDs from a directory (Replaces CSV)
def get_file_ids(directory):
    img_folder = os.path.join(directory, "images")
    if not os.path.exists(img_folder):
        raise ValueError(f"Could not find directory: {img_folder}")
    return [os.path.splitext(f)[0] for f in os.listdir(img_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Load IDs
train_ids = get_file_ids(train_dir)
valid_ids = get_file_ids(val_dir)
print(f"Found {len(train_ids)} training images and {len(valid_ids)} validation images.")

# Hyperparameters
img_w = 480
img_h = 640
batch_size = 4
epochs = 100

train_gen = DataGen(train_ids, train_dir, img_w=img_w, img_h=img_h, batch_size=batch_size)
valid_gen = DataGen(valid_ids, val_dir, img_w=img_w, img_h=img_h, batch_size=batch_size)

train_steps = len(train_ids) // batch_size
valid_steps = len(valid_ids) // batch_size

# 1. Initialize Model
model = Pretrained_ResUNet(img_h, img_w)

# 2. Compile with a lower learning rate (1e-4) to protect the pre-trained weights
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss=dice_coef_loss, metrics=[dice_coef])

# 3. Callbacks (Configured to save to output_dir)
checkpoint = keras.callbacks.ModelCheckpoint(
    os.path.join(output_dir, "ResUNet_best.h5"), 
    monitor="val_loss", 
    verbose=1, 
    save_best_only=True, 
    save_weights_only=True, 
    mode="min"
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
)

# 4. Train the entire network end-to-end
history = model.fit(
    train_gen, 
    validation_data=valid_gen, 
    steps_per_epoch=train_steps, 
    validation_steps=valid_steps, 
    epochs=epochs, 
    callbacks=[checkpoint, reduce_lr]
)

# Save Last Weights to output_dir
model.save_weights(os.path.join(output_dir, "ResUNet_last.h5"))

# --- Post-Training: Save Excel and Plots ---
# Extract metrics from Keras history
hist_dict = history.history
hist_dict['epoch'] = list(range(1, len(hist_dict['loss']) + 1)) 

# Save to Excel in output_dir
df = pd.DataFrame(hist_dict)
excel_path = os.path.join(output_dir, 'training_metrics.xlsx')
df.to_excel(excel_path, index=False)
print(f'Metrics saved to {excel_path}')

# Plotting
try:
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Training and Validation Loss
    plt.subplot(1, 2, 1)
    plt.plot(hist_dict['epoch'], hist_dict['loss'], label='Train Loss', color='blue')
    plt.plot(hist_dict['epoch'], hist_dict['val_loss'], label='Val Loss', color='orange', linestyle='--')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Dice Loss)')
    plt.legend()
    plt.grid(True)

    # Plot 2: Training and Validation Dice Coefficient
    val_dice_key = 'val_dice_coef' if 'val_dice_coef' in hist_dict else 'val_dice'
    dice_key = 'dice_coef' if 'dice_coef' in hist_dict else 'dice'
    
    plt.subplot(1, 2, 2)
    plt.plot(hist_dict['epoch'], hist_dict.get(val_dice_key, []), label='Val Dice', color='red')
    plt.plot(hist_dict['epoch'], hist_dict.get(dice_key, []), label='Train Dice', color='pink', linestyle='--')
    plt.title('Dice Coefficient over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coeff')
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(output_dir, 'training_plot.jpg')
    plt.savefig(plot_path)
    plt.close()
    print(f'Training plot saved to {plot_path}')
except Exception as e:
    print(f"Failed to save plots: {e}")