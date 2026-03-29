## train.py
import os
import random
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

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
output_dir = "training_outputs/"

os.makedirs(output_dir, exist_ok=True)

img_w = 480
img_h = 640

# --- HYPERPARAMETERS ---
BATCH_SIZE = 4
OPTIMAL_LR = 1e-4  # Change this to what tune.py found!
EPOCHS = 100
# -----------------------

def get_file_ids(directory):
    img_folder = os.path.join(directory, "images")
    if not os.path.exists(img_folder): return []
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

def ResUNet(img_h, img_w):
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
    return keras.models.Model(inputs, outputs)

smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

# ==============================================================================
# Training Setup
# ==============================================================================
train_ids = get_file_ids(train_dir)
valid_ids = get_file_ids(val_dir)
print(f"Found {len(train_ids)} training images and {len(valid_ids)} validation images.")

train_gen = DataGen(train_ids, train_dir, batch_size=BATCH_SIZE, img_h=img_h, img_w=img_w)
valid_gen = DataGen(valid_ids, val_dir, batch_size=BATCH_SIZE, img_h=img_h, img_w=img_w)

train_steps = len(train_ids) // BATCH_SIZE
valid_steps = len(valid_ids) // BATCH_SIZE

# Build & Compile Model
model = ResUNet(img_h, img_w)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=OPTIMAL_LR), 
    loss='binary_crossentropy', 
    metrics=[dice_coef]
)
model.summary()

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1)
early_stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1)
checkpoint = ModelCheckpoint(
    os.path.join(output_dir, "Custom_ResUNet_best.h5"), 
    monitor="val_loss", 
    save_best_only=True, 
    save_weights_only=True, 
    verbose=1
)

# Train
print("\nStarting final model training...")
history = model.fit(
    train_gen, 
    validation_data=valid_gen,
    steps_per_epoch=train_steps, 
    validation_steps=valid_steps,
    epochs=EPOCHS, 
    callbacks=[checkpoint, reduce_lr, early_stop]
)

# Save Last Epoch
model.save_weights(os.path.join(output_dir, "Custom_ResUNet_last.h5"))
print(f"Last epoch weights saved to {output_dir}")

# ==============================================================================
# EXCEL & PLOTS (Referencing logic from train.py)
# ==============================================================================
hist_dict = history.history
hist_dict['epoch'] = list(range(1, len(hist_dict['loss']) + 1)) 

# Save to Excel
df = pd.DataFrame(hist_dict)
excel_path = os.path.join(output_dir, 'custom_training_metrics.xlsx')
df.to_excel(excel_path, index=False)
print(f'Metrics saved to {excel_path}')

# Plotting
try:
    plt.figure(figsize=(18, 5))
    
    # Plot 1: Loss
    plt.subplot(1, 3, 1)
    plt.plot(hist_dict['epoch'], hist_dict['loss'], label='Train Loss', color='blue')
    plt.plot(hist_dict['epoch'], hist_dict['val_loss'], label='Val Loss', color='orange', linestyle='--')
    plt.title('Loss over Epochs (BCE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot 2: Dice Coefficient
    val_dice_key = 'val_dice_coef' if 'val_dice_coef' in hist_dict else 'val_dice'
    dice_key = 'dice_coef' if 'dice_coef' in hist_dict else 'dice'
    
    plt.subplot(1, 3, 2)
    plt.plot(hist_dict['epoch'], hist_dict.get(val_dice_key, []), label='Val Dice', color='red')
    plt.plot(hist_dict['epoch'], hist_dict.get(dice_key, []), label='Train Dice', color='pink', linestyle='--')
    plt.title('Dice Coefficient over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coeff')
    plt.legend()
    plt.grid(True)

    # Plot 3: Learning Rate
    if 'lr' in hist_dict:
        plt.subplot(1, 3, 3)
        plt.plot(hist_dict['epoch'], hist_dict['lr'], label='Learning Rate', color='green')
        plt.title('Learning Rate over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'custom_training_plot.jpg')
    plt.savefig(plot_path)
    plt.close()
    print(f'Training plot saved to {plot_path}')
except Exception as e:
    print(f"Failed to save plots: {e}")