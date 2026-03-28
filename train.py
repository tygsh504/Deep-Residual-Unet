# ## Imports
# import os
# import random

# import numpy as np
# import pandas as pd
# import cv2
# import matplotlib.pyplot as plt

# import tensorflow as tf
# from tensorflow import keras

# ## Seeding 
# seed = 2019
# random.seed(seed)
# np.random.seed(seed)
# tf.random.set_seed(seed)

# # ======================================================================
# # ENABLE MIXED PRECISION (For RTX 4060)
# # ======================================================================
# policy = keras.mixed_precision.Policy('mixed_float16')
# keras.mixed_precision.set_global_policy(policy)
# print("Compute dtype: %s" % policy.compute_dtype)
# print("Variable dtype: %s" % policy.variable_dtype)

# # ======================================================================
# # CONFIGURATION: DIRECTORIES & PATHS
# # ======================================================================
# train_dir = "dataset/train/"       # Path for Training Dataset
# val_dir = "dataset/val/"           # Path for Validation Dataset
# output_dir = "training_outputs/"   # Path to store weights, excel, and plots

# # Automatically create the output directory if it does not exist
# os.makedirs(output_dir, exist_ok=True)

# # ======================================================================
# # 1. DATA GENERATOR (With Enhanced Augmentation)
# # ======================================================================
# class DataGen(keras.utils.Sequence):
#     def __init__(self, ids, path, batch_size=8, img_w=512, img_h=512, augment=True):
#         self.ids = ids
#         self.path = path
#         self.batch_size = batch_size
#         self.img_w = img_w
#         self.img_h = img_h
#         self.augment = augment
#         self.on_epoch_end()
        
#     def __load__(self, id_name):
#         ## Path (Dynamically check for file extension to prevent crashes)
#         img_base = os.path.join(self.path, "images", id_name)
#         if os.path.exists(img_base + ".png"):
#             image_path = img_base + ".png"
#         elif os.path.exists(img_base + ".jpg"):
#             image_path = img_base + ".jpg"
#         else:
#             image_path = img_base + ".jpeg"
            
#         mask_path = os.path.join(self.path, "masks", id_name) + ".png"
        
#         ## Reading Image & Mask
#         image = cv2.imread(image_path)
#         # FIX: Convert OpenCV's default BGR to RGB
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image = cv2.resize(image, (self.img_w, self.img_h))
        
#         mask = cv2.imread(mask_path, 0)
#         mask = cv2.resize(mask, (self.img_w, self.img_h))
#         mask = np.expand_dims(mask, axis=-1)
        
#         # --- DATA AUGMENTATION ---
#         if self.augment:
#             # 1. Geometric: 50% chance to flip horizontally
#             if random.random() > 0.5:
#                 image = cv2.flip(image, 1)
#                 mask = cv2.flip(mask, 1)
                
#             # 2. Geometric: 50% chance to flip vertically
#             if random.random() > 0.5:
#                 image = cv2.flip(image, 0)
#                 mask = cv2.flip(mask, 0)
                
#             # 3. Rotation: 50% chance to rotate between -30 and 30 degrees
#             if random.random() > 0.5:
#                 angle = random.randint(-30, 30)
#                 M = cv2.getRotationMatrix2D((self.img_w / 2, self.img_h / 2), angle, 1.0)
#                 image = cv2.warpAffine(image, M, (self.img_w, self.img_h))
#                 mask = cv2.warpAffine(mask, M, (self.img_w, self.img_h))
                
#             # 4. Zoom: 50% chance to zoom in slightly
#             if random.random() > 0.5:
#                 zoom = random.uniform(1.0, 1.2)
#                 h, w = image.shape[:2]
#                 new_h, new_w = int(h * zoom), int(w * zoom)
                
#                 # Resize
#                 image = cv2.resize(image, (new_w, new_h))
#                 mask = cv2.resize(mask, (new_w, new_h))
                
#                 # Crop back to original dimensions
#                 start_h = (new_h - h) // 2
#                 start_w = (new_w - w) // 2
#                 image = image[start_h:start_h + h, start_w:start_w + w]
#                 mask = mask[start_h:start_h + h, start_w:start_w + w]

#             # 5. Photometric: 50% chance to adjust brightness and contrast
#             if random.random() > 0.5:
#                 alpha = random.uniform(0.7, 1.3) # Contrast control
#                 beta = random.randint(-30, 30)   # Brightness control
#                 image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
#         # -------------------------
        
#         # Re-apply dimensions for mask if lost during augmentation
#         if len(mask.shape) == 2: mask = np.expand_dims(mask, axis=-1)
        
#         ## FIX: Correct Normalization for Pre-trained ResNet50
#         image = np.array(image, dtype=np.float32)
#         image = keras.applications.resnet50.preprocess_input(image)
        
#         # Mask needs to be 0-1 for Sigmoid output
#         mask = mask / 255.0
        
#         return image, mask
    
#     def __getitem__(self, index):
#         # FIX: Removed the self-destructing batch size logic that was causing iteration drops
#         files_batch = self.ids[index*self.batch_size : (index+1)*self.batch_size]
        
#         image = []
#         mask  = []
        
#         for id_name in files_batch:
#             _img, _mask = self.__load__(id_name)
#             image.append(_img)
#             mask.append(_mask)
            
#         # Cast to float32 for Keras input 
#         image = np.array(image, dtype=np.float32)
#         mask  = np.array(mask, dtype=np.float32)
        
#         return image, mask
    
#     def on_epoch_end(self):
#         # FIX: Shuffle the dataset after every epoch to prevent sequence memorization
#         if self.augment:
#             random.shuffle(self.ids)
    
#     def __len__(self):
#         return int(np.ceil(len(self.ids)/float(self.batch_size)))

# # ======================================================================
# # 2. MODEL ARCHITECTURE & BLOCKS
# # ======================================================================
# def bn_act(x, act=True):
#     x = keras.layers.BatchNormalization()(x)
#     if act == True:
#         x = keras.layers.Activation("relu")(x)
#     return x

# def conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
#     conv = bn_act(x)
#     conv = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
#     return conv

# def residual_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
#     res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    
#     # Increased to 50% Dropout to strongly prevent overfitting
#     res = keras.layers.Dropout(0.5)(res)
    
#     res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)
    
#     shortcut = keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
#     shortcut = bn_act(shortcut, act=False)
    
#     output = keras.layers.Add()([shortcut, res])
#     return output

# def upsample_concat_block(x, xskip):
#     u = keras.layers.UpSampling2D((2, 2))(x)
#     c = keras.layers.Concatenate()([u, xskip])
#     return c

# def Pretrained_ResUNet(img_h, img_w):
#     # 1. Input Layer
#     inputs = keras.layers.Input((img_h, img_w, 3))
    
#     # 2. Load the Pre-trained ResNet50 Encoder
#     encoder = keras.applications.ResNet50(weights="imagenet", include_top=False, input_tensor=inputs)
    
#     # 3. Extract Skip Connections from the pre-trained encoder
#     s1 = encoder.input                                   
#     s2 = encoder.get_layer("conv1_relu").output          
#     s3 = encoder.get_layer("conv2_block3_out").output    
#     s4 = encoder.get_layer("conv3_block4_out").output    
    
#     # 4. Bridge
#     b1 = encoder.get_layer("conv4_block6_out").output    
    
#     # 5. Decoder (Using existing custom residual blocks)
#     f = [16, 32, 64, 128, 256]
    
#     u1 = upsample_concat_block(b1, s4)
#     d1 = residual_block(u1, f[3])
    
#     u2 = upsample_concat_block(d1, s3)
#     d2 = residual_block(u2, f[2])
    
#     u3 = upsample_concat_block(d2, s2)
#     d3 = residual_block(u3, f[1])
    
#     u4 = upsample_concat_block(d3, s1)
#     d4 = residual_block(u4, f[0])
    
#     # Final classification layer - MUST be dtype='float32' for numerical stability in mixed precision
#     outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid", dtype='float32')(d4)
    
#     model = keras.models.Model(inputs, outputs)
    
#     return model

# # ======================================================================
# # 3. METRICS
# # ======================================================================
# smooth = 1.

# def dice_coef(y_true, y_pred):
#     y_true_f = tf.reshape(y_true, [-1])
#     y_pred_f = tf.reshape(y_pred, [-1])
    
#     # Ensure float32 for metric calculation
#     y_true_f = tf.cast(y_true_f, tf.float32)
#     y_pred_f = tf.cast(y_pred_f, tf.float32)
    
#     intersection = tf.reduce_sum(y_true_f * y_pred_f)
#     return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

# def dice_coef_loss(y_true, y_pred):
#     return 1.0 - dice_coef(y_true, y_pred)


# # ======================================================================
# # 4. TRAINING SETUP 
# # ======================================================================

# # Function to dynamically fetch IDs from a directory
# def get_file_ids(directory):
#     img_folder = os.path.join(directory, "images")
#     if not os.path.exists(img_folder):
#         raise ValueError(f"Could not find directory: {img_folder}")
#     return [os.path.splitext(f)[0] for f in os.listdir(img_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# # Load IDs
# train_ids = get_file_ids(train_dir)
# valid_ids = get_file_ids(val_dir)
# print(f"Found {len(train_ids)} training images and {len(valid_ids)} validation images.")

# # Set standard U-Net compatible dimensions
# img_w = 480
# img_h = 640
# batch_size = 8    
# epochs = 100

# # FIX: Set augment=True for the training generator
# train_gen = DataGen(train_ids, train_dir, img_w=img_w, img_h=img_h, batch_size=batch_size, augment=True)
# valid_gen = DataGen(valid_ids, val_dir, img_w=img_w, img_h=img_h, batch_size=batch_size, augment=False)

# train_steps = len(train_ids) // batch_size
# valid_steps = len(valid_ids) // batch_size

# # 1. Initialize Model
# model = Pretrained_ResUNet(img_h, img_w)

# # 2. Compile with an even lower learning rate (5e-5) to protect pre-trained weights
# model.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-5), loss=dice_coef_loss, metrics=[dice_coef])

# # 3. Callbacks
# checkpoint = keras.callbacks.ModelCheckpoint(
#     os.path.join(output_dir, "ResUNet_best.h5"), 
#     monitor="val_loss", 
#     verbose=1, 
#     save_best_only=True, 
#     save_weights_only=True, 
#     mode="min"
# )

# reduce_lr = keras.callbacks.ReduceLROnPlateau(
#     monitor="val_loss", factor=0.5, patience=6, min_lr=1e-7, verbose=1
# )

# # 4. Train the entire network end-to-end
# history = model.fit(
#     train_gen, 
#     validation_data=valid_gen, 
#     steps_per_epoch=train_steps, 
#     validation_steps=valid_steps, 
#     epochs=epochs, 
#     callbacks=[checkpoint, reduce_lr]
# )

# # Save Last Weights
# model.save_weights(os.path.join(output_dir, "ResUNet_last.h5"))

# # ======================================================================
# # 5. POST-TRAINING: EXCEL & PLOTS
# # ======================================================================
# # Extract metrics from Keras history
# hist_dict = history.history
# hist_dict['epoch'] = list(range(1, len(hist_dict['loss']) + 1)) 

# # Save to Excel
# df = pd.DataFrame(hist_dict)
# excel_path = os.path.join(output_dir, 'training_metrics.xlsx')
# df.to_excel(excel_path, index=False)
# print(f'Metrics saved to {excel_path}')

# # Plotting
# try:
#     plt.figure(figsize=(18, 5))
    
#     # Plot 1: Training and Validation Loss
#     plt.subplot(1, 3, 1)
#     plt.plot(hist_dict['epoch'], hist_dict['loss'], label='Train Loss', color='blue')
#     plt.plot(hist_dict['epoch'], hist_dict['val_loss'], label='Val Loss', color='orange', linestyle='--')
#     plt.title('Loss over Epochs')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss (Dice Loss)')
#     plt.legend()
#     plt.grid(True)

#     # Plot 2: Training and Validation Dice Coefficient
#     val_dice_key = 'val_dice_coef' if 'val_dice_coef' in hist_dict else 'val_dice'
#     dice_key = 'dice_coef' if 'dice_coef' in hist_dict else 'dice'
    
#     plt.subplot(1, 3, 2)
#     plt.plot(hist_dict['epoch'], hist_dict.get(val_dice_key, []), label='Val Dice', color='red')
#     plt.plot(hist_dict['epoch'], hist_dict.get(dice_key, []), label='Train Dice', color='pink', linestyle='--')
#     plt.title('Dice Coefficient over Epochs')
#     plt.xlabel('Epoch')
#     plt.ylabel('Dice Coeff')
#     plt.legend()
#     plt.grid(True)

#     # Plot 3: Learning Rate
#     if 'lr' in hist_dict:
#         plt.subplot(1, 3, 3)
#         plt.plot(hist_dict['epoch'], hist_dict['lr'], label='Learning Rate', color='green')
#         plt.title('Learning Rate over Epochs')
#         plt.xlabel('Epoch')
#         plt.ylabel('Learning Rate')
#         plt.legend()
#         plt.grid(True)

#     plt.tight_layout()
#     plot_path = os.path.join(output_dir, 'training_plot.jpg')
#     plt.savefig(plot_path)
#     plt.close()
#     print(f'Training plot saved to {plot_path}')
# except Exception as e:
#     print(f"Failed to save plots: {e}")

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
# ENABLE MIXED PRECISION (For RTX 4060)
# ======================================================================
policy = keras.mixed_precision.Policy('mixed_float16')
keras.mixed_precision.set_global_policy(policy)
print("Compute dtype: %s" % policy.compute_dtype)
print("Variable dtype: %s" % policy.variable_dtype)

# ======================================================================
# CONFIGURATION: DIRECTORIES & PATHS
# ======================================================================
train_dir = "dataset/train/"       # Path for Training Dataset
val_dir = "dataset/val/"           # Path for Validation Dataset
output_dir = "training_outputs/"   # Path to store weights, excel, and plots

# Automatically create the output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# ======================================================================
# 1. DATA GENERATOR (With Random Crop & Patch)
# ======================================================================
class DataGen(keras.utils.Sequence):
    def __init__(self, ids, path, batch_size=4, orig_w=640, orig_h=480, patch_size=256, augment=True):
        self.ids = ids
        self.path = path
        self.batch_size = batch_size
        self.orig_w = orig_w
        self.orig_h = orig_h
        self.patch_size = patch_size
        self.augment = augment
        self.on_epoch_end()
        
    def __load__(self, id_name):
        ## Path (Dynamically check for file extension)
        img_base = os.path.join(self.path, "images", id_name)
        if os.path.exists(img_base + ".png"):
            image_path = img_base + ".png"
        elif os.path.exists(img_base + ".jpg"):
            image_path = img_base + ".jpg"
        else:
            image_path = img_base + ".jpeg"
            
        mask_path = os.path.join(self.path, "masks", id_name) + ".png"
        
        ## Reading Image & Mask
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.orig_w, self.orig_h))
        
        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (self.orig_w, self.orig_h))
        mask = np.expand_dims(mask, axis=-1)
        
        # --- DATA AUGMENTATION (Applied to full image before patching) ---
        if self.augment:
            if random.random() > 0.5:
                image = cv2.flip(image, 1)
                mask = cv2.flip(mask, 1)
                
            if random.random() > 0.5:
                image = cv2.flip(image, 0)
                mask = cv2.flip(mask, 0)
                
            if random.random() > 0.5:
                angle = random.randint(-30, 30)
                M = cv2.getRotationMatrix2D((self.orig_w / 2, self.orig_h / 2), angle, 1.0)
                image = cv2.warpAffine(image, M, (self.orig_w, self.orig_h))
                mask = cv2.warpAffine(mask, M, (self.orig_w, self.orig_h))
                
            if random.random() > 0.5:
                zoom = random.uniform(1.0, 1.2)
                h, w = image.shape[:2]
                new_h, new_w = int(h * zoom), int(w * zoom)
                
                image = cv2.resize(image, (new_w, new_h))
                mask = cv2.resize(mask, (new_w, new_h))
                
                start_h = (new_h - h) // 2
                start_w = (new_w - w) // 2
                image = image[start_h:start_h + h, start_w:start_w + w]
                mask = mask[start_h:start_h + h, start_w:start_w + w]

            if random.random() > 0.5:
                alpha = random.uniform(0.7, 1.3) 
                beta = random.randint(-30, 30)   
                image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        # -----------------------------------------------------------------
        
        if len(mask.shape) == 2: mask = np.expand_dims(mask, axis=-1)
        
        ## Normalization
        image = np.array(image, dtype=np.float32)
        image = keras.applications.resnet50.preprocess_input(image)
        mask = mask / 255.0
        
        return image, mask

    def get_random_patches(self, img_array, mask_array, num_patches=4):
        """Extracts 4 random 256x256 patches from the arrays to prevent OOM."""
        img_patches = []
        mask_patches = []
        
        max_y = img_array.shape[0] - self.patch_size
        max_x = img_array.shape[1] - self.patch_size
        
        for _ in range(num_patches):
            y = random.randint(0, max_y)
            x = random.randint(0, max_x)
            
            img_patches.append(img_array[y:y+self.patch_size, x:x+self.patch_size])
            mask_patches.append(mask_array[y:y+self.patch_size, x:x+self.patch_size])
            
        return img_patches, mask_patches
    
    def __getitem__(self, index):
        files_batch = self.ids[index*self.batch_size : (index+1)*self.batch_size]
        
        image_batch = []
        mask_batch  = []
        
        for id_name in files_batch:
            _img, _mask = self.__load__(id_name)
            
            # Extract 4 random patches per image
            img_patches, mask_patches = self.get_random_patches(_img, _mask)
            
            image_batch.extend(img_patches)
            mask_batch.extend(mask_patches)
            
        # Cast to float32 for Keras input 
        image_batch = np.array(image_batch, dtype=np.float32)
        mask_batch  = np.array(mask_batch, dtype=np.float32)
        
        return image_batch, mask_batch
    
    def on_epoch_end(self):
        if self.augment:
            random.shuffle(self.ids)
    
    def __len__(self):
        return int(np.ceil(len(self.ids)/float(self.batch_size)))

# ======================================================================
# 2. MODEL ARCHITECTURE & BLOCKS
# ======================================================================
def bn_act(x, act=True):
    x = keras.layers.BatchNormalization()(x)
    if act == True:
        x = keras.layers.Activation("relu")(x)
    return x

def conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = bn_act(x)
    conv = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
    return conv

def residual_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    res = keras.layers.Dropout(0.5)(res)
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
    inputs = keras.layers.Input((img_h, img_w, 3))
    
    encoder = keras.applications.ResNet50(weights="imagenet", include_top=False, input_tensor=inputs)
    
    s1 = encoder.input                                   
    s2 = encoder.get_layer("conv1_relu").output          
    s3 = encoder.get_layer("conv2_block3_out").output    
    s4 = encoder.get_layer("conv3_block4_out").output    
    
    b1 = encoder.get_layer("conv4_block6_out").output    
    
    f = [16, 32, 64, 128, 256]
    
    u1 = upsample_concat_block(b1, s4)
    d1 = residual_block(u1, f[3])
    
    u2 = upsample_concat_block(d1, s3)
    d2 = residual_block(u2, f[2])
    
    u3 = upsample_concat_block(d2, s2)
    d3 = residual_block(u3, f[1])
    
    u4 = upsample_concat_block(d3, s1)
    d4 = residual_block(u4, f[0])
    
    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid", dtype='float32')(d4)
    
    model = keras.models.Model(inputs, outputs)
    
    return model

# ======================================================================
# 3. METRICS
# ======================================================================
smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    
    y_true_f = tf.cast(y_true_f, tf.float32)
    y_pred_f = tf.cast(y_pred_f, tf.float32)
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


# ======================================================================
# 4. TRAINING SETUP 
# ======================================================================

def get_file_ids(directory):
    img_folder = os.path.join(directory, "images")
    if not os.path.exists(img_folder):
        raise ValueError(f"Could not find directory: {img_folder}")
    return [os.path.splitext(f)[0] for f in os.listdir(img_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

train_ids = get_file_ids(train_dir)
valid_ids = get_file_ids(val_dir)
print(f"Found {len(train_ids)} training images and {len(valid_ids)} validation images.")

# Config
orig_w = 640
orig_h = 480
patch_size = 256
batch_size = 4 
epochs = 100

train_gen = DataGen(train_ids, train_dir, orig_w=orig_w, orig_h=orig_h, patch_size=patch_size, batch_size=batch_size, augment=True)
valid_gen = DataGen(valid_ids, val_dir, orig_w=orig_w, orig_h=orig_h, patch_size=patch_size, batch_size=batch_size, augment=False)

train_steps = len(train_ids) // batch_size
valid_steps = len(valid_ids) // batch_size

# 1. Initialize Model (Targeting Patch Size Dimensions)
model = Pretrained_ResUNet(patch_size, patch_size)

# 2. Compile
model.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-5), loss=dice_coef_loss, metrics=[dice_coef])

# 3. Callbacks
checkpoint = keras.callbacks.ModelCheckpoint(
    os.path.join(output_dir, "ResUNet_best.h5"), 
    monitor="val_loss", 
    verbose=1, 
    save_best_only=True, 
    save_weights_only=True, 
    mode="min"
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=6, min_lr=1e-7, verbose=1
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=15,               
    restore_best_weights=True, 
    verbose=1
)

# 4. Train
history = model.fit(
    train_gen, 
    validation_data=valid_gen, 
    steps_per_epoch=train_steps, 
    validation_steps=valid_steps, 
    epochs=epochs, 
    callbacks=[checkpoint, reduce_lr, early_stopping]
)

# Save Last Weights
model.save_weights(os.path.join(output_dir, "ResUNet_last.h5"))

# ======================================================================
# 5. POST-TRAINING: EXCEL & PLOTS
# ======================================================================
hist_dict = history.history
hist_dict['epoch'] = list(range(1, len(hist_dict['loss']) + 1)) 

df = pd.DataFrame(hist_dict)
excel_path = os.path.join(output_dir, 'training_metrics.xlsx')
df.to_excel(excel_path, index=False)
print(f'Metrics saved to {excel_path}')

try:
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(hist_dict['epoch'], hist_dict['loss'], label='Train Loss', color='blue')
    plt.plot(hist_dict['epoch'], hist_dict['val_loss'], label='Val Loss', color='orange', linestyle='--')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Dice Loss)')
    plt.legend()
    plt.grid(True)

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

    if 'lr' in hist_dict:
        plt.subplot(1, 3, 3)
        plt.plot(hist_dict['epoch'], hist_dict['lr'], label='Learning Rate', color='green')
        plt.title('Learning Rate over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_plot.jpg')
    plt.savefig(plot_path)
    plt.close()
    print(f'Training plot saved to {plot_path}')
except Exception as e:
    print(f"Failed to save plots: {e}")