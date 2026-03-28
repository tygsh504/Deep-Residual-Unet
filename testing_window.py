# import os
# import logging
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import cv2
# from tqdm import tqdm
# from pathlib import Path

# import tensorflow as tf
# from tensorflow import keras

# # ======================================================================
# # ENABLE MIXED PRECISION (Must match train.py)
# # ======================================================================
# policy = keras.mixed_precision.Policy('mixed_float16')
# keras.mixed_precision.set_global_policy(policy)

# # --- USER CONFIGURATION SECTION ---
# MODEL_PATH = 'training_outputs/ResUNet_best.h5' 
# BASE_DATA_PATH = r"C:\Users\User\Desktop\Paddy_Dataset"
# MAIN_OUTPUT_DIR = r"C:\Users\User\Desktop\ResUNet_Test_Results_4"

# # The 7 disease folders
# DISEASES = ["Bacterial Leaf Blight", "Bacterial Leaf Streak", "Blast", "Brown Spot", "DownyMildew", "Hispa", "Tungro"]

# # FIX: Aligned with train.py orig_h=480, orig_w=640
# INPUT_SHAPE = [480, 640] # [Height, Width] 
# PATCH_SIZE = 256 # Matching the patch size from training
# # ----------------------------------

# # ==========================================
# # 1. DEEP RESIDUAL UNET ARCHITECTURE (Matched with train.py)
# # ==========================================
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
#     inputs = keras.layers.Input((img_h, img_w, 3))
    
#     encoder = keras.applications.ResNet50(weights=None, include_top=False, input_tensor=inputs)
    
#     s1 = encoder.input                                   
#     s2 = encoder.get_layer("conv1_relu").output          
#     s3 = encoder.get_layer("conv2_block3_out").output    
#     s4 = encoder.get_layer("conv3_block4_out").output    
    
#     b1 = encoder.get_layer("conv4_block6_out").output    
    
#     f = [16, 32, 64, 128, 256]
    
#     u1 = upsample_concat_block(b1, s4)
#     d1 = residual_block(u1, f[3])
    
#     u2 = upsample_concat_block(d1, s3)
#     d2 = residual_block(u2, f[2])
    
#     u3 = upsample_concat_block(d2, s2)
#     d3 = residual_block(u3, f[1])
    
#     u4 = upsample_concat_block(d3, s1)
#     d4 = residual_block(u4, f[0])
    
#     outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid", dtype='float32')(d4)
    
#     model = keras.models.Model(inputs, outputs)
#     return model

# # ==========================================
# # 2. METRICS & VISUALIZATION
# # ==========================================
# def calculate_complexity(model):
#     """Calculates model complexity."""
#     params = model.count_params()
#     flops = 0 
#     return params, flops

# def calculate_metrics(pred_np, true_np):
#     pred_bin = (pred_np.flatten() > 0.5).astype(np.uint8)
#     true_bin = (true_np.flatten() > 0.5).astype(np.uint8)

#     tp = np.sum((pred_bin == 1) & (true_bin == 1))
#     tn = np.sum((pred_bin == 0) & (true_bin == 0))
#     fp = np.sum((pred_bin == 1) & (true_bin == 0))
#     fn = np.sum((pred_bin == 0) & (true_bin == 1))

#     epsilon = 1e-7
#     accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
#     precision = tp / (tp + fp + epsilon)
#     recall = tp / (tp + fn + epsilon)
#     f1 = 2 * (precision * recall) / (precision + recall + epsilon)
#     iou = tp / (tp + fp + fn + epsilon)
#     dice = (2 * tp) / (2 * tp + fp + fn + epsilon)

#     return {"Dice": dice, "IoU": iou, "Precision": precision, "Recall": recall, "Accuracy": accuracy, "F1_Score": f1}

# def save_visual_result(image_plot, true_np, pred_np, filename, dice_score, output_dir):
#     true_bin = (true_np > 0.5).astype(np.uint8)
#     pred_bin = (pred_np > 0.5).astype(np.uint8)

#     fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
#     ax[0].imshow(image_plot); ax[0].set_title(f"Original: {filename}"); ax[0].axis("off")
#     ax[1].imshow(true_bin, cmap='gray'); ax[1].set_title("Ground Truth"); ax[1].axis("off")
#     ax[2].imshow(pred_bin, cmap='gray'); ax[2].set_title(f"Pred (Dice: {dice_score:.2f})"); ax[2].axis("off")

#     plt.tight_layout()
#     plt.savefig(output_dir / f"{filename}_eval.png", dpi=150)
#     plt.close(fig)

# # ==========================================
# # 3. SLIDING WINDOW INFERENCE LOGIC
# # ==========================================
# def sliding_window_inference(image, model, patch_size=(256, 256), stride=128):
#     """Predicts a large image seamlessly using patches to match training dimensions."""
#     img_h, img_w, _ = image.shape
#     patch_h, patch_w = patch_size
    
#     prob_map = np.zeros((img_h, img_w, 1), dtype=np.float32)
#     count_map = np.zeros((img_h, img_w, 1), dtype=np.float32)
    
#     for y in range(0, img_h - patch_h + stride, stride):
#         for x in range(0, img_w - patch_w + stride, stride):
#             y_start = min(y, img_h - patch_h)
#             x_start = min(x, img_w - patch_w)
#             y_end = y_start + patch_h
#             x_end = x_start + patch_w
            
#             patch = image[y_start:y_end, x_start:x_end, :]
#             patch_input = np.expand_dims(patch, axis=0)
            
#             patch_pred = model.predict(patch_input, verbose=0)[0]
            
#             prob_map[y_start:y_end, x_start:x_end] += patch_pred
#             count_map[y_start:y_end, x_start:x_end] += 1

#     final_prob_map = prob_map / count_map
#     return final_prob_map[:, :, 0] # Return as 2D array


# # ==========================================
# # 4. TESTING LOGIC
# # ==========================================
# def run_test_on_disease(disease_name, net, params, flops):
#     img_dir = os.path.join(BASE_DATA_PATH, disease_name, "Infer_Ori")
#     mask_dir = os.path.join(BASE_DATA_PATH, disease_name, "Infer_GT")
    
#     if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
#         logging.warning(f"Skipping {disease_name}: Path not found.")
#         return None

#     disease_output_dir = Path(MAIN_OUTPUT_DIR) / disease_name
#     img_output_dir = disease_output_dir / "predictions"
#     disease_output_dir.mkdir(parents=True, exist_ok=True)
#     img_output_dir.mkdir(parents=True, exist_ok=True)

#     img_h, img_w = INPUT_SHAPE[0], INPUT_SHAPE[1]
#     valid_extensions = ('.jpg', '.png', '.jpeg')
#     image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(valid_extensions)]

#     results = []
    
#     for filename in tqdm(image_files, desc=f"Testing {disease_name}"):
#         img_path = os.path.join(img_dir, filename)
        
#         mask_name = filename if os.path.exists(os.path.join(mask_dir, filename)) else os.path.splitext(filename)[0] + '.png'
#         mask_path = os.path.join(mask_dir, mask_name)

#         if not os.path.exists(mask_path):
#             logging.warning(f"Missing mask for {filename}, skipping.")
#             continue

#         image_bgr = cv2.imread(img_path)
#         image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#         image_resized = cv2.resize(image_rgb, (img_w, img_h)) 
        
#         image_plot = image_resized / 255.0
        
#         image_norm = np.array(image_resized, dtype=np.float32)
#         image_norm = keras.applications.resnet50.preprocess_input(image_norm)
        
#         mask = cv2.imread(mask_path, 0)
#         mask = cv2.resize(mask, (img_w, img_h))
#         mask_norm = (mask > 0).astype(np.float32) 

#         # --- FIX: Replaced direct prediction with sliding window inference ---
#         pred_mask = sliding_window_inference(image_norm, net, patch_size=(PATCH_SIZE, PATCH_SIZE), stride=128)
#         # ---------------------------------------------------------------------

#         metrics = calculate_metrics(pred_mask, mask_norm)
#         metrics['Filename'] = filename
#         results.append(metrics)
        
#         save_visual_result(image_plot, mask_norm, pred_mask, filename, metrics['Dice'], img_output_dir)

#     if results:
#         df = pd.DataFrame(results)
#         metric_cols = ['Dice', 'IoU', 'Precision', 'Recall', 'Accuracy', 'F1_Score']
#         means = df[metric_cols].mean().to_dict()
#         summary_df = pd.DataFrame([{'Metric': k, 'Value': v} for k, v in means.items()] + 
#                                   [{'Metric': 'Params', 'Value': params}, {'Metric': 'FLOPs', 'Value': flops}])

#         excel_path = disease_output_dir / f'{disease_name}_metrics.xlsx'
#         with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
#             summary_df.to_excel(writer, sheet_name='Summary', index=False)
#             df[['Filename'] + metric_cols].to_excel(writer, sheet_name='Detailed', index=False)
            
#         return means
#     return None

# if __name__ == '__main__':
#     logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

#     try:
#         logging.info("Building Deep Residual UNet architecture...")
#         # FIX: Instantiate the network using the dimensions of the patches it was trained on
#         net = Pretrained_ResUNet(img_h=PATCH_SIZE, img_w=PATCH_SIZE)
        
#         logging.info(f"Loading weights from {MODEL_PATH}...")
#         net.load_weights(MODEL_PATH)
#         logging.info("Model loaded successfully.")
#     except Exception as e:
#         logging.error(f"Failed to load model: {e}")
#         exit(1)

#     params, flops = calculate_complexity(net)

#     all_disease_results = []

#     for disease in DISEASES:
#         disease_means = run_test_on_disease(disease, net, params, flops)
#         if disease_means:
#             disease_means['Disease'] = disease
#             all_disease_results.append(disease_means)

#     if all_disease_results:
#         overall_df = pd.DataFrame(all_disease_results)
        
#         cols = ['Disease'] + [c for c in overall_df.columns if c != 'Disease']
#         overall_df = overall_df[cols]
        
#         mean_row = overall_df.mean(numeric_only=True).to_dict()
#         mean_row['Disease'] = 'OVERALL_MEAN'
        
#         overall_df = pd.concat([overall_df, pd.DataFrame([mean_row])], ignore_index=True)
        
#         mean_output_path = Path(MAIN_OUTPUT_DIR) / 'calculated_mean.xlsx'
#         overall_df.to_excel(mean_output_path, index=False)
#         logging.info(f"Overall means saved to {mean_output_path}")

#     print("\n--- All Testing Completed ---")

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

# ======================================================================
# ENABLE MIXED PRECISION (Must match train.py)
# ======================================================================
policy = keras.mixed_precision.Policy('mixed_float16')
keras.mixed_precision.set_global_policy(policy)

# --- USER CONFIGURATION SECTION ---
MODEL_PATH = 'training_outputs/ResUNet_best.h5' 
BASE_DATA_PATH = r"C:\Users\User\Desktop\Paddy_Dataset"
MAIN_OUTPUT_DIR = r"C:\Users\User\Desktop\ResUNet_Test_Results_4"

# The 7 disease folders
DISEASES = ["Bacterial Leaf Blight", "Bacterial Leaf Streak", "Blast", "Brown Spot", "DownyMildew", "Hispa", "Tungro"]

INPUT_SHAPE = [480, 640] # [Height, Width] 
PATCH_SIZE = 256 # Matching the patch size from training
# ----------------------------------

# ==========================================
# 1. DEEP RESIDUAL UNET ARCHITECTURE 
# ==========================================
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
    
    encoder = keras.applications.ResNet50(weights=None, include_top=False, input_tensor=inputs)
    
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

# ==========================================
# 2. METRICS & VISUALIZATION
# ==========================================
def calculate_complexity(model):
    params = model.count_params()
    flops = 0 
    return params, flops

def calculate_metrics(pred_np, true_np):
    pred_bin = (pred_np.flatten() > 0.5).astype(np.uint8)
    true_bin = (true_np.flatten() > 0.5).astype(np.uint8)

    tp = np.sum((pred_bin == 1) & (true_bin == 1))
    tn = np.sum((pred_bin == 0) & (true_bin == 0))
    fp = np.sum((pred_bin == 1) & (true_bin == 0))
    fn = np.sum((pred_bin == 0) & (true_bin == 1))

    epsilon = 1e-7
    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    iou = tp / (tp + fp + fn + epsilon)
    dice = (2 * tp) / (2 * tp + fp + fn + epsilon)

    return {"Dice": dice, "IoU": iou, "Precision": precision, "Recall": recall, "Accuracy": accuracy, "F1_Score": f1}

def save_visual_result(image_plot, true_np, pred_np, filename, dice_score, output_dir):
    true_bin = (true_np > 0.5).astype(np.uint8)
    pred_bin = (pred_np > 0.5).astype(np.uint8)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    ax[0].imshow(image_plot); ax[0].set_title(f"Original: {filename}"); ax[0].axis("off")
    ax[1].imshow(true_bin, cmap='gray'); ax[1].set_title("Ground Truth"); ax[1].axis("off")
    ax[2].imshow(pred_bin, cmap='gray'); ax[2].set_title(f"Pred (Dice: {dice_score:.2f})"); ax[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / f"{filename}_eval.png", dpi=150)
    plt.close(fig)

# ==========================================
# 3. SLIDING WINDOW INFERENCE LOGIC 
# ==========================================

def get_gaussian_map(patch_h, patch_w, sigma=0.5):
    """Generates a 2D Gaussian map to smoothly blend overlapping predictions."""
    x = np.linspace(-1, 1, patch_w)
    y = np.linspace(-1, 1, patch_h)
    x, y = np.meshgrid(x, y)
    d = np.sqrt(x*x + y*y)
    g = np.exp(-( (d)**2 / ( 2.0 * sigma**2 ) ) )
    return np.expand_dims(g, axis=-1).astype(np.float32)

def sliding_window_inference(image, model, patch_size=(256, 256), stride=128):
    """Predicts a large image using Gaussian-blended overlapping patches."""
    img_h, img_w, _ = image.shape
    patch_h, patch_w = patch_size
    
    prob_map = np.zeros((img_h, img_w, 1), dtype=np.float32)
    weight_map = np.zeros((img_h, img_w, 1), dtype=np.float32)
    
    # Generate the Gaussian weight matrix once
    gaussian_weight = get_gaussian_map(patch_h, patch_w)
    
    for y in range(0, img_h - patch_h + stride, stride):
        for x in range(0, img_w - patch_w + stride, stride):
            y_start = min(y, img_h - patch_h)
            x_start = min(x, img_w - patch_w)
            y_end = y_start + patch_h
            x_end = x_start + patch_w
            
            patch = image[y_start:y_end, x_start:x_end, :]
            patch_input = np.expand_dims(patch, axis=0)
            
            patch_pred = model.predict(patch_input, verbose=0)[0]
            
            # Slice the gaussian map in case min() altered the dimensions (failsafe)
            h_slice = y_end - y_start
            w_slice = x_end - x_start
            gw = gaussian_weight[:h_slice, :w_slice, :]
            
            # Apply Gaussian weighting
            prob_map[y_start:y_end, x_start:x_end] += patch_pred * gw
            weight_map[y_start:y_end, x_start:x_end] += gw

    # Normalize by accumulated weights rather than simple counts
    final_prob_map = prob_map / (weight_map + 1e-7)
    return final_prob_map[:, :, 0]


# ==========================================
# 4. TESTING LOGIC
# ==========================================
def run_test_on_disease(disease_name, net, params, flops):
    img_dir = os.path.join(BASE_DATA_PATH, disease_name, "Infer_Ori")
    mask_dir = os.path.join(BASE_DATA_PATH, disease_name, "Infer_GT")
    
    if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
        logging.warning(f"Skipping {disease_name}: Path not found.")
        return None

    disease_output_dir = Path(MAIN_OUTPUT_DIR) / disease_name
    img_output_dir = disease_output_dir / "predictions"
    disease_output_dir.mkdir(parents=True, exist_ok=True)
    img_output_dir.mkdir(parents=True, exist_ok=True)

    img_h, img_w = INPUT_SHAPE[0], INPUT_SHAPE[1]
    valid_extensions = ('.jpg', '.png', '.jpeg')
    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(valid_extensions)]

    results = []
    
    for filename in tqdm(image_files, desc=f"Testing {disease_name}"):
        img_path = os.path.join(img_dir, filename)
        
        mask_name = filename if os.path.exists(os.path.join(mask_dir, filename)) else os.path.splitext(filename)[0] + '.png'
        mask_path = os.path.join(mask_dir, mask_name)

        if not os.path.exists(mask_path):
            logging.warning(f"Missing mask for {filename}, skipping.")
            continue

        image_bgr = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (img_w, img_h)) 
        
        image_plot = image_resized / 255.0
        
        image_norm = np.array(image_resized, dtype=np.float32)
        image_norm = keras.applications.resnet50.preprocess_input(image_norm)
        
        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (img_w, img_h))
        mask_norm = (mask > 0).astype(np.float32) 

        # Uses the new Gaussian sliding window algorithm
        pred_mask = sliding_window_inference(image_norm, net, patch_size=(PATCH_SIZE, PATCH_SIZE), stride=128)

        metrics = calculate_metrics(pred_mask, mask_norm)
        metrics['Filename'] = filename
        results.append(metrics)
        
        save_visual_result(image_plot, mask_norm, pred_mask, filename, metrics['Dice'], img_output_dir)

    if results:
        df = pd.DataFrame(results)
        metric_cols = ['Dice', 'IoU', 'Precision', 'Recall', 'Accuracy', 'F1_Score']
        means = df[metric_cols].mean().to_dict()
        summary_df = pd.DataFrame([{'Metric': k, 'Value': v} for k, v in means.items()] + 
                                  [{'Metric': 'Params', 'Value': params}, {'Metric': 'FLOPs', 'Value': flops}])

        excel_path = disease_output_dir / f'{disease_name}_metrics.xlsx'
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            df[['Filename'] + metric_cols].to_excel(writer, sheet_name='Detailed', index=False)
            
        return means
    return None

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    try:
        logging.info("Building Deep Residual UNet architecture...")
        net = Pretrained_ResUNet(img_h=PATCH_SIZE, img_w=PATCH_SIZE)
        
        logging.info(f"Loading weights from {MODEL_PATH}...")
        net.load_weights(MODEL_PATH)
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        exit(1)

    params, flops = calculate_complexity(net)

    all_disease_results = []

    for disease in DISEASES:
        disease_means = run_test_on_disease(disease, net, params, flops)
        if disease_means:
            disease_means['Disease'] = disease
            all_disease_results.append(disease_means)

    if all_disease_results:
        overall_df = pd.DataFrame(all_disease_results)
        
        cols = ['Disease'] + [c for c in overall_df.columns if c != 'Disease']
        overall_df = overall_df[cols]
        
        mean_row = overall_df.mean(numeric_only=True).to_dict()
        mean_row['Disease'] = 'OVERALL_MEAN'
        
        overall_df = pd.concat([overall_df, pd.DataFrame([mean_row])], ignore_index=True)
        
        mean_output_path = Path(MAIN_OUTPUT_DIR) / 'calculated_mean.xlsx'
        overall_df.to_excel(mean_output_path, index=False)
        logging.info(f"Overall means saved to {mean_output_path}")

    print("\n--- All Testing Completed ---")