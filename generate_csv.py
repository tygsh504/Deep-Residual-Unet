import os
import pandas as pd

def generate_train_csv():
    # Define paths based on the required structure
    dataset_dir = "dataset"
    images_dir = os.path.join(dataset_dir, "train", "images")
    masks_dir = os.path.join(dataset_dir, "train", "masks")
    csv_path = os.path.join(dataset_dir, "train.csv")

    # Check if directories exist before running
    if not os.path.exists(images_dir):
        print(f"Error: Could not find the images directory at {images_dir}")
        return
    if not os.path.exists(masks_dir):
        print(f"Error: Could not find the masks directory at {masks_dir}")
        return

    image_ids = []
    
    # Get all files in the images directory
    print(f"Scanning '{images_dir}' for images...")
    for filename in sorted(os.listdir(images_dir)):
        # Check if the file is an image
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            # Extract the ID (filename without the extension)
            file_id = os.path.splitext(filename)[0]
            
            # Verify that a corresponding mask exists
            # Assuming masks have the exact same name and are .png files
            mask_filename = file_id + ".png" 
            mask_path = os.path.join(masks_dir, mask_filename)
            
            if os.path.exists(mask_path):
                image_ids.append(file_id)
            else:
                print(f"  [Warning] Skipping '{filename}': No matching mask found in '{masks_dir}'")

    # Create a Pandas DataFrame
    df = pd.DataFrame({"id": image_ids})

    # Save to CSV without the index column
    df.to_csv(csv_path, index=False)
    
    print("\n--- Success ---")
    print(f"Generated {csv_path} containing {len(image_ids)} image IDs.")

if __name__ == "__main__":
    generate_train_csv()