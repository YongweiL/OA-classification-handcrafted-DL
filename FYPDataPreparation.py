import os
import cv2
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

# ======================== Configuration ========================
INPUT_DIR = r"C:\UTM Degree\y4s2\PSM1_Dr Nies\KOA\data\ori data\FYP data\selected_clear_images"
IMAGE_SIZE = (224, 224)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.2
SEED = 42
# Output paths
handcrafted_output_base = r'C:\UTM Degree\y4s2\PSM1_Dr Nies\KOA\data\ori data\FYP data\hand\augSplitAfter'
cnn_output_base = r'C:\UTM Degree\y4s2\PSM1_Dr Nies\KOA\data\ori data\FYP data\cnn\augSplitAfter'

# ==================== Preprocessing Functions ====================
# CNN preprocessing: CLAHE + Avg Filter
def preprocess_cnn(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    crop = img[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, IMAGE_SIZE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray_resized)
    avg_filtered = cv2.blur(clahe_img, (3, 3))
    final_img = cv2.cvtColor(avg_filtered, cv2.COLOR_GRAY2BGR)
    return final_img

# Handcrafted preprocessing: ROI + basic threshold segmentation
def preprocess_handcrafted(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    crop = img[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, IMAGE_SIZE)
    _, segmented = cv2.threshold(gray_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    final_img = cv2.cvtColor(segmented, cv2.COLOR_GRAY2BGR)
    return final_img

# ==================== Data Split Functions ====================
def load_images_by_class():
    image_info = []
    for label in ['0', '1', '2', '3', '4']:
        paths = glob(os.path.join(INPUT_DIR, label, "*.jpg")) + glob(os.path.join(INPUT_DIR, label, "*.png"))
        for p in paths:
            image_info.append((p, label))
    return image_info

def split_data(image_info):
    train_val, test = train_test_split(image_info, test_size=(1 - TRAIN_RATIO),
                                       stratify=[lbl for _, lbl in image_info], random_state=SEED)
    train, val = train_test_split(train_val, test_size=VAL_RATIO,
                                  stratify=[lbl for _, lbl in train_val], random_state=SEED)
    return train, val, test

# ==================== Augmentation ====================
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

def augment_and_save(img_array, label, img_index, output_dir, csv_records, augment_count=2, all_images_dir=None):
    img_array = img_array.reshape((1,) + img_array.shape)
    for i, batch in enumerate(datagen.flow(img_array, batch_size=1)):
        aug_filename = f"aug_{label}_{img_index}_{i}.jpg"
        aug_path = os.path.join(output_dir, aug_filename)
        array_to_img(batch[0]).save(aug_path)
        csv_records.append((aug_filename, label, "train"))

        if all_images_dir:
            os.makedirs(all_images_dir, exist_ok=True)
            array_to_img(batch[0]).save(os.path.join(all_images_dir, aug_filename))

        if i + 1 >= augment_count:
            break

# ==================== Image Saving ====================
def save_image(img, filename, base_output_dir, set_name, label):
    save_path = os.path.join(base_output_dir, set_name, label)
    os.makedirs(save_path, exist_ok=True)
    full_path = os.path.join(save_path, filename)
    cv2.imwrite(full_path, img)
    return full_path
def process_and_save(data_split, set_name, csv_records, base_output_dir, augment=False, preprocess_fn=None, all_images_dir=None):
    for i, (img_path, label) in enumerate(tqdm(data_split, desc=f"Processing {set_name}")):
        base_filename = f"{set_name}_{label}_{i}.jpg"
        img = preprocess_fn(img_path)
        if img is None:
            continue
        # Save to class subfolder
        save_image(img, base_filename, base_output_dir, set_name, label)
        csv_records.append((base_filename, label, set_name))
        # Save to ALL_Images
        if all_images_dir:
            os.makedirs(all_images_dir, exist_ok=True)
            cv2.imwrite(os.path.join(all_images_dir, base_filename), img)
        # Augmentation for train only
        if augment and set_name == "train":
            img_array = img_to_array(img)
            augment_and_save(
                img_array=img_array,
                label=label,
                img_index=i,
                output_dir=os.path.join(base_output_dir, set_name, label),
                csv_records=csv_records,
                augment_count=2,
                all_images_dir=all_images_dir
            )
# ==================== Pipeline Runner ====================
def run_pipeline(output_base_dir, csv_output_filename, preprocess_fn):
    print(f"\n🚀 Running pipeline for: {output_base_dir}")
    image_info = load_images_by_class()
    train, val, test = split_data(image_info)
    all_images_dir = os.path.join(output_base_dir, "ALL_Images")
    csv_records = []
    process_and_save(train, "train", csv_records, output_base_dir, augment=True,
                     preprocess_fn=preprocess_fn, all_images_dir=all_images_dir)
    process_and_save(val, "val", csv_records, output_base_dir, augment=False,
                     preprocess_fn=preprocess_fn, all_images_dir=all_images_dir)
    process_and_save(test, "test", csv_records, output_base_dir, augment=False,
                     preprocess_fn=preprocess_fn, all_images_dir=all_images_dir)

    df = pd.DataFrame(csv_records, columns=["filename", "label", "set"])
    csv_path = os.path.join(output_base_dir, csv_output_filename)
    df.to_csv(csv_path, index=False)

    print(f"✅ CSV saved to: {csv_path}")
    print(f"✅ Total processed: {len(csv_records)} images")

# ==================== Main ====================
if __name__ == "__main__":
    run_pipeline(handcrafted_output_base, "labels_with_set.csv", preprocess_fn=preprocess_handcrafted)
    run_pipeline(cnn_output_base, "labels_with_set.csv", preprocess_fn=preprocess_cnn)
