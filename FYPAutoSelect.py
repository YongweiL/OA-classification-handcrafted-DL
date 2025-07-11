import os
import cv2
import shutil
import numpy as np
from glob import glob
from tqdm import tqdm

# ======================== Config =========================
INPUT_DIR = r"C:\UTM Degree\y4s2\PSM1_Dr Nies\KOA\data\ori data\FYP data\oai"
OUTPUT_DIR = r"C:\UTM Degree\y4s2\PSM1_Dr Nies\KOA\data\ori data\FYP data\selected_clear_images"

MAX_TOTAL_IMAGES = 5300
GRADE_4_LABEL = '4'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============== Blurriness Detection =====================
def calculate_blur_score(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return -1
    return cv2.Laplacian(img, cv2.CV_64F).var()

# ============== Main Selection Logic =====================
def select_images_under_limit():
    all_scores = {}
    # Step 1: Always include all Grade 4 images
    print(f"\n📌 Including all Grade {GRADE_4_LABEL} images...")
    grade4_input_path = os.path.join(INPUT_DIR, GRADE_4_LABEL)
    grade4_output_path = os.path.join(OUTPUT_DIR, GRADE_4_LABEL)
    os.makedirs(grade4_output_path, exist_ok=True)
    grade4_images = glob(os.path.join(grade4_input_path, "*.jpg")) + glob(os.path.join(grade4_input_path, "*.png"))
    for path in tqdm(grade4_images, desc="Copying Grade 4"):
        shutil.copy2(path, os.path.join(grade4_output_path, os.path.basename(path)))

    total_remaining = MAX_TOTAL_IMAGES - len(grade4_images)
    print(f"✅ Copied {len(grade4_images)} images from Grade 4.")
    print(f"🎯 Remaining images to select: {total_remaining}")
    # Step 2: Score and store images for other grades
    for label in ['0', '1', '2', '3']:
        class_input_path = os.path.join(INPUT_DIR, label)
        image_paths = glob(os.path.join(class_input_path, "*.jpg")) + glob(os.path.join(class_input_path, "*.png"))
        
        scores = []
        for path in tqdm(image_paths, desc=f"Scoring {label}"):
            score = calculate_blur_score(path)
            if score >= 0:
                scores.append((path, score))
        all_scores[label] = scores
    # Step 3: Calculate how many images to select per class 0–3
    total_available = sum(len(v) for v in all_scores.values())
    selection_ratios = {label: len(v) / total_available for label, v in all_scores.items()}
    selection_counts = {label: int(total_remaining * selection_ratios[label]) for label in all_scores}

    # Step 4: Select top sharp images and copy
    for label, scores in all_scores.items():
        scores.sort(key=lambda x: x[1], reverse=True)
        selected = scores[:selection_counts[label]]
        class_output_path = os.path.join(OUTPUT_DIR, label)
        os.makedirs(class_output_path, exist_ok=True)
        for src_path, _ in selected:
            filename = os.path.basename(src_path)
            dst_path = os.path.join(class_output_path, filename)
            shutil.copy2(src_path, dst_path)
        print(f"✅ Selected {len(selected)} sharpest images for {label}")
# ========================== Run ==========================
if __name__ == "__main__":
    select_images_under_limit()
    print("\n🎉 All selected images saved to:")
    print(f"📁 {OUTPUT_DIR}")




