import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.applications import VGG19, ResNet101
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_vgg
from tensorflow.keras.applications.resnet import preprocess_input as preprocess_resnet
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
import pywt
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from functools import lru_cache
import tensorflow as tf
# Initialize session state
if 'full_data' not in st.session_state:
    st.session_state.full_data = []
st.title("Optimized KOA Feature Extraction Pipeline")
handcrafted_images_path = r'C:\UTM Degree\y4s2\PSM1_Dr Nies\KOA\data\ori data\FYP data\hand\augSplitAfter\ALL_Images'
handcrafted_labels_path = r'C:\UTM Degree\y4s2\PSM1_Dr Nies\KOA\data\ori data\FYP data\hand\augSplitAfter\labels_with_set.csv'
cnn_images_path = r'C:\UTM Degree\y4s2\PSM1_Dr Nies\KOA\data\ori data\FYP data\cnn\augSplitAfter\ALL_Images'
cnn_labels_path = r'C:\UTM Degree\y4s2\PSM1_Dr Nies\KOA\data\ori data\FYP data\cnn\augSplitAfter\labels_with_set.csv'
# Output directory 
save_dir = r'C:\UTM Degree\y4s2\PSM1_Dr Nies\KOA\data\ori data\ComboFix\features'
os.makedirs(save_dir, exist_ok=True)

# Optimized image loading with caching
@lru_cache(maxsize=1000)
def cached_imread(path):
    if path.lower().endswith(('.jpg', '.jpeg')):
        with open(path, 'rb') as f:
            return cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    return cv2.imread(path)

@st.cache_resource
def get_vgg19_model():
    vgg_model = VGG19(weights='imagenet', include_top=True)
    return Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('fc2').output)

@st.cache_resource
def get_resnet101_model():
    return ResNet101(weights='imagenet', include_top=False, pooling='avg')

def extract_cnn_features_batch(image_paths, model, preprocess_fn, batch_size=4):
    # Load all valid images first
    images = []
    valid_indices = []
    for i, path in enumerate(image_paths):
        img = cached_imread(path)
        if img is not None:
            img = cv2.resize(img, (224, 224))  # Resize to expected input size
            images.append(img)
            valid_indices.append(i)
    # Preprocess in batches
    features = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        batch = np.array([preprocess_fn(img) for img in batch])
        batch_features = model.predict(batch, verbose=0)
        features.extend(batch_features)
    # Create full array with None for invalid images
    full_features = [None] * len(image_paths)
    for idx, feat in zip(valid_indices, features):
        full_features[idx] = feat
    
    return full_features

# Precompute GLCM properties to avoid repeated calculations
GLCM_PROPS = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']

def extract_glcm_features_optimized(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    glcm = graycomatrix(gray, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    
    # Vectorized property calculation
    features = np.array([graycoprops(glcm, prop)[0,0] for prop in GLCM_PROPS])
    
    # More efficient matrix calculations
    glcm_mat = glcm[:, :, 0, 0]
    i, j = np.indices(glcm_mat.shape)
    idm = np.sum(glcm_mat / (1 + (i - j) ** 2))
    
    glcm_flat = glcm_mat.ravel()
    mask = glcm_flat > 0
    glcm_flat_nonzero = glcm_flat[mask]
    
    entropy = -np.sum(glcm_flat_nonzero * np.log2(glcm_flat_nonzero))
    mean = np.mean(glcm_flat)
    var = np.var(glcm_flat)
    std = np.std(glcm_flat)
    skew = (np.mean((glcm_flat - mean)**3)) / (std**3 + 1e-6)
    kurtosis = (np.mean((glcm_flat - mean)**4)) / (std**4 + 1e-6)
    
    return np.concatenate([features, [entropy, mean, var, std, skew, kurtosis, idm]])

def extract_dwt_features_optimized(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cA, (cH, cV, cD) = pywt.dwt2(gray, 'haar')
    
    def get_stats(coeff):
        flat = coeff.ravel()
        return [np.mean(flat), np.std(flat), np.sum(flat**2)]
    
    return np.concatenate([
        get_stats(cA),
        get_stats(cH),
        get_stats(cV),
        get_stats(cD)
    ])

def extract_lbp_features_optimized(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    grid_x, grid_y = 3, 3
    block_h = height // grid_y
    block_w = width // grid_x
    radius = 3
    n_points = 24
    n_bins = n_points + 2
    features = []
    
    for y in range(grid_y):
        for x in range(grid_x):
            block = gray[y*block_h:(y+1)*block_h, x*block_w:(x+1)*block_w]
            lbp = local_binary_pattern(block, P=n_points, R=radius, method='uniform')
            hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins + 1), range=(0, n_bins))
            hist = hist.astype('float') / (hist.sum() + 1e-6)
            features.extend(hist[:-3])
    
    return features[:-4]

def extract_handcrafted_features_optimized(image_path):
    img = cached_imread(image_path)
    if img is None:
        return None
    
    try:
        glcm = extract_glcm_features_optimized(img)
        dwt = extract_dwt_features_optimized(img)
        lbp = extract_lbp_features_optimized(img)
        return np.concatenate([glcm, dwt, lbp])
    except Exception as e:
        st.error(f"Error processing {image_path}: {str(e)}")
        return None

def load_data():
    # Load handcrafted data
    hand_labels_df = pd.read_csv(handcrafted_labels_path)
    hand_data = []
    
    for _, row in hand_labels_df.iterrows():
        img_path = os.path.join(handcrafted_images_path, row['filename'])
        hand_data.append({
            'handcrafted_path': img_path,
            'label': row['label'],
            'set': row['set']
        })
    
    # Load CNN data and merge
    cnn_labels_df = pd.read_csv(cnn_labels_path)
    full_data = []
    
    for hand_item in hand_data:
        filename = os.path.basename(hand_item['handcrafted_path'])
        cnn_row = cnn_labels_df[cnn_labels_df['filename'] == filename]
        
        if not cnn_row.empty:
            cnn_path = os.path.join(cnn_images_path, filename)
            full_data.append({
                'filename': filename,
                'handcrafted_path': hand_item['handcrafted_path'],
                'cnn_path': cnn_path,
                'label': hand_item['label'],
                'set': hand_item['set']
            })
    
    return full_data

def optimized_feature_extraction():
    if not st.session_state.full_data:
        st.error("Please load data first")
        return
    data = st.session_state.full_data
    total_samples = len(data)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Get all paths
    cnn_paths = [item['cnn_path'] for item in data]
    handcrafted_paths = [item['handcrafted_path'] for item in data]
    labels = [item['label'] for item in data]
    sets = [item['set'] for item in data]
    # Load models
    status_text.text("Loading models...")
    vgg_model = get_vgg19_model()
    resnet_model = get_resnet101_model()
    # Batch extract CNN features
    status_text.text("Extracting VGG19 features ...")
    vgg_features = extract_cnn_features_batch(cnn_paths, vgg_model, preprocess_vgg)
    status_text.text("Extracting ResNet101 features ...")
    resnet_features = extract_cnn_features_batch(cnn_paths, resnet_model, preprocess_resnet)
    # Parallel extract handcrafted features
    status_text.text("Extracting handcrafted features...")
    handcrafted_features = [None] * len(data)
    with ThreadPoolExecutor(max_workers=int(multiprocessing.cpu_count() * 0.75)) as executor:
        futures = {executor.submit(extract_handcrafted_features_optimized, path): i 
                  for i, path in enumerate(handcrafted_paths)}
        
        for future in as_completed(futures):
            i = futures[future]
            try:
                handcrafted_features[i] = future.result()
            except Exception as e:
                st.error(f"Error processing {handcrafted_paths[i]}: {str(e)}")
    
    # Filter valid samples
    valid_indices = [i for i in range(len(data)) 
                    if (vgg_features[i] is not None and 
                        resnet_features[i] is not None and 
                        handcrafted_features[i] is not None)]
    
    vgg_features = [vgg_features[i] for i in valid_indices]
    resnet_features = [resnet_features[i] for i in valid_indices]
    handcrafted_features = [handcrafted_features[i] for i in valid_indices]
    labels = [labels[i] for i in valid_indices]
    sets = [sets[i] for i in valid_indices]
    filenames = [data[i]['filename'] for i in valid_indices]
    
    # Save results
    pd.DataFrame(vgg_features).to_csv(os.path.join(save_dir, 'Vgg19.csv'), index=False)
    pd.DataFrame(resnet_features).to_csv(os.path.join(save_dir, 'Resnet101.csv'), index=False)
    pd.DataFrame(handcrafted_features).to_csv(os.path.join(save_dir, 'Handcrafted.csv'), index=False)
    
    # Save metadata
    pd.DataFrame({
        'filename': filenames,
        'label': labels,
        'set': sets
    }).to_csv(os.path.join(save_dir, 'labels_with_set.csv'), index=False)
    
    # Final report
    st.success(f"""
    Feature extraction complete!
    - Expected samples: {total_samples}
    - Processed samples: {len(valid_indices)}
    - Failed samples: {total_samples - len(valid_indices)}
    """)
    
    # Show label distribution
    st.write("### Label Distribution")
    label_counts = pd.Series(labels).value_counts().sort_index()
    st.table(label_counts)
    
    # Show set distribution
    st.write("### Set Distribution")
    set_counts = pd.Series(sets).value_counts()
    st.table(set_counts)
    
    # Verify file sizes
    st.write("### Saved Files Verification")
    for fname in ['Vgg19.csv', 'Resnet101.csv', 
                 'Handcrafted.csv', 'labels_with_set.csv']:
        path = os.path.join(save_dir, fname)
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024 * 1024)  # in MB
            st.write(f"{fname}: {size:.2f} MB")
        else:
            st.error(f"Missing file: {fname}")

def main():
    st.subheader("1. Data Loading")
    if st.button("Load Data"):
        data = load_data()
        st.session_state.full_data = data
        
        st.write("### Dataset Summary")
        st.write(f"Total samples: {len(data)}")
        
        # Show label distribution
        label_counts = pd.Series([item['label'] for item in data]).value_counts().sort_index()
        st.write("Label distribution:")
        st.table(label_counts)
        
        # Show set distribution
        set_counts = pd.Series([item['set'] for item in data]).value_counts()
        st.write("Set distribution:")
        st.table(set_counts)
    
    st.subheader("2. Feature Extraction")
    if st.button("Start Feature Extraction"):
        if not st.session_state.full_data:
            st.warning("Please load data first")
        else:
            optimized_feature_extraction()

if __name__ == "__main__":
    # Disable TensorFlow logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')
    
    main()