"""Utility functions for project 1."""
from picturework import apply_blue_tone_and_extract_feature, hog_area, detect_floor_region, meta_finder, adjust_brightness_to_mean,doandmask
import yaml
import os
import numpy as np
from pathlib import Path
import pandas as pd
from PIL import Image
from sklearn.metrics import mean_absolute_error, r2_score
from skimage.transform import resize
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import cv2
from PIL import Image, ImageOps, ImageFilter
from tqdm import tqdm
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
from joblib import Parallel, delayed
from grid_search import progress

IMAGE_SIZE = (300, 300)



def load_config():
    with open("./config.yaml", "r") as file:
        config = yaml.safe_load(file)
    print(config)
    config["data_dir"] = Path(config["data_dir"])

    if config["load_rgb"] is None or config["downsample_factor"] is None:
        raise NotImplementedError("Make sure to set load_rgb and downsample_factor!")
    
    

    print(f"[INFO]: Configs are loaded with: \n {config}")
    return config


def dataset_process(config, split, cum, dyna ,row):
    image = Image.open(config["data_dir"] / f"{split}_images" / f"{row['ID']}.png")
    raw_img = image.copy()
    if not config["load_rgb"]:
        image = image.convert("L")

    raw_img = np.asarray(raw_img)
    image_np = np.asarray(image)
    feature_vec = np.zeros((3,3))
    feature_vec_re = None
    match cum:          #0 for black/white // 1 for only rgb // 2 for only edges // 3 for hog+edges // 4 for contour // 5 for LAB //6 for extreme things   
        case 0:
            feature_vec = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        case 1:
            feature_vec = image_np
        case 2:
            feature_vec_re = hog_area(raw_img,True,False)
        case 3:
            feature_vec_re = hog_area(raw_img,True,True)
        case 4:
            feature_vec = detect_floor_region(image_np)
        case 5:
            image = image.resize(
                (
                    IMAGE_SIZE[0] // config["downsample_factor"],
                    IMAGE_SIZE[1] // config["downsample_factor"],
                ),
                resample=Image.BILINEAR,
            )
            image_np = np.asarray(image)
            feature_vec_re = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB).reshape(-1)  
        case 6:
            feature_vec = apply_blue_tone_and_extract_feature(image_np,False)
        case 7:
            feature_vec = doandmask(image_np)
    if dyna:
        meta_features = meta_finder(raw_img)

        
    if cum !=5 and cum != 3 and cum != 2:
        feature_vec_re = cv2.resize(feature_vec,
         (
             IMAGE_SIZE[0] // config["downsample_factor"],
             IMAGE_SIZE[1] // config["downsample_factor"],
         ), interpolation=cv2.INTER_LINEAR).reshape(-1)  
    
    return (meta_features, feature_vec_re) if dyna else (None, feature_vec_re)

def load_custom_dataset(config, split="train", cum = 1, dyna=False):
    X_meta_list = []
    labels = pd.read_csv(
        config["data_dir"] / f"{split}_labels.csv", dtype={"ID": str}
    )

    feature_dim = (IMAGE_SIZE[0] // config["downsample_factor"]) * (
        IMAGE_SIZE[1] // config["downsample_factor"]
    )
    feature_dim = feature_dim * 3 if config["load_rgb"] else feature_dim

    images = np.zeros((len(labels), feature_dim))
    all_features = []
    #for _, row in labels.iterrows():
    results = Parallel(n_jobs=-1)(
    delayed(lambda row: dataset_process(config, split, cum, dyna ,row))(row)
    for _, row in tqdm(labels.iterrows(), desc="Processing Rows", total=len(labels))
    )

    X_meta_list, all_features = zip(*results) if dyna else ([], [r for _, r in results])
    X_meta = np.array(X_meta_list)
    images = np.vstack(all_features)
    distances = labels["distance"].to_numpy()
    return X_meta, images, distances



def process_image(config, cum, dyna, img_file, img_root):
    image = Image.open(os.path.join(img_root, img_file))
    raw_img = image
    if not config["load_rgb"]:
        image = image.convert("L")
    raw_img = np.asarray(raw_img)
    image_np = np.asarray(image)
    image_flat = image_np.reshape(-1)
    feature_vec = np.zeros((3,3))
    match cum:          #0 for black/white // 1 for only rgb // 2 for only edges // 3 for hog+edges // 4 for contour // 5 for LAB //6 for extreme things   
        case 0:
            feature_vec = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        case 1:
            feature_vec = image_np
        case 2:
            feature_vec_re = hog_area(raw_img,True,False)
        case 3:
            feature_vec_re = hog_area(raw_img,True,True)
        case 4:
            feature_vec = detect_floor_region(image_np)
        case 5:
            image = image.resize(
                (
                    IMAGE_SIZE[0] // config["downsample_factor"],
                    IMAGE_SIZE[1] // config["downsample_factor"],
                ),
                resample=Image.BILINEAR,
            )
            image_np = np.asarray(image)
            feature_vec_re = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB).reshape(-1)  
        case 6:
            feature_vec = apply_blue_tone_and_extract_feature(image_np,False)
        case 7:
            feature_vec = doandmask(image_np)
    if dyna:
        meta_features = meta_finder(raw_img)

    if cum !=5 and cum != 3 and cum != 2:
        feature_vec_re = cv2.resize(feature_vec,
         (
             IMAGE_SIZE[0] // config["downsample_factor"],
             IMAGE_SIZE[1] // config["downsample_factor"],
         ), interpolation=cv2.INTER_LINEAR).reshape(-1) 
        
    return (meta_features, feature_vec_re) if dyna else (None, feature_vec_re)


def load_test_custom_dataset(config, cum, dyna=False):
    X_meta_list = []
    feature_dim = (IMAGE_SIZE[0] // config["downsample_factor"]) * (
        IMAGE_SIZE[1] // config["downsample_factor"]
    )
    feature_dim = feature_dim * 3 if config["load_rgb"] else feature_dim
    all_features = []
    images = []
    img_root = os.path.join(config["data_dir"], "test_images")
    img_files = (os.listdir(img_root))

    results = Parallel(n_jobs=-1)(
    delayed(process_image)(config, cum, dyna, img_file, img_root)
    for img_file in tqdm(img_files, desc="Processing Rows", total=len(img_files))
)
    X_meta_list, all_features = zip(*results) if dyna else ([], [r for _, r in results])
    X_meta = np.vstack(X_meta_list)
    images = np.vstack(all_features)
    #progressbar.close()
    return X_meta, images


def print_results(gt, pred):
    print(f"MAE: {round(mean_absolute_error(gt, pred)*100, 3)}")
    print(f"R2: {round(r2_score(gt, pred)*100, 3)}")


def save_results(pred):
    text = "ID,Distance\n"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for i, distance in enumerate(pred):
        text += f"{i:03d},{distance}\n"

    with open(f"prediction_{timestamp}_.csv", 'w') as f: 
        f.write(text)

def save_model(model):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"models/random_forest_model_{timestamp}.pkl"
    
    joblib.dump(model, filename)
    print(f"[INFO] Model saved to: {filename}")



"""
def load_dataset(config, split="train"):
    labels = pd.read_csv(
        config["data_dir"] / f"{split}_labels.csv", dtype={"ID": str}
    )

    feature_dim = (IMAGE_SIZE[0] // config["downsample_factor"]) * (
        IMAGE_SIZE[1] // config["downsample_factor"]
    )
    feature_dim = feature_dim * 3 if config["load_rgb"] else feature_dim

    images = np.zeros((len(labels), feature_dim))

    idx = 0
    for _, row in labels.iterrows():
        image = Image.open(
            config["data_dir"] / f"{split}_images" / f"{row['ID']}.png"
        )
        if not config["load_rgb"]:
            image = image.convert("L")
        image = image.resize(
            (
                IMAGE_SIZE[0] // config["downsample_factor"],
                IMAGE_SIZE[1] // config["downsample_factor"],
            ),
            resample=Image.BILINEAR,
        )
        image = np.asarray(image).reshape(-1)
        images[idx] = image
        idx += 1

    distances = labels["distance"].to_numpy()
    return images, distances
"""

"""
def load_test_dataset(config):
    feature_dim = (IMAGE_SIZE[0] // config["downsample_factor"]) * (
        IMAGE_SIZE[1] // config["downsample_factor"]
    )
    feature_dim = feature_dim * 3 if config["load_rgb"] else feature_dim

    images = []
    img_root = os.path.join(config["data_dir"], "test_images")

    for img_file in sorted(os.listdir(img_root)):
        if img_file.endswith(".png"):
            image = Image.open(os.path.join(img_root, img_file))
            if not config["load_rgb"]:
                image = image.convert("L")
            image = image.resize(
                (
                    IMAGE_SIZE[0] // config["downsample_factor"],
                    IMAGE_SIZE[1] // config["downsample_factor"],
                ), interpolation=cv2.INTER_LINEAR
            )
            image = np.asarray(image).reshape(-1)
            images.append(image)

    return images
"""

############################## pipeline ##########################################

from sklearn.base import BaseEstimator, TransformerMixin

class ImagePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, data_dir, split="train", downsample_factor=1, load_rgb=False, 
                 enhance_contrast=True, normalize=True):
        self.data_dir = Path(data_dir)
        self.split = split
        self.downsample_factor = downsample_factor
        self.load_rgb = load_rgb
        self.enhance_contrast = enhance_contrast
        self.normalize = normalize
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        """X should be a DataFrame with 'ID' column"""
        processed_images = []
        for _, row in X.iterrows():
            # Construct full image path
            img_path = self.data_dir / f"{self.split}_images" / f"{row['ID']}.png"
            
            try:
                img = Image.open(img_path)
                
                if not self.load_rgb:
                    img = img.convert('L')
                    
                if self.enhance_contrast:
                    img = ImageOps.autocontrast(img)
                    
                img = img.resize((
                    IMAGE_SIZE[0] // self.downsample_factor,
                    IMAGE_SIZE[1] // self.downsample_factor
                ), Image.BILINEAR)
                
                img_array = np.array(img)
                if self.normalize:
                    img_array = img_array / 255.0
                    
                processed_images.append(img_array.flatten())
                
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                # Return zero array if image fails to load
                dim = (IMAGE_SIZE[0]//self.downsample_factor) * (IMAGE_SIZE[1]//self.downsample_factor)
                dim = dim * (3 if self.load_rgb else 1)
                processed_images.append(np.zeros(dim))
        
        return np.array(processed_images)
    
class DistanceFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, extract_blur=True, extract_edges=True):
        self.extract_blur = extract_blur
        self.extract_edges = extract_edges
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []
        for img_array in X:
            img = img_array.reshape(IMAGE_SIZE)  # Reshape to original dimensions
            
            feature_dict = {}
            
            # Blur estimation (distance affects blur)
            if self.extract_blur:
                blur_value = cv2.Laplacian(img, cv2.CV_64F).var()
                feature_dict['blur'] = blur_value
            
            # Edge density (closer objects have more edges)
            if self.extract_edges:
                edges = cv2.Canny(img, 100, 200)
                edge_density = np.mean(edges)
                feature_dict['edge_density'] = edge_density
            
            # Combine with original pixels
            combined = np.concatenate([img_array.flatten(), list(feature_dict.values())])
            features.append(combined)
        
        return np.array(features)