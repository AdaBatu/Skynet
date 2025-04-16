"""Utility functions for project 1."""
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
                ),
                resample=Image.BILINEAR,
            )
            image = np.asarray(image).reshape(-1)
            images.append(image)

    return images


def print_results(gt, pred):
    print(f"MAE: {round(mean_absolute_error(gt, pred)*100, 3)}")
    print(f"R2: {round(r2_score(gt, pred)*100, 3)}")


def save_results(pred):
    text = "ID,Distance\n"

    for i, distance in enumerate(pred):
        text += f"{i:03d},{distance}\n"

    with open("prediction.csv", 'w') as f: 
        f.write(text)

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