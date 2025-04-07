"""Utility functions for project 1."""
import yaml
import os
import cv2
from skimage.feature import hog
from skimage.feature import local_binary_pattern
import numpy as np
from pathlib import Path
import pandas as pd
from PIL import Image
from sklearn.metrics import mean_absolute_error, r2_score
from skimage.transform import resize
from skimage import color

IMAGE_SIZE = (300, 300)

def extract_extra_features(image_array, image):
    from skimage.feature import hog

    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    gray_image = color.rgb2gray(image)  # Convert to grayscale
    
    # Edge features
    edges = cv2.Canny(gray, 50, 150)
    edge_count = np.sum(edges > 0)

    # Contour features
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    max_area = max(areas) if areas else 0
    mean_area = np.mean(areas) if areas else 0

    # HOG features
    #if gray.shape[0] < 64 or gray.shape[1] < 64:
    #    raise ValueError(f"Image is too small for HOG feature extraction. Image shape: {gray.shape}")
    hog_features, _ = hog(gray_image, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)

    return np.concatenate([[edge_count, max_area, mean_area], hog_features])


def load_config():
    with open("./config.yaml", "r") as file:
        config = yaml.safe_load(file)
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

    all_features = []
    for _, row in labels.iterrows():
        image = Image.open(
            config["data_dir"] / f"{split}_images" / f"{row['ID']}.png"
        )
        image_raw = image
        if not config["load_rgb"]:
            image = image.convert("L")
        image = image.resize(
            (
                IMAGE_SIZE[0] // config["downsample_factor"],
                IMAGE_SIZE[1] // config["downsample_factor"],
            ),
            resample=Image.BILINEAR,
        )
        image_np = np.asarray(image)
        image_flat = image_np.reshape(-1)

        extra_features = extract_extra_features(image_np, image_raw)
        full_feature_vector = np.concatenate([image_flat, extra_features])

        all_features.append(full_feature_vector)
    images = np.vstack(all_features)

    
    distances = labels["distance"].to_numpy()
    return images, distances


def load_test_dataset(config):
    feature_dim = (IMAGE_SIZE[0] // config["downsample_factor"]) * (
        IMAGE_SIZE[1] // config["downsample_factor"]
    )
    feature_dim = feature_dim * 3 if config["load_rgb"] else feature_dim

    images = []
    img_root = os.path.join(config["data_dir"], "test_images")
    all_features = []
    for img_file in sorted(os.listdir(img_root)):
        if img_file.endswith(".png"):
            image = Image.open(os.path.join(img_root, img_file))
            image_raw = image
            if not config["load_rgb"]:
                image = image.convert("L")
            image = image.resize(
                (
                    IMAGE_SIZE[0] // config["downsample_factor"],
                    IMAGE_SIZE[1] // config["downsample_factor"],
                ),
                resample=Image.BILINEAR,
            )
        image_np = np.asarray(image)
        image_flat = image_np.reshape(-1)

        extra_features = extract_extra_features(image_np, image_raw)
        full_feature_vector = np.concatenate([image_flat, extra_features])

        all_features.append(full_feature_vector)
    images = np.vstack(all_features)

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
    
#gridsearch