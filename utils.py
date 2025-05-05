"""Utility functions for project 1."""
from picturework import apply_blue_tone_and_extract_feature, hog_area, detect_floor_region, hog_area_old, meta_finder, adjust_brightness_to_mean,doandmask
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
    return round(mean_absolute_error(gt, pred)*100, 3)


def save_results(pred, mae = 0, note = "nonote"):
    text = "ID,Distance\n"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for i, distance in enumerate(pred):
        text += f"{i:03d},{distance}\n"

    with open(f"prediction_mae_{str(mae)}__{note}__{timestamp}.csv", 'w') as f: 
        f.write(text)
    

