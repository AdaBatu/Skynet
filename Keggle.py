import numpy as np
# sklearn imports
import joblib
import xgboost as xgb  # Import XGBoost
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import GridSearchCV
# SVRs are not allowed in this project.
import yaml
import os
import cv2
import cupy as cp
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

    hog_features = np.zeros(1)
    gray = image_array
    #gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    #gray_image = color.rgb2gray(image)  # Convert to grayscale
    
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
    #hog_features, _ = hog(gray_image, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)

    return np.concatenate([[edge_count, max_area, mean_area], hog_features])


def load_config():
    #/kaggle/input/data12
    return {'data_dir': Path('./data'), 'load_rgb': False, 'downsample_factor': 10}


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
        if int(row['ID']) % 1000 == 0:
            print(row['ID'])
        #image_raw = image
        image_raw = 0
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

    print(images)
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
            if int(img_file.split(".")[0]) % 100 == 0:
                print(img_file)
            #image_raw = image
            image_raw = 0
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


config = load_config()
    # Load dataset: images and corresponding minimum distance values
images, distances = load_dataset(config)
wololo = load_test_dataset(config)
print(f"[INFO]: Dataset loaded with {len(images)} samples.")

    
X = images
la = wololo
y = distances
X_train, X_test, y_train, y_test = train_test_split(X,y)

def grid_search(model):
    param_grid = {
    'max_depth': [10, 30 , 50],
    'min_samples_split': [2, 3, 5],
    'min_samples_leaf': [1, 2, 5]
    }

    scorer = make_scorer(mean_absolute_error, greater_is_better=False)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,  # 5-fold cross-validation
        scoring=scorer,
        n_jobs=-1,  # Use all cores
        verbose=2
        )

    grid_search.fit(X,y)

    print("Best parameters found:")
    print(grid_search.best_params_)

    print("Best MAE:")
    print(-grid_search.best_score_)

    return grid_search.best_estimator_

def load_par(model):
    return {'max_depth': 30, 'learning_rate': 0.005, 'n_estimators': 5000, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1,'reg_lambda': 1.0}

def save_results(pred):
    text = "ID,Distance\n"

    for i, distance in enumerate(pred):
        text += f"{i:03d},{distance}\n"

    with open("/kaggle/working/prediction1.csv", 'w') as f: 
        f.write(text)

X_train_cp = cp.array(X)
y_train_cp = cp.array(y)
X_test_cp = cp.array(X_test)
y_test_cp = cp.array(y_test)


best_params = load_par(None)  # you don't need the 'model' arg here unless you want to customize
model = xgb.XGBRegressor(**best_params, objective='reg:squarederror', random_state=42, verbosity=1)  # XGBoost Regressor with best params
model.set_params(tree_method='hist', device='cuda')
X_train_cp1 = cp.array(X_train)
print("loaded Xtrain to gpu")
y_train_cp1 = cp.array(y_train)
print("loaded Ytrain to gpu")
model.fit(X_train_cp1, y_train_cp1, eval_set=[(X_test_cp, y_test_cp)], early_stopping_rounds=15)
joblib.dump(model, "/kaggle/working/random_forest_model.pkl")

import matplotlib.pyplot as plt
y_pred_X = model.predict(X)
y_pred = model.predict(X_test)
y_dif= (y - y_pred_X)*100
plt.hist(y_dif,density=True, color='skyblue', edgecolor='black')
pred_end = model.predict(la)
print_results(y_test, y_pred)
save_results(pred_end)


if input("Grid Search ?") == "Yes":
    model = RandomForestRegressor({"n_estimators" : [400]}, random_state=42)
    print(model)
    model = grid_search(model)
    print(model)
else:
    best_params = load_par(None)  # you don't need the 'model' arg here unless you want to customize
    model = RandomForestRegressor(**best_params, random_state=42, verbose=2)
    model.fit(X_train, y_train)  # <--- this is what was missing
    #prediction
joblib.dump(model, "/kaggle/working/random_forest_model.pkl")