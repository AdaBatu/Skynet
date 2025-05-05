from datetime import datetime
import json
import os
import warnings
from joblib import Parallel, delayed
import cv2
import joblib
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from skopt import BayesSearchCV
from tqdm import tqdm
from utils import load_config, load_dataset, load_test_dataset, print_results
import numpy as np
from skopt.space import Real, Integer, Categorical
import pandas as pd
# sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import time
from skimage.feature import hog
import pickle
import mplcursors
#from models import AverageRegressor, try_reg,stacking_reg,model_ADA,model_GB, model_KNN, model_R, model_RF, model_KRR, HIST_BOOST,model_log_linear, model_DYNAMIC_SELECTOR, showme,model_12

def save_results(pred, mae = 0, note = "nonote"):
    text = "ID,Distance\n"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for i, distance in enumerate(pred):
        text += f"{i:03d},{distance}\n"

    with open(f"prediction_mae_{str(mae)}__{note}__{timestamp}.csv", 'w') as f: 
        f.write(text)

class progress:
    def __init__(self, len=50, name= "Bayesian Search Progress"):
        self.progressbar = tqdm(total=len, desc=name, unit="iteration")
    def update(self, result):
        self.progressbar.update(1)
    def __del__(self):
        self.progressbar.close()
        print("DONE!")

def bayesian_search_model_with_progress(model, param_space, X_train=None, y_train=None, cv=5, scoring='neg_mean_squared_error', save_params=True, save_dir="saved_params", model_name= None):
    """
    Performs Bayesian optimization (using BayesSearchCV) to find the best hyperparameters.
    """
    print(f"Starting Bayesian search with parameter space: {param_space}...")

    # Define the search space using skopt's search space definitions
    search_space = {param: param_space[param] for param in param_space}

    n_it = 100

    # Initialize the tqdm progress bar with a total of n_iter (iterations)
    asd = progress(n_it, "Bayesian Search Progress")


    # Set up the BayesSearchCV with the model and search space
    bayes_search = BayesSearchCV(
        model,
        search_space,
        n_iter=n_it,  # Number of iterations
        cv=cv,
        n_jobs=-1,
        verbose=False,
        scoring=scoring,
        random_state=42, 
    )

    # Attach the custom callback function to update the progress bar

    # Fit the model
    warnings.filterwarnings("ignore", message=".*Singular matrix in solving dual problem.*")

    bayes_search.fit(X_train, y_train, callback=asd.update)
    del asd
    best_model = bayes_search.best_estimator_
    best_params = bayes_search.best_params_

    print(f"Best parameters for {model_name}:", best_params)

    # Save parameters if enabled
    if save_params:
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"best_params_{model_name}.json")
        with open(filename, "w") as f:
            json.dump(best_params, f, indent=4)

    # Close the progress bar after optimization is done
    #progressbar.close()
    safe_set_random_state(best_model,42)
    return best_model, best_params

def model_KRR(gridsearch=False, personalized_pre_processing = False ,X_train=None, y_train=None):
    
    
    if gridsearch:
        if personalized_pre_processing:
            model_name = "KernelRidge_pipe"
            model = Pipeline([
            #('resize', FunctionTransformer(resize_data, validate=False)),  
            ('scaler', RobustScaler(quantile_range=(25,75))),
            ('dim_reduction', PCA(n_components=120)), #100
            ('regressor',  KernelRidge())
            ])
            param_space = {
    'regressor__alpha': Real(1e-6, 1e3, prior='log-uniform'),
    'regressor__kernel': Categorical(['linear', 'poly', 'rbf']),
    'regressor__degree': Integer(2, 5),
    'regressor__coef0': Real(1e-2, 1.0),
    'regressor__gamma': Real(0.0001, 1.0),
        }
        else:
            model_name = "KernelRidge"
            model = KernelRidge()
            param_space = {
    'alpha': Real(1e-6, 1e3), #, prior='log-uniform'
    'kernel': Categorical(['linear', 'poly', 'rbf']),
    'degree': Integer(2, 5),
    'coef0': Real(1e-2, 1.0),
    'gamma': Real(0, 1.0),
        }


        param_grid = {
            'alpha': [0.1, 1.0, 10.0],
            'kernel': ['linear', 'rbf', 'polynomial'],
            'gamma': [None, 0.0001, 0.001, 0.1, 1.0],  # Only applicable for some kernels
            'degree': [3, 4],  # Only applicable for polynomial kernel
        }
        
        if gridsearch==2:
            best_model, best_params = bayesian_search_model_with_progress(model, param_space, X_train, y_train, save_params=True, model_name=model_name)
        #else:
        #    best_model, best_params = grid_search_model_PB(model, param_grid, X_train, y_train)
        return best_model
    else:
        if personalized_pre_processing:
            model_name = "KernelRidge_pipe"
            model = Pipeline([
            #('resize', FunctionTransformer(resize_data, validate=False)),  
            ('scaler', RobustScaler(quantile_range=(25,75))),
            ('dim_reduction', PCA(n_components=120)), #120
            ('regressor',  KernelRidge())
            ])
        else:
            model_name = "KernelRidge"
            model =  KernelRidge()
        best_params = load_best_params(model_name)
        if best_params:

            model.set_params(**best_params)
            safe_set_random_state(model,42)
            model.fit(X_train,y_train)
            return model
        else:
            model =  KernelRidge()
            safe_set_random_state(model,42)
            return model


def model_KNN(gridsearch=False, personalized_pre_processing=False,  X_train=None, y_train=None):
    
    random_state = 42  # Consistent seed for all model
    

    if personalized_pre_processing:
        model_name = "KNeighborsRegressor_pipe"
        # Pipeline approach - will handle its own loading
        print("Using full pipeline with built-in loading")
        knn_pipeline = Pipeline([
            ('scaler', RobustScaler(quantile_range=(17,74))),
            ('dim_reduction', PCA(n_components=83)), #83
            ('regressor', KNeighborsRegressor())
        ])
        
        # Need to pass the label DataFrame to the pipeline
        if gridsearch:

            param_space = {
    'regressor__n_neighbors': Integer(1, 50),
    'regressor__weights': Categorical(['uniform', 'distance']),
    'regressor__metric': Categorical(['euclidean', 'manhattan']),
    'regressor__algorithm': Categorical(['auto', 'ball_tree', 'kd_tree', 'brute']),
    'regressor__leaf_size': Integer(10, 100),
    'regressor__p': Integer(1, 3),
            }
            if gridsearch==2:
                best_model, best_params = bayesian_search_model_with_progress(knn_pipeline, param_space, X_train, y_train,save_params=True, model_name=model_name)
            else:
                param_grid = {
                'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 20, 25, 30],
                'weights': ['distance'],
                'algorithm': ['brute'],
                'p': [1, 1.5, 2, 3],
                'metric_params': [
                    None,
                    {'w': np.random.rand(X_train.shape[1])}
                ]
            }
              #  best_model, best_params = grid_search_model_PB(knn_pipeline, param_grid, X_train, y_train)
            return best_model
        else:
            best_params = load_best_params(model_name)
            safe_set_random_state(knn_pipeline,42)
            if best_params:
                knn_pipeline.set_params(**best_params)
                knn_pipeline.fit(X_train,y_train)
                return knn_pipeline
            else:
                return knn_pipeline

    else:  # Original non-personalized processing
        model_name = "KNeighborsRegressor"
        if gridsearch:
            model = KNeighborsRegressor()

            param_space = {
    'n_neighbors': Integer(1, 200),
    'weights': Categorical(['uniform', 'distance']),
    'metric': Categorical(['euclidean', 'manhattan']),
    'algorithm': Categorical(['auto', 'ball_tree', 'kd_tree', 'brute']),
    'leaf_size': Integer(10, 100),
    'p': Integer(1, 2),
            }
            if gridsearch==2:
                best_model, best_params = bayesian_search_model_with_progress(model, param_space, X_train, y_train, model_name=model_name)
            else:
                param_grid = {
                'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 20, 25, 30],
                'weights': ['distance'],
                'algorithm': ['brute'],
                'p': [1, 1.5, 2, 3],
                'metric_params': [
                    None,
                    {'w': np.random.rand(X_train.shape[1])}
                ]
            }
               # best_model, best_params = grid_search_model_PB(model, param_grid, X_train, y_train)
            return best_model
        else:
            best_params = load_best_params(model_name)
            if best_params:
                model =  KNeighborsRegressor(**best_params)
                safe_set_random_state(model,42)
                return model
            else:
                model =  KNeighborsRegressor()
                safe_set_random_state(model,42)
                return model


def HIST_BOOST(gridsearch=False, personalized_pre_processing=False,  X_train=None, y_train=None):
    
    randomi = 42  # Consistent seed for all model
    

    if personalized_pre_processing:
        model_name = "HistGradientBoostingRegressor_pipe"
        # Pipeline approach - will handle its own loading
        print("Using full pipeline with built-in loading")
        hist_pipeline = Pipeline([
            #('resize', FunctionTransformer(resize_data, validate=False)),
            #('scaler', RobustScaler(quantile_range=(25,75))),
            #('dim_reduction', PCA(n_components=100)),
            ('regressor', HistGradientBoostingRegressor())
        ])
        
        # Need to pass the label DataFrame to the pipeline
        if bool(gridsearch):
            param_grid = {
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                #'n_iter_no_change': [5, 10, 20],
                'max_iter': [100, 200, 500],
                'max_depth': [3, 5, 7],
                'min_samples_leaf': [5, 10, 20],
            }

            param_space = {
            'regressor__max_iter': Integer(100, 500),
            'regressor__max_depth': Integer(5, 30),
            'regressor__learning_rate': Real(0.01, 0.3),
            }
            if gridsearch==2:
                best_model, best_params = bayesian_search_model_with_progress(hist_pipeline, param_space, X_train, y_train, model_name=model_name)
            #else:
             #   best_model, best_params = grid_search_model_PB(hist_pipeline, param_grid, X_train, y_train)
            return best_model
        else:
            best_params = load_best_params(model_name)
            if best_params:
                hist_pipeline.set_params(**best_params)
                return hist_pipeline
            else:
                return hist_pipeline
            
        #model.fit(X_train, y_train)  # X_train should be DataFrame with 'ID' column
        #return model

    else:  # Original non-personalized processing
        model_name = "HistGradientBoostingRegressor"
        if gridsearch:
            model = HistGradientBoostingRegressor()
            param_grid = {
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                #'n_iter_no_change': [5, 10, 20],
                'max_iter': [100, 200, 500],
                'max_depth': [3, 5, 7],
                'min_samples_leaf': [5, 10, 20],
                #'max_bins': [50, 100, 255],
                #'early_stopping': [True],  # Optional, based on whether you want to use early stopping
                #'loss': ['squared_error', 'absolute_error'],  # Optional, based on your preference
            }

            param_space = {
            'max_iter': Integer(100, 500),
            'max_depth': Integer(5, 30),
            'learning_rate': Real(0.01, 0.3),
            }
            if gridsearch==2:
                best_model, best_params = bayesian_search_model_with_progress(model, param_space, X_train, y_train,save_params=True, model_name=model_name)
            #else:
              #  best_model, best_params = grid_search_model_PB(model, param_grid, X_train, y_train)
            return best_model
        else:
            best_params = load_best_params(model_name)
            if best_params:
                model =  HistGradientBoostingRegressor(**best_params)
                safe_set_random_state(model,42)
                model.fit(X_train,y_train)
                return model
            else:
                model =  HistGradientBoostingRegressor()
                safe_set_random_state(model,42)
                return model



def hog_area(image, areainf=True, hogo=True, max_areas=6):
    gray = image.copy()
    gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    features = [np.mean(gray), np.std(gray)]

    contour_features = []

    if areainf:
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours by area (descending), then keep top N
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_areas]

        for c in contours:
            area = cv2.contourArea(c)
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = M["m10"] / M["m00"] / w  # normalize x
                cy = M["m01"] / M["m00"] / h  # normalize y
            else:
                cx, cy = 0, 0

            x, y, bw, bh = cv2.boundingRect(c)
            aspect_ratio = bw / bh if bh != 0 else 0

            contour_features.extend([area, cx, cy, aspect_ratio])

        # Pad to fixed length
        while len(contour_features) < 4 * max_areas:
            contour_features.extend([0, 0, 0, 0])
    else:
        contour_features = [0] * (4 * max_areas)

    hog_features = []
    if hogo:
        hog_features, _ = hog(
            gray,
            orientations=9,
            pixels_per_cell=(15, 15),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            visualize=True
        )
    return np.concatenate([features, contour_features, hog_features])

def prepare_image_views(flattened_image):
    # Reshape to 300x300x3
    image = flattened_image.reshape((300, 300, 3)).astype(np.uint8)
    
    # View 1: KNN – low-res grayscale
    image_gray = image.copy()
    image_gray = cv2.cvtColor(image_gray, cv2.COLOR_RGB2GRAY)
    knn_view = cv2.resize(image_gray, (20, 20), interpolation=cv2.INTER_AREA).reshape(-1) #10,10
    

    # View 2: KRR – downsampled RGB
    image_knn = image.copy()
    krr_view = (hog_area(image_knn,True,True,10)).reshape(-1)

    # View 3: HGB – full image + engineered features
    #lol = doandmask(image)
    #area_feats = hog_area(image, True, False, 10).reshape(-1)
    image_hgb = image.copy()
    area_feats = cv2.resize(image_hgb, (20, 20), interpolation=cv2.INTER_AREA).reshape(-1)

    return (krr_view, krr_view, knn_view)

def metam(imgs):
    results = Parallel(n_jobs=-1)(
    delayed(lambda img: hog_area(img.reshape((300, 300, 3)).astype(np.uint8),True,False,6))(img)
    for img in tqdm(imgs, desc="Tertiary Process", total=len(imgs))
    )
    meta = np.vstack(results)

    return meta


def work_is_work(imgs):
    results = Parallel(n_jobs=-1)(
    delayed(lambda img: prepare_image_views(img))(img)
    for img in tqdm(imgs, desc="Secondary Process", total=len(imgs))
    )
    knn_views, krr_views, area_feats = zip(*results)
    area_feats = np.vstack(area_feats)
    knn_views = np.vstack(knn_views)
    krr_views = np.vstack(krr_views)
    return knn_views, krr_views, area_feats


def safe_set_random_state(model, seed=42):
    # Try to set random_state if it's a valid param
    if "random_state" in model.get_params():
        model.set_params(random_state=seed)
    elif hasattr(model, "steps"):  # It's a Pipeline
        for name, step in model.steps:
            if "random_state" in step.get_params():
                model.set_params(**{f"{name}__random_state": seed})

def showme(errors):
    coldict = {
        0 : 'pink',
        1 : 'skyblue',
        2 : 'red',
        3 : 'purple',
        4: 'orange',
        5: 'green'
    }
    for i in range(errors.shape[0]):
        plt.hist(errors[i,:] , alpha=0.4, density=False, color=coldict[i], bins=20, edgecolor='black')
    mplcursors.cursor(hover=True)
    plt.show()


def resize_data(X):
    X_reshaped = X.reshape(-1, 50, 50)
    new_height, new_width = 20, 20  # Specify your desired dimensions
    resized_X = np.array([cv2.resize(x, (new_height, new_width),interpolation=cv2.INTER_AREA) for x in X_reshaped])  # Resizing images
    return resized_X.reshape(len(X), -1)  # Flatten images to 2D for model input


class AverageRegressor:
    def __init__(self, gridsearch=False, personalized_pre_processing=False,X_train=None, y_train=None,x1=None, y1=None):
        self.gridsearch = gridsearch
        self.personalized_pre_processing = personalized_pre_processing
        self.results = None
        a, b, c = work_is_work(X_train)
        infk = metam(x1)
        self.base_models = [
            ('knn', model_KNN(False, True, a, y_train)),
            ('kr', model_KRR(False, True, b, y_train)),
            #('his', HIST_BOOST(False, False, c, y_train)),
        ]
        self.meta = LinearRegression(n_jobs=-1)
        c,d,e = work_is_work(x1)
        
        ll = {'knn': c, 'kr': d, 'his': e}
        results = np.array([model.predict(ll[i]) for i, model in self.base_models]).T

        #self.meta.fit(np.concatenate([infk,results], axis=1), y1)
        self.meta.fit(results, y1)

    def predict(self, X_test):
        # Make predictions with each base model
        infk = metam(X_test)
        a, b, c = work_is_work(X_test)
        ll = {'knn': a, 'kr': b, 'his': c}
        results = np.array([model.predict(ll[i]) for i, model in self.base_models])
        #result = np.divide(np.sum(results, axis=0),2)   # Sum predictions from all models
        self.results = results  # Store results for later use
        #result = self.meta.predict(np.concatenate([infk,results.T], axis=1))  # Concatenate meta features with base model predictions
        result = self.meta.predict(results.T)
        # Stack predictions and average them
        
        return result
    
    def compare(self, y_test):
        print(self.results.shape[0])
        for i in range(self.results.shape[0]):
            print_results(self.results[i], y_test)
        showme(self.results - y_test)


    def train_two(self, X_train=None, y_train=None):
        a, b, c = work_is_work(X_train)
        self.base_models = [
            ('knn', model_KNN(False, True, a, y_train)),
            ('kr', model_KRR(False, True, b, y_train)),
            #('his', HIST_BOOST(False, False, c, y_train)),
        ]
    def save(self):
        joblib.dump(self, "2_models_pipeline.pkl")


def load_best_params(model_name, save_dir="saved_params"):
    filepath = os.path.join(save_dir, f"best_params_{model_name}.json")
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            print(f"✅ Loaded best_params for {model_name} from file.")
            return json.load(f)
    else:
        print(f"⚠️ No saved best_params found for {model_name}. Using defaults.")
        return None


if __name__ == "__main__":
    # Load configs from "config.yaml"
    config = load_config()
    gs = False
    dyna = False
    crop = False  #1 if no crop 2 if crop
    personalized_pre_processing = True  # Set to False to use traditional approach
    preprocess_var = 1   #0 for     black/white // 1 for only rgb // 2 for only edges // 3 for hog+edges // 4 for contour // 5 for LAB //6 for extreme things   
    
    #Train_meta, images, distances = load_custom_dataset(config, "train", preprocess_var, dyna=dyna)  
    #Actual_meta, X_test_ch = load_test_custom_dataset(config, preprocess_var, dyna=dyna)
    
    images, distances = load_dataset(config)
    X_test_ch = load_test_dataset(config)

    X_train, X_test, y_train, y_test = train_test_split(images, distances, train_size=0.8, random_state=42)
    X1_test, X2_test, y1_test, y2_test = train_test_split(X_test, y_test, train_size=0.5, random_state=42)

    # Model - pipeline approach
    """model = model_KNN(
        personalized_pre_processing = True, 
        gridsearch=gs,
        config=config,
        X_train=X_train_df,
        y_train=y_train
    )
    """
    
    #model = try_reg()
    #model = model_12()
    #model = stacking_reg(False, True, X_train,y_train)
    model = AverageRegressor(False, True, X_train,y_train,X1_test, y1_test)

    #model = model_RF(gs, False, X_train, y_train)
    #model = model_KNN(gs, personalized_pre_processing, X_train, y_train)
    #model = model_ADA(gs, True, X_train, y_train)
    #model = model_KRR(gs, personalized_pre_processing, X_train, y_train)
    #model = HIST_BOOST(gs, False, X_train, y_train)
    
    #model.fit(X_train, y_train)
    # Prediction
    y_pred = model.predict(X2_test)
    model.compare(y2_test)
    mae = print_results(y_pred, y2_test)
    #model.fit(images,distances)
    model.train_two(images, distances)
    y_pred_ch = model.predict(X_test_ch)
    save_results(y_pred_ch,mae,f"{model.__class__.__name__}")
    model.save()

