from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from grid_search import grid_search_model
import json
import os 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from utils import ImagePreprocessor, load_config
from sklearn.model_selection import GridSearchCV

def load_best_params(model_name, save_dir="saved_params"):
    filepath = os.path.join(save_dir, f"best_params_{model_name}.json")
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            print(f"✅ Loaded best_params for {model_name} from file.")
            return json.load(f)
    else:
        print(f"⚠️ No saved best_params found for {model_name}. Using defaults.")
        return None


### Importante: every has to be called with (gridsearch: True/False, X_train, Y_train)


def model_KRR(gs=False, personalized_pre_processing = False ,X_train=None, y_train=None):
    model_name = "KernelRidge"
    
    if gs:
        model = KernelRidge()
        param_grid = {
            'alpha': [0.1, 1.0, 10.0],
            'kernel': ['linear', 'rbf', 'polynomial'],
            'gamma': [None, 0.1, 1.0],  # Only applicable for some kernels
            'degree': [3, 4],  # Only applicable for polynomial kernel
        }
        best_model, best_params = grid_search_model(model, param_grid, X_train, y_train)
        return best_model
    else:
        best_params = load_best_params(model_name)
        if best_params:
            return KernelRidge(**best_params)
        else:
            return KernelRidge()


def model_KNN(gridsearch=False, personalized_pre_processing=False, config = None,  X_train=None, y_train=None):
    model_name = "KNeighborsRegressor"
    random_state = 42  # Consistent seed for all model
    

    if personalized_pre_processing:
        # Pipeline approach - will handle its own loading
        print("Using full pipeline with built-in loading")
        knn_pipeline = Pipeline([
            ('loader', ImagePreprocessor(
                data_dir=config["data_dir"],
                downsample_factor=1,
                load_rgb=True,
                enhance_contrast=True
            )),
            ('scaler', StandardScaler()),
            ('dim_reduction', PCA(n_components=50)),
            ('regressor', KNeighborsRegressor())
        ])
        
        # Need to pass the label DataFrame to the pipeline
        if gridsearch:
            param_grid = {
                'regressor__n_neighbors': [3,5,7],
                'loader__downsample_factor': [1,2],
                'dim_reduction__n_components': [30,50,100]
            }
            model = GridSearchCV(knn_pipeline, param_grid)
        else:
            model = knn_pipeline
            
        model.fit(X_train, y_train)  # X_train should be DataFrame with 'ID' column
        return model

    else:  # Original non-personalized processing
        if gridsearch:
            model = KNeighborsRegressor()
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
            best_model, best_params = grid_search_model(model, param_grid, X_train, y_train)
            return best_model
        else:
            best_params = load_best_params(model_name)
            if best_params:
                return KNeighborsRegressor(**best_params)
            else:
                return KNeighborsRegressor()


def model_RF(gs=False, personalized_pre_processing = False, X_train=None, y_train=None):
    model_name = "RandomForestRegressor"
    random_state = 42

    if gs:
        model = RandomForestRegressor(random_state=random_state)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10],
            'max_features': ['auto', 'sqrt'],
            'random_state': [random_state]  # Locked for grid search
        }
        best_model, best_params = grid_search_model(model, param_grid, X_train, y_train)
        return best_model
    else:
        best_params = load_best_params(model_name)
        if best_params:
            return RandomForestRegressor(**best_params, random_state=random_state)
        else:
            return RandomForestRegressor(random_state=random_state)


def model_GB(gridsearch=False, personalized_pre_processing = False, X_train=None, y_train=None):
    model_name = "GradientBoostingRegressor"
    random_state = 42

    if gridsearch:
        model = GradientBoostingRegressor(random_state=random_state)
        param_grid = {
            'n_estimators': [50, 100, 200, 300, 500],  # Wider range with smaller increments
            'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],  # More granular learning rates
            'max_depth': [2, 3, 4, 5, 6, None],  # Includes shallower trees
            'min_samples_split': [2, 5, 10],  # New - controls node splitting
            'min_samples_leaf': [1, 2, 4],  # New - controls leaf size
            'max_features': ['sqrt', 0.8, None],  # Feature subsampling
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],  # More aggressive subsampling
            'validation_fraction': [0.1, 0.2],  # Early stopping
            'n_iter_no_change': [5, 10],  # Early stopping patience
            'random_state': [random_state]
         }
        best_model, best_params = grid_search_model(model, param_grid, X_train, y_train)
        return best_model
    else:
        best_params = load_best_params(model_name)
        if best_params:
            return GradientBoostingRegressor(**best_params, random_state=random_state)
        else:
            return GradientBoostingRegressor(random_state=random_state)


def model_R(gridsearch=False, personalized_pre_processing = False, X_train=None, y_train=None):


    model_name = "Ridge"
    random_state = 42  # Ridge doesn't use random_state, but added for consistency

    if gridsearch:
        model = Ridge()
        param_grid = {
            'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr']
        }
        best_model, best_params = grid_search_model(model, param_grid, X_train, y_train)
        return best_model
    else:
        best_params = load_best_params(model_name)
        if best_params:
            return Ridge(**best_params)
        else:
            return Ridge()
        
