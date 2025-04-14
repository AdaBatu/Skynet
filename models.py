from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from grid_search import grid_search_model
import json
import os 

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


def model_KNN(gridsearch=False, X_train=None, y_train=None):
    model_name = "KNeighborsRegressor"

    if gridsearch:
        model = KNeighborsRegressor()
        param_grid = {
            'n_neighbors': [5, 10, 15],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': [30, 40, 50]
        }
        best_model, best_params = grid_search_model(model, param_grid, X_train, y_train)
        return best_model
    else:
        best_params = load_best_params(model_name)
        if best_params:
            return KNeighborsRegressor(**best_params)
        else:
            return KNeighborsRegressor()


def model_RF(gs=False, X_train=None, y_train=None):
    model_name = "RandomForestRegressor"

    if gs:
        model = RandomForestRegressor()
        param_grid = {                      # to be updated 
            'n_estimators': [100, 200],
            'max_depth': [None, 10],
            'max_features': ['auto', 'sqrt'],
        }
        best_model, best_params = grid_search_model(model, param_grid, X_train, y_train)
        return best_model
    
    else:
        best_params = load_best_params(model_name)
        if best_params:
            return RandomForestRegressor(**best_params)
        else:
            return RandomForestRegressor()  # Fallback defaults


def model_GB(gridsearch=False, X_train=None, y_train=None):
    model_name = "GradientBoostingRegressor"

    if gridsearch:
        model = GradientBoostingRegressor()
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5],
            'subsample': [0.8, 1.0]
        }
        best_model, best_params = grid_search_model(model, param_grid, X_train, y_train)
        return best_model
    else:
        best_params = load_best_params(model_name)
        if best_params:
            return GradientBoostingRegressor(**best_params)
        else:
            return GradientBoostingRegressor()


def model_R(gridsearch=False, X_train=None, y_train=None):
    model_name = "Ridge"

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

