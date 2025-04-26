from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from grid_search import grid_search_model_PB, bayesian_search_model_with_progress
from picturework import meta_finder
from skopt.space import Real, Integer, Categorical
import json
import os 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.decomposition import PCA
from utils import ImagePreprocessor, load_config
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression

def safe_set_random_state(model, seed=42):
    # Try to set random_state if it's a valid param
    if "random_state" in model.get_params():
        model.set_params(random_state=seed)
    elif hasattr(model, "steps"):  # It's a Pipeline
        for name, step in model.steps:
            if "random_state" in step.get_params():
                model.set_params(**{f"{name}__random_state": seed})

def resize_data(X):
    X_reshaped = X.reshape(-1, 75, 75)
    new_height, new_width = 20, 20  # Specify your desired dimensions
    resized_X = np.array([np.resize(x, (new_height, new_width)) for x in X_reshaped])  # Resizing images
    return resized_X.reshape(len(X), -1)  # Flatten images to 2D for model input


class DynamicWeightRegressor:
    def __init__(self, base_models):
        self.base_models = base_models  # Dict of model_name -> model instance
        self.meta_model = LinearRegression()

    def fit(self, X_image,y, X_meta, X_test, y_test):
        # Train each base model
        error_targets  = []
        for name, model in self.base_models.items():
            print(name)
            safe_set_random_state(model)
            model.fit(X_image, y)
            preds = model.predict(X_test)
            error_targets.append(np.abs(preds-y_test).reshape(-1, 1))

        error_targets  = np.hstack(error_targets)
        # Train meta-model to learn weights
        self.meta_model.fit(X_meta, error_targets)

    def predict(self, X_image, y_lab, X_meta):
        base_preds = []
        base_mae = []
        for name, model in self.base_models.items():
            safe_set_random_state(model)
            predd = model.predict(X_image)
            mae = np.mean(np.abs(predd - y_lab))
            
            print(f"{name} MAE: {mae:.2f}")
            base_preds.append(predd.reshape(-1, 1))

        base_preds = np.hstack(base_preds)
        predicted_errors = self.meta_model.predict(X_meta)

        weights = 1 / (predicted_errors + 1e-8)
        weights = weights / np.sum(weights, axis=1, keepdims=True)

        # Apply weights row-wise
        y_pred = np.sum(base_preds * weights, axis=1)
        return y_pred,predicted_errors


def model_DYNAMIC_SELECTOR(gridsearch1=False, personalized_pre_processing1=False, X_train=None, y_train=None, X_meta = None, X_test = None, y_test = None):
    # Base models
    base_models = {
        "KNN": model_KNN(gridsearch=gridsearch1, personalized_pre_processing=personalized_pre_processing1, X_train=X_train, y_train=y_train),
        "HB": HIST_BOOST(gridsearch=gridsearch1, personalized_pre_processing=False, X_train=X_train, y_train=y_train),
        "KRR": model_KRR(gridsearch=gridsearch1, personalized_pre_processing=personalized_pre_processing1, X_train=X_train, y_train=y_train)
        }
    #"LLR": model_log_linear(gridsearch=gridsearch1, personalized_pre_processing=personalized_pre_processing1, X_train=X_train, y_train=y_train),
    
    model = DynamicWeightRegressor(base_models)
    model.fit(X_train, y_train, X_meta, X_test, y_test)
    return model

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




def model_KRR(gridsearch=False, personalized_pre_processing = False ,X_train=None, y_train=None):
    
    
    if gridsearch:
        if personalized_pre_processing:
            model_name = "KernelRidge_pipe"
            model = Pipeline([
            #('resize', FunctionTransformer(resize_data, validate=False)),  
            ('scaler', StandardScaler()),
            ('dim_reduction', PCA(n_components=50)),
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
        else:
            best_model, best_params = grid_search_model_PB(model, param_grid, X_train, y_train)
        return best_model
    else:
        if personalized_pre_processing:
            model_name = "KernelRidge_pipe"
            model = Pipeline([
            #('resize', FunctionTransformer(resize_data, validate=False)),  
            ('scaler', StandardScaler()),
            ('dim_reduction', PCA(n_components=50)),
            ('regressor',  KernelRidge())
            ])
        else:
            model_name = "KernelRidge"
            model =  KernelRidge()
        best_params = load_best_params(model_name)
        if best_params:

            model.set_params(**best_params)
            safe_set_random_state(model,42)
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
            ('scaler', StandardScaler()),
            ('dim_reduction', PCA(n_components=400)),
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
    'regressor__p': Integer(1, 2),
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
                best_model, best_params = grid_search_model_PB(knn_pipeline, param_grid, X_train, y_train)
            return best_model
        else:
            best_params = load_best_params(model_name)
            safe_set_random_state(knn_pipeline,42)
            if best_params:
                knn_pipeline.set_params(**best_params)
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
                best_model, best_params = grid_search_model_PB(model, param_grid, X_train, y_train)
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
            ('resize', FunctionTransformer(resize_data, validate=False)),
            ('scaler', StandardScaler()),
            #('dim_reduction', PCA(n_components=100)),
            ('regressor', HistGradientBoostingRegressor())
        ])
        
        # Need to pass the label DataFrame to the pipeline
        if bool(gridsearch):
            param_grid = {
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'n_iter_no_change': [5, 10, 20],
                'max_iter': [100, 200, 500],
                'max_depth': [3, 5, 7],
                'min_samples_leaf': [5, 10, 20],
                'max_bins': [50, 100, 255],
                'early_stopping': [True],  # Optional, based on whether you want to use early stopping
                'loss': ['squared_error', 'absolute_error'],  # Optional, based on your preference
            }

            param_space = {
            'regressor__max_iter': Integer(100, 500),
            'regressor__max_depth': Integer(5, 30),
            'regressor__learning_rate': Real(0.01, 0.3),
            }
            if gridsearch==2:
                best_model, best_params = bayesian_search_model_with_progress(hist_pipeline, param_space, X_train, y_train, model_name=model_name)
            else:
                best_model, best_params = grid_search_model_PB(hist_pipeline, param_grid, X_train, y_train)
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
                'n_iter_no_change': [5, 10, 20],
                'max_iter': [100, 200, 500],
                'max_depth': [3, 5, 7],
                'min_samples_leaf': [5, 10, 20],
                'max_bins': [50, 100, 255],
                'early_stopping': [True],  # Optional, based on whether you want to use early stopping
                'loss': ['squared_error', 'absolute_error'],  # Optional, based on your preference
            }

            param_space = {
            'max_iter': Integer(100, 500),
            'max_depth': Integer(5, 30),
            'learning_rate': Real(0.01, 0.3),
            }
            if gridsearch==2:
                best_model, best_params = bayesian_search_model_with_progress(model, param_space, X_train, y_train,save_params=True, model_name=model_name)
            else:
                best_model, best_params = grid_search_model_PB(model, param_grid, X_train, y_train)
            return best_model
        else:
            best_params = load_best_params(model_name)
            if best_params:
                model =  HistGradientBoostingRegressor(**best_params)
                safe_set_random_state(model,42)
                return model
            else:
                model =  HistGradientBoostingRegressor()
                safe_set_random_state(model,42)
                return model

class InverseLogTransformer:
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.expm1(X)  # np.expm1 is the inverse of np.log1p

def model_log_linear(gridsearch=False, personalized_pre_processing=False, X_train=None, y_train=None): 
    model_name = "LogLinearRegression"
    random_state = 42  # Consistent seed for all models
    if y_train is None or len(y_train) == 0:
        raise ValueError("y_train is None or empty. Please provide a valid target variable.")
    # Log transformation of target variable automatically within the pipeline
    log_transformer = FunctionTransformer(np.log1p, validate=True)  # np.log1p is log(x + 1)
    
    if personalized_pre_processing:
        # Pipeline approach - handles scaling, PCA for dimensionality reduction, and log transformation
        print("Using full pipeline with built-in loading")
        
        log_linear_pipeline = Pipeline([
            ('log_transform', log_transformer),  # Apply log transformation to target during training
            ('scaler', StandardScaler()),
            ('dim_reduction', PCA(n_components=50)),
            ('regressor', LinearRegression()),
            ('inverse_log', InverseLogTransformer())  # Apply inverse transformation after predictions
        ])
        
        
        model = log_linear_pipeline
        
        model.fit(X_train, y_train)  # Automatically log-transform and fit the model
        return model
    else:
        # If no personalized pre-processing, fit a simple log-linear model
        model = LinearRegression()
        y_train_log = np.log1p(y_train)  # Apply log transformation manually
        model.fit(X_train, y_train_log)
        return model



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
        
