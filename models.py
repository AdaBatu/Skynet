from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from grid_search import grid_search_model, grid_search_model_PB, bayesian_search_model_with_progress
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

class DynamicWeightRegressor:
    def __init__(self, base_models):
        self.base_models = base_models  # Dict of model_name -> model instance
        self.meta_model = LinearRegression()

    def fit(self, X_image,y_train, X_meta, y):
        # Train each base model
        error_targets  = []
        for name, model in self.base_models.items():
            print(name)
            model.fit(X_image, y)
            preds = model.predict(X_image).reshape(-1, 1)
            error_targets.append(np.abs(preds-y_train.reshape(-1,1)))

        error_targets  = np.hstack(error_targets )

        # Train meta-model to learn weights
        self.meta_model.fit(X_meta, error_targets )

    def predict(self, X_image, y_lab, X_meta):
        base_preds = []
        for name, model in self.base_models.items():
            predd = model.predict(X_image)
            mae = np.mean(np.abs(predd - y_lab))
            print(f"{name} MAE: {mae:.2f}")
            preds = model.predict(X_image).reshape(-1, 1)
            base_preds.append(preds)

        base_preds = np.hstack(base_preds)
        predicted_errors = self.meta_model.predict(X_meta)

        weights = 1 / (predicted_errors + 1e-8)
        weights = weights / np.sum(weights, axis=1, keepdims=True)

        # Apply weights row-wise
        y_pred = np.sum(base_preds * weights, axis=1)
        return y_pred


def model_DYNAMIC_SELECTOR(gridsearch1=False, personalized_pre_processing1=False, X_train=None, y_train=None, X_meta = None):
    # Base models
    base_models = {
        "HB": HIST_BOOST(gridsearch=gridsearch1, personalized_pre_processing=personalized_pre_processing1, X_train=X_train, y_train=y_train),
        "KRR": model_KRR(gridsearch=gridsearch1, personalized_pre_processing=personalized_pre_processing1, X_train=X_train, y_train=y_train),
        "KNN": model_KNN(gridsearch=gridsearch1, personalized_pre_processing=personalized_pre_processing1, X_train=X_train, y_train=y_train)
        }
    #"LLR": model_log_linear(gridsearch=gridsearch1, personalized_pre_processing=personalized_pre_processing1, X_train=X_train, y_train=y_train),
    
    model = DynamicWeightRegressor(base_models)
    model.fit(X_train, y_train, X_meta, y_train)
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
    model_name = "KernelRidge"
    
    if gridsearch:

        model = KernelRidge()
        param_grid = {
            'alpha': [0.1, 1.0, 10.0],
            'kernel': ['linear', 'rbf', 'polynomial'],
            'gamma': [None, 0.0001, 0.001, 0.1, 1.0],  # Only applicable for some kernels
            'degree': [3, 4],  # Only applicable for polynomial kernel
        }
        param_space = {
    'regressor__alpha': Real(1e-6, 1e3, prior='log-uniform'),
    'regressor__kernel': Categorical(['linear', 'poly', 'rbf', 'sigmoid']),
    'regressor__degree': Integer(2, 5),
    'regressor__coef0': Real(0.0, 1.0),
    'regressor__gamma': Real(0.0001, 1.0),
        }
        if gridsearch==2:
            best_model, best_params = bayesian_search_model_with_progress(model, param_space, X_train, y_train)
        else:
            best_model, best_params = grid_search_model_PB(model, param_grid, X_train, y_train)
        return best_model
    else:
        best_params = load_best_params(model_name)
        if best_params:
            return KernelRidge(**best_params)
        else:
            return KernelRidge()


def model_KNN(gridsearch=False, personalized_pre_processing=False,  X_train=None, y_train=None):
    model_name = "KNeighborsRegressor"
    random_state = 42  # Consistent seed for all model
    

    if personalized_pre_processing:
        # Pipeline approach - will handle its own loading
        print("Using full pipeline with built-in loading")
        knn_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('dim_reduction', PCA(n_components=50)),
            ('regressor', KNeighborsRegressor())
        ])
        
        # Need to pass the label DataFrame to the pipeline
        if gridsearch:

            param_space = {
    'regressor__n_neighbors': Integer(1, 50),
    'regressor__weights': Categorical(['uniform', 'distance']),
    'regressor__metric': Categorical(['euclidean', 'manhattan', 'cosine']),
    'regressor__algorithm': Categorical(['auto', 'ball_tree', 'kd_tree', 'brute']),
    'regressor__leaf_size': Integer(10, 100),
    'regressor__p': Integer(1, 2),
            }
            if gridsearch==2:
                best_model, best_params = bayesian_search_model_with_progress(knn_pipeline, param_space, X_train, y_train)
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
            knn_pipeline.set_params(**best_params)
            if best_params:
                return knn_pipeline
            else:
                return knn_pipeline

    else:  # Original non-personalized processing
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
                best_model, best_params = bayesian_search_model_with_progress(model, param_space, X_train, y_train)
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
                return KNeighborsRegressor(**best_params)
            else:
                return KNeighborsRegressor()   


def HIST_BOOST(gridsearch=False, personalized_pre_processing=False,  X_train=None, y_train=None):
    model_name = "HistGradientBoostingRegressor"
    randomi = 42  # Consistent seed for all model
    

    if personalized_pre_processing:
        # Pipeline approach - will handle its own loading
        print("Using full pipeline with built-in loading")
        hist_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('dim_reduction', PCA(n_components=50)),
            ('regressor', HistGradientBoostingRegressor())
        ])
        
        # Need to pass the label DataFrame to the pipeline
        if gridsearch:
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
                best_model, best_params = bayesian_search_model_with_progress(hist_pipeline, param_space, X_train, y_train)
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
        if gridsearch:
            model = HistGradientBoostingRegressor(random_state=randomi)
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
                best_model, best_params = bayesian_search_model_with_progress(model, param_space, X_train, y_train)
            else:
                best_model, best_params = grid_search_model_PB(model, param_grid, X_train, y_train)
            return best_model
        else:
            best_params = load_best_params(model_name)
            if best_params:
                return HistGradientBoostingRegressor(**best_params, random_state=randomi)
            else:
                return HistGradientBoostingRegressor()

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
        
