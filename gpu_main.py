from utils import load_config, load_dataset, load_test_dataset, print_results, save_results
import numpy as np
# sklearn imports
import joblib
import cupy as cp
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import GridSearchCV
import xgboost as xgb  # Import XGBoost

def manual_grid_search(X_train, y_train, X_test, y_test):
    best_mae = float('inf')
    best_params = None
    best_model = None
    
    param_grid = {
        'n_estimators': [100, 200, 400, 1000],  # Number of boosting rounds (trees)
        'max_depth': [10, 30 , 50, 80],  # Maximum depth of trees
        'learning_rate': [0.01, 0.1, 0.2],  # Learning rate (controls contribution of each tree)
        'subsample': [0.8, 1.0],  # Fraction of samples used for each tree
    }

    X_train_cp = cp.array(X_train)
    y_train_cp = cp.array(y_train)
    X_test_cp = cp.array(X_test)

    for max_depth in param_grid['max_depth']:
        for learning_rate in param_grid['learning_rate']:
            for n_estimators in param_grid['n_estimators']:
                for subsample in param_grid['subsample']:
                    # Create the model with the current hyperparameters
                    model = xgb.XGBRegressor(
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        n_estimators=n_estimators,
                        subsample=subsample,
                        objective='reg:squarederror',
                        tree_method='hist',  # Use GPU
                        device='cuda',  # Specify the device to use
                        random_state=42
                    )
                    
                    # Convert to DMatrix (GPU compatible)
                    dtrain = xgb.DMatrix(X_train, label=y_train)  # DMatrix for training data
                    # Train the model
                    model.fit(X_train_cp, y_train_cp, verbose=False)
                    
                    # Predict and evaluate performance
                    y_pred = model.predict(X_test_cp)
                    mae = mean_absolute_error(y_test, y_pred)
                    print(f"MAE for params {max_depth}, {learning_rate}, {n_estimators}, {subsample}: {mae}")
                    
                    # If the performance is better, save the model
                    if mae < best_mae:
                        best_mae = mae
                        best_params = {
                            'max_depth': max_depth,
                            'learning_rate': learning_rate,
                            'n_estimators': n_estimators,
                            'subsample': subsample
                        }
                        best_model = model
    print(f"Best parameters: {best_params}")
    print(f"Best MAE: {best_mae}")
    return best_model



def grid_search(model):
    param_grid = {
        'n_estimators': [100, 200, 400, 1000],  # Number of boosting rounds (trees)
        'max_depth': [10, 30 , 50, 80],  # Maximum depth of trees
        'learning_rate': [0.01, 0.1, 0.2],  # Learning rate (controls contribution of each tree)
        'subsample': [0.8, 1.0],  # Fraction of samples used for each tree
    }

    scorer = make_scorer(mean_absolute_error, greater_is_better=False)  # MAE scoring
    model.set_params(tree_method='hist', device='cuda')
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,  # 5-fold cross-validation
        scoring=scorer,
        verbose=2
    )
    X_train_cp = cp.array(X_train)
    y_train_cp = cp.array(y_train)
    grid_search.fit(X_train_cp, y_train_cp)

    print("Best parameters found:")
    print(grid_search.best_params_)

    print("Best MAE:")
    print(-grid_search.best_score_)

    return grid_search.best_estimator_

def load_par():
    # Set default parameters for XGBoost
    return {'max_depth': 80, 'learning_rate': 0.1, 'n_estimators': 600, 'subsample': 0.8}


if __name__ == "__main__":
    # Load configs from "config.yaml"
    config = load_config()
    # Load dataset: images and corresponding minimum distance values
    images, distances = load_dataset(config)
    images_test = load_test_dataset(config)
    print(f"[INFO]: Dataset loaded with {len(images)} samples.")

    # TODO: Your implementation starts here

    # Train-test split
    X = images
    X_test = images_test
    y = distances
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # XGBoost model
    

    # Grid Search (if selected)
    if input("Grid Search? (yes/no) ") == "yes":
        model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, verbosity=2)  # Using XGBoost Regressor
        #model = manual_grid_search(X_train, y_train, X_test, y_test)
        model = grid_search(model)
    else:
        best_params = load_par()  # Load predefined best parameters
        model = xgb.XGBRegressor(**best_params, objective='reg:squarederror', random_state=42, verbosity=2)  # XGBoost Regressor with best params


    # Save the trained model
    X_train_cp1 = cp.array(X_train)
    print("loaded Xtrain to gpu")
    y_train_cp1 = cp.array(y_train)
    print("loaded Ytrain to gpu")
    

    model.fit(X_train_cp1, y_train_cp1)  # Train the model
    joblib.dump(model, "xgboost_model_preprocessed.pkl")

    # Prediction on test dat
    if input("Prediction? (yes/no) ") == "yes":
        X_test_cp_array = cp.array(X_test)

        pred_end = model.predict(X_test_cp_array)  # Predict on the test dataset
        # Optionally, evaluate with metrics (uncomment if needed)
        print_results(X_test, y_test)  # Evaluate with test data
        save_results(pred_end)  # Save the predictions
