from utils import load_config, load_dataset, load_test_dataset, print_results, save_results
import numpy as np
# sklearn imports
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import GridSearchCV
# SVRs are not allowed in this project.

def grid_search(model):
    param_grid = {
    'n_estimators': [200, 250, 300, 350, 400],
    'max_depth': [30, 40, 50, 60],
    'min_samples_split': [1, 3, 5],
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
    return {'max_depth': 30, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}


if __name__ == "__main__":
    # Load configs from "config.yaml"
    config = load_config()
    # Load dataset: images and corresponding minimum distance values
    images, distances = load_dataset(config)
    wololo = load_test_dataset(config)
    print(f"[INFO]: Dataset loaded with {len(images)} samples.")

    # TODO: Your implementation starts here

    # possible preprocessing steps ... training the model

    # Evaluation
    # print_results(gt, pred)

    # Save the results
    # save_results(test_pred)

    

    #train test split
    
    X = images
    la = wololo
    y = distances
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    
    
    #model
    model = RandomForestRegressor(random_state=42)
    print(model)
    #gridsearch
    if input("Grid Search ?") == "yes":
        model = grid_search(model)
    else:
        best_params = load_par(None)  # you don't need the 'model' arg here unless you want to customize
        model = RandomForestRegressor(**best_params, random_state=42)
        model.fit(X_train, y_train)  # <--- this is what was missing
    #prediction
    joblib.dump(model, "random_forest_model.pkl")
    if input("Predicition ?") == "yes":
        #y_pred = model.predict(X_test)
        pred_end = model.predict(la)
        #accuracy
        #print_results(y_test, y_pred)
        save_results(pred_end)

#hallo
