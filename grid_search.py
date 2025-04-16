import json
import os
from sklearn.model_selection import GridSearchCV, ParameterGrid, cross_val_score
import numpy as np
from sklearn.base import clone
from tqdm import tqdm

def grid_search_model(model, param_grid, X_train=None, y_train=None, cv=5, scoring='neg_mean_squared_error', save_params=True, save_dir="saved_params"):
    # Fit the grid search
    grid = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1  # Higher numbers = more verbose
    )
    print(f"Starting grid search with {len(param_grid)} parameter combinations x {cv} folds...")

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    best_params = grid.best_params_

    model_name = type(model).__name__
    print(f"Best parameteSrs for {model_name}:", best_params)

    # Save parameters if enabled
    if save_params:
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"best_params_{model_name}.json")
        with open(filename, "w") as f:
            json.dump(best_params, f, indent=4)

    return best_model, best_params



def grid_search_model_PBs(model, param_grid, X_train=None, y_train=None, cv=5,
                      scoring='neg_mean_squared_error', save_params=True, save_dir="saved_params"):
    param_combinations = list(ParameterGrid(param_grid))
    total = len(param_combinations) * cv
    print(f"Starting grid search with {len(param_combinations)} parameter combinations x {cv} folds...")

    # tqdm progress wrapper
    with tqdm(total=total, desc="GridSearch Progress") as pbar:
        class ProgressGridSearchCV(GridSearchCV):
            def _run_search(self, evaluate_candidates):
                def wrapped_evaluate(candidate_params):
                    out = evaluate_candidates(candidate_params)
                    pbar.update(len(candidate_params) * cv)
                    return out
                return super()._run_search(wrapped_evaluate)

        grid = ProgressGridSearchCV(
            model,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=0
        )

        grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    best_params = grid.best_params_

    model_name = type(model).__name__
    print(f"Best parameters for {model_name}:", best_params)

    if save_params:
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"best_params_{model_name}.json")
        with open(filename, "w") as f:
            json.dump(best_params, f, indent=4)

    return best_model, best_params


def grid_search_model_PB(model, param_grid, X_train=None, y_train=None, cv=5,
                      scoring='neg_mean_squared_error', save_params=True, save_dir="saved_params"):

    param_combinations = list(ParameterGrid(param_grid))
    print(f"Starting grid search with {len(param_combinations)} parameter combinations x {cv} folds...")

    best_score = -np.inf
    best_params = None
    best_model = None

    for params in tqdm(param_combinations, desc="GridSearch Progress"):
        current_model = clone(model)
        current_model.set_params(**params)

        try:
            scores = cross_val_score(current_model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
            mean_score = scores.mean()
        except Exception as e:
            print(f"Failed for params {params}: {e}")
            continue

        if mean_score > best_score:
            best_score = mean_score
            best_params = params
            best_model = clone(current_model)

    best_model.fit(X_train, y_train)
    model_name = type(model).__name__
    print(f"Best parameters for {model_name}:", best_params)

    if save_params and best_params is not None:
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"best_params_{model_name}.json")
        with open(filename, "w") as f:
            json.dump(best_params, f, indent=4)

    return best_model, best_params