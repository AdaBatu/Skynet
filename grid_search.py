import json
import os
from sklearn.model_selection import GridSearchCV

def grid_search_model(model, param_grid, X_train=None, y_train=None, cv=5, scoring='neg_mean_squared_error', save_params=True, save_dir="saved_params"):
    # Fit the grid search
    grid = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=2  # Higher numbers = more verbose
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
