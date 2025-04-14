import json
import os
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from sklearn.base import clone

class TQDMGridSearchCV(GridSearchCV):
    def fit(self, X, y=None, **fit_params):
        # Estimate total number of fits
        n_candidates = len(self.param_grid)
        total_fits = self.cv * n_candidates

        with tqdm(total=total_fits, desc="Grid Search Progress", unit="fit") as pbar:
            original_fit = self._run_search

            def wrapped_run_search(evaluate_candidates):
                def wrapped_evaluate(candidate_params):
                    pbar.update(self.cv)
                    results = evaluate_candidates(candidate_params)
                    pbar.update(len(candidate_params) * self.cv)
                    return results
                original_fit(wrapped_evaluate)

            self._run_search = wrapped_run_search
            return super().fit(X, y, **fit_params)

def grid_search_model(model, param_grid, X_train=None, y_train=None, cv=5, scoring='neg_mean_squared_error', save_params=True, save_dir="saved_params"):
    grid = TQDMGridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    best_params = grid.best_params_

    model_name = type(model).__name__
    print(f"\n✅ Best parameters for {model_name}: {best_params}")

    if save_params:
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"best_params_{model_name}.json")
        with open(filename, "w") as f:
            json.dump(best_params, f, indent=4)

    return best_model, best_params
