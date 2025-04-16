from utils import load_config, load_dataset, load_test_dataset, print_results, save_results
import numpy as np
import pandas as pd
# sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import GridSearchCV

from models import model_GB, model_KNN, model_R, model_RF, model_KRR


if __name__ == "__main__":
    # Load configs from "config.yaml"
    config = load_config()
    gs = False
    personalized_pre_processing = False  # Set to False to use traditional approach

    if not personalized_pre_processing:
        print("Using traditional approach with pre-processed arrays")
        # Load dataset: images and corresponding minimum distance values
        images, distances = load_dataset(config)
        X_test_ch = load_test_dataset(config)
        print(f"[INFO]: Dataset loaded with {len(images)} samples.")

        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(images, distances)

        # Model - traditional approach
        model = model_KNN(gs, personalized_pre_processing, X_train, y_train)
        model.fit(X_train, y_train)
        
        # Prediction
        y_pred = model.predict(X_test)
        y_pred_ch = model.predict(X_test_ch)

    else:
        print("Using pipeline approach with built-in loading")
        labels = pd.read_csv(config["data_dir"] / "train_labels.csv")
        
        # Split the labels DataFrame
        X_train_df, X_test_df, y_train, y_test = train_test_split(
            labels[['ID']],  # Just the ID column for image paths
            labels["distance"]
        )
        
        # Model - pipeline approach
        model = model_KNN(
            personalized_pre_processing = True, 
            gridsearch=gs,
            config=config,
            X_train=X_train_df,
            y_train=y_train
        )
        
        # Need to prepare test data in same format
        # For pipeline approach, X_test should be DataFrame with 'ID' column
        y_pred = model.predict(X_test_df)
        
        # For challenge data you'll need to prepare similar DataFrame
        challenge_labels = pd.read_csv(config["data_dir"] / "sample_submission.csv")
        X_test_ch_df = challenge_labels[['ID']]
        y_pred_ch = model.predict(X_test_ch_df)

    # Accuracy and saving
    print_results(y_test, y_pred)
    
    # Final training on all data
    if not personalized_pre_processing:
        model.fit(images, distances)
        save_results(model.predict(X_test_ch))
    else:
        model.fit(labels[['ID']], labels["distance"])
        save_results(model.predict(X_test_ch_df))