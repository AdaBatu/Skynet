from utils import load_config, load_dataset, load_test_dataset, print_results, save_results, load_custom_dataset, load_test_custom_dataset
import numpy as np
import pandas as pd
# sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from models import model_GB, model_KNN, model_R, model_RF, model_KRR
import mplcursors


if __name__ == "__main__":
    # Load configs from "config.yaml"
    config = load_config()
    gs = False
    personalized_pre_processing = True  # Set to False to use traditional approach
    preprocess_var = 5   #0 for black/white // 1 for only rgb // 2 for only edges // 3 for hog+edges // 4 for contour // 5 for LAB //6 for extreme things   

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

        images, distances = load_custom_dataset(config, "train", preprocess_var)  
        X_test_ch = load_test_custom_dataset(config, preprocess_var)
        
        X_train, X_test, y_train, y_test = train_test_split(images, distances)

        # Model - pipeline approach
        """model = model_KNN(
            personalized_pre_processing = True, 
            gridsearch=gs,
            config=config,
            X_train=X_train_df,
            y_train=y_train
        )
        """
        model = model_KNN(gs, False, X_train, y_train)
        model.fit(X_train, y_train)
        
        # Prediction
        y_pred = model.predict(X_test)
        y_pred_ch = model.predict(X_test_ch)

    # Accuracy and saving
    print_results(y_test, y_pred)
    y_pred = model.predict(X_test)
    y_dif= (y_test - y_pred)*100
    plt.hist(y_dif,density=False, color='skyblue', edgecolor='black')
    mplcursors.cursor(hover=True)
    plt.show()
    # Final training on all data
    if not personalized_pre_processing:
        model.fit(images, distances)
        save_results(model.predict(X_test_ch))
    else:
        model.fit(images, distances)
        save_results(model.predict(X_test_ch))





"""        print("Using pipeline approach with built-in loading")
        labels = pd.read_csv(config["data_dir"] / "train_labels.csv")
        
        # Split the labels DataFrame
        X_train_df, X_test_df, y_train, y_test = train_test_split(
            labels[['ID']],  # Just the ID column for image paths
            labels["distance"]
        )"""