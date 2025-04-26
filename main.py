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
from models import model_GB, model_KNN, model_R, model_RF, model_KRR, HIST_BOOST,model_log_linear, model_DYNAMIC_SELECTOR
import mplcursors


if __name__ == "__main__":
    # Load configs from "config.yaml"
    config = load_config()
    gs = 2
    dyna = False
    personalized_pre_processing = True  # Set to False to use traditional approach
    preprocess_var = 5   #0 for     black/white // 1 for only rgb // 2 for only edges // 3 for hog+edges // 4 for contour // 5 for LAB //6 for extreme things   

    if dyna:
        print("Using traditional approach with pre-processed arrays")
        # Load dataset: images and corresponding minimum distance values
        Train_meta, images, distances = load_custom_dataset(config, "train", preprocess_var, dyna=dyna)  
        Actual_meta, X_test_ch = load_test_custom_dataset(config, preprocess_var, dyna=dyna)

        print(f"[INFO]: Dataset loaded with {len(images)} samples.")

        # Train test split
        X_train, X_test, X_meta_train, X_meta_test, y_train, y_test = train_test_split(images, Train_meta, distances, train_size=0.8, random_state=42)
        model = model_DYNAMIC_SELECTOR(gridsearch1=gs, 
                                       personalized_pre_processing1=personalized_pre_processing, 
                                       X_train=X_train, 
                                       y_train=y_train,
                                       X_meta= X_meta_test,
                                       X_test= X_test,
                                       y_test= y_test)
        y_pred, errors = model.predict(X_test, y_test, X_meta_test)
        plt.hist(errors[:,0],density=False, color='skyblue', bins=20, edgecolor='black')
        plt.hist(errors[:,1],density=False, color='red', bins=20, edgecolor='black')
        plt.hist(errors[:,2],density=False, color='black', bins=20, edgecolor='black')
        mplcursors.cursor(hover=True)
        plt.show()

        y_pred_ch, errors = model.predict(X_test_ch, np.zeros(len(X_test_ch)), Actual_meta)


    else:
        _, images, distances = load_custom_dataset(config, "train", preprocess_var,dyna)  
        _, X_test_ch = load_test_custom_dataset(config, preprocess_var,dyna)
        
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
        #model = model_KNN(gs, personalized_pre_processing, X_train, y_train)
        model = model_KRR(gs, personalized_pre_processing, X_train, y_train)
        #model = HIST_BOOST(gs, personalized_pre_processing, X_train, y_train)
        
        model.fit(X_train, y_train)
        
        # Prediction
        y_pred = model.predict(X_test)
        y_pred_ch = model.predict(X_test_ch)

    # Accuracy and saving
    print_results(y_test, y_pred)
    y_dif= np.abs(y_test - y_pred)*100
    plt.hist(y_dif,density=False, color='skyblue', bins=20, edgecolor='black')
    mplcursors.cursor(hover=True)
    plt.show()
    # Final training on all data
    """if not personalized_pre_processing:
        model.fit(images, distances)
        save_results(model.predict(X_test_ch))
    else:
        model.fit(images, distances)
        save_results(model.predict(X_test_ch))"""





"""        print("Using pipeline approach with built-in loading")
        labels = pd.read_csv(config["data_dir"] / "train_labels.csv")
        
        # Split the labels DataFrame
        X_train_df, X_test_df, y_train, y_test = train_test_split(
            labels[['ID']],  # Just the ID column for image paths
            labels["distance"]
        )"""