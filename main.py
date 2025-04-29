from sklearn.impute import SimpleImputer
from utils import load_config, print_results, save_results, load_custom_dataset, load_test_custom_dataset
import numpy as np
import pandas as pd
# sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import mplcursors
from models import stacking_reg,model_ADA,model_GB, model_KNN, model_R, model_RF, model_KRR, HIST_BOOST,model_log_linear, model_DYNAMIC_SELECTOR, showme



if __name__ == "__main__":
    # Load configs from "config.yaml"
    config = load_config()
    gs = False
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
        X1_test, X2_test, X1_meta_test, X2_meta_test, y1_test, y2_test = train_test_split(X_test, X_meta_test, y_test, train_size=0.5, random_state=42)

        model = model_DYNAMIC_SELECTOR(gridsearch1=gs, 
                                       personalized_pre_processing1=personalized_pre_processing, 
                                       X_train=X_train, 
                                       y_train=y_train,
                                       X_meta= X1_meta_test,
                                       X_test= X1_test,
                                       y_test= y1_test)
        y_pred, errors = model.predict(X2_test, y2_test, X2_meta_test)
        showme(errors)
        y_pred_ch, errors = model.predict(X_test_ch, np.zeros(len(X_test_ch)), Actual_meta)

        save_results(y_pred_ch)
        print_results(y2_test, y_pred)
        y_dif= np.abs(y2_test - y_pred)*100
    else:
        Train_meta, images, distances = load_custom_dataset(config, "train", preprocess_var, dyna=dyna)  
        Actual_meta, X_test_ch = load_test_custom_dataset(config, preprocess_var, dyna=dyna)

        
        X_train, X_test, X_meta_train, X_meta_test, y_train, y_test = train_test_split(images, Train_meta, distances, train_size=0.8, random_state=42)
        X1_test, X2_test, X1_meta_test, X2_meta_test, y1_test, y2_test = train_test_split(X_test, X_meta_test, y_test, train_size=0.5, random_state=42)

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
        #model = model_RF(gs, False, X_train, y_train)
        model = stacking_reg()
        #model = model_ADA(gs, True, X_train, y_train)
        #model = model_KRR(gs, personalized_pre_processing, X_train, y_train)
        #model = HIST_BOOST(gs, False, X_train, y_train)
        
        model.fit(X_train, y_train)
        
        # Prediction
        y_pred = model.predict(X1_test)
        print_results(y_test, y_pred)
        y_pred_ch = model.predict(X_test_ch)
        save_results(y_pred_ch)
        y_dif= np.abs(y1_test - y_pred)
        model2 = PoissonRegressor()
        imputer = SimpleImputer(strategy='mean')

        # Fit the imputer and transform the data
        X1_meta_test_imputed = imputer.fit_transform(X1_meta_test)
        model2.fit(X1_meta_test_imputed,y_dif)
        X2_meta_test_imputed = imputer.fit_transform(X2_meta_test)
        err_pred = model2.predict(X2_meta_test_imputed)
        y2_dif= np.abs(y2_test - model.predict(X2_test))
        showme(np.column_stack((y_dif, y2_dif)))
        showme(np.column_stack((y2_dif, err_pred)))
        print_results(y2_dif, err_pred)
    # Accuracy and saving

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