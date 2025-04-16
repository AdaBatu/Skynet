from utils import load_config, load_dataset, load_test_dataset, print_results, save_results
import numpy as np
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

    # Load dataset: images and corresponding minimum distance values
    images, distances = load_dataset(config)
    X_test_ch = load_test_dataset(config)
    print(f"[INFO]: Dataset loaded with {len(images)} samples.")

    #train test split

    X = images
    y = distances
    X_train, X_test, y_train, y_test = train_test_split(X,y)

    #model
    gs = False
    personalized_pre_processing = False
    model = model_KNN(gs,personalized_pre_processing ,X_train, y_train) #RF, GB, KNN, R, KRR
    if gs == False:
        model.fit(X_train, y_train)
    #prediction

    y_pred = model.predict(X_test)
    y_pred_ch = model.predict(X_test_ch)
    #accuracy
    print_results(y_test, y_pred)
    #safe this .-.
    save_results(y_pred_ch)

