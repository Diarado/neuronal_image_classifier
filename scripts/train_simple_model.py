import os
import numpy as np
import pandas as pd
import gc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from tensorflow.keras.applications import ResNet50
from scripts.parse_csv_to_dict import parse_csv_to_dict
from scripts.link_images_to_scores import link_images_to_scores
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV

def extract_image_features(X):
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(512, 512, 3))
    features = base_model.predict(X)
    return features

def train_xgboost(X, y, extracted_features, use_image_features):
    # If we want to extract features from the images, do it here
    if use_image_features:
        print("Extracting image features using ResNet...")
        X_features = extract_image_features(X)  # Extract features from images using ResNet
        combined_features = np.concatenate([X_features, extracted_features], axis=1)  # Combine image features and manual features
    else:
        combined_features = extracted_features

    y -= 1
    # Split the data into training and testing sets for each output
    X_train, X_test, y_train_peeling, y_test_peeling = train_test_split(combined_features, y[:, 0], test_size=0.2, random_state=173)
    X_train, X_test, y_train_contamination, y_test_contamination = train_test_split(combined_features, y[:, 1], test_size=0.2, random_state=956)
    X_train, X_test, y_train_density, y_test_density = train_test_split(combined_features, y[:, 2], test_size=0.2, random_state=889)

    # Standardize the feature set
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train separate XGBoost classifiers for each output
    print("Training XGBoost for peeling_output...")
    peeling_model = XGBClassifier(objective='multi:softmax', num_class=3)
    peeling_model.fit(X_train, y_train_peeling)

    print("Training XGBoost for contamination_output...")
    contamination_model = XGBClassifier(objective='multi:softmax', num_class=3)
    contamination_model.fit(X_train, y_train_contamination)

    print("Training XGBoost for density_output...")
    density_model = XGBClassifier(objective='multi:softmax', num_class=3)
    density_model.fit(X_train, y_train_density)

    # Make predictions and calculate accuracy for each output
    peeling_pred = peeling_model.predict(X_test)
    contamination_pred = contamination_model.predict(X_test)
    density_pred = density_model.predict(X_test)

    peeling_accuracy = accuracy_score(y_test_peeling, peeling_pred)
    contamination_accuracy = accuracy_score(y_test_contamination, contamination_pred)
    print("actual contamination: " + str(y_test_contamination))
    print("predicted contamination: " + str(contamination_pred))
    density_accuracy = accuracy_score(y_test_density, density_pred)
    print("actual density: " + str(y_test_density))
    print("predicted desnity: " + str(density_pred))

    print(f"Peeling Test Accuracy: {peeling_accuracy}")
    print(f"Contamination Test Accuracy: {contamination_accuracy}")
    print(f"Density Test Accuracy: {density_accuracy}")

    # Save the models and scaler
    peeling_model.save_model('models/xgb_peeling_model.json')
    contamination_model.save_model('models/xgb_contamination_model.json')
    density_model.save_model('models/xgb_density_model.json')
    np.save('models/xgb_scaler.npy', scaler)

    return peeling_model, contamination_model, density_model, scaler

def load_data(image_dir_lst, csv_lst):
    X, y, extracted_feature_lst = [], [], []

    for image_dir, csv_file in zip(image_dir_lst, csv_lst):
        csv_dict = parse_csv_to_dict(csv_file)  # You may want to replace this function if it's custom
        X_part, y_part, extracted_feature_lst_part = link_images_to_scores(image_dir, csv_dict) 

        X.extend(X_part)
        y.extend(y_part)
        extracted_feature_lst.extend(extracted_feature_lst_part)

        del X_part, y_part, extracted_feature_lst_part
        gc.collect()

    X = np.array(X).astype('float32')
    y = np.array(y)
    extracted_feature_lst = np.array(extracted_feature_lst)

    return X, y, extracted_feature_lst

def ensure_rgb(X):
    """
    Ensures the input images have 3 channels (RGB).
    If the image is grayscale (single-channel), it duplicates the channel to create an RGB image.
    """
    if len(X.shape) == 3:  # If the image is grayscale with shape (num_images, height, width)
        X = np.expand_dims(X, axis=-1)  # Add a channel dimension (num_images, height, width, 1)
    
    if X.shape[-1] == 1:  # If the image has 1 channel, duplicate it to create 3 channels (RGB)
        X = np.repeat(X, 3, axis=-1)
    
    return X



# Hyperparameter search space for XGBClassifier
param_dist = {
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [50, 100, 150, 200],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2],
    'reg_lambda': [0.5, 1, 1.5],
}

def cross_validate_and_tune_model(X, y, extracted_features, use_image_features, n_splits=5):
    if use_image_features:
        print("Extracting image features using ResNet...")
        X_features = extract_image_features(X)  # Extract features from images using ResNet
        combined_features = np.concatenate([X_features, extracted_features], axis=1)  # Combine image features and manual features
    else:
        combined_features = extracted_features

    y -= 1  # Adjust target labels if necessary (assuming 1-based indexing)

    # Standardize the feature set
    scaler = StandardScaler()
    combined_features = scaler.fit_transform(combined_features)

    # Stratified K-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Store best models and accuracies for each random state
    best_models = {
        'peeling': None,
        'contamination': None,
        'density': None
    }
    best_accuracies = {
        'peeling': 0,
        'contamination': 0,
        'density': 0
    }

    random_states = [123, 456, 789, 66]
    
    for state in random_states:
        print(f"Tuning hyperparameters with random_state={state}...")

        # Prepare individual models for peeling, contamination, and density
        peeling_model = XGBClassifier(objective='multi:softmax', num_class=3)
        contamination_model = XGBClassifier(objective='multi:softmax', num_class=3)
        density_model = XGBClassifier(objective='multi:softmax', num_class=3)

        # Randomized search for hyperparameter tuning
        peeling_search = RandomizedSearchCV(peeling_model, param_distributions=param_dist, n_iter=10, scoring='accuracy', cv=3, random_state=state)
        contamination_search = RandomizedSearchCV(contamination_model, param_distributions=param_dist, n_iter=10, scoring='accuracy', cv=3, random_state=state)
        density_search = RandomizedSearchCV(density_model, param_distributions=param_dist, n_iter=10, scoring='accuracy', cv=3, random_state=state)

        # Accumulate accuracies for all folds
        peeling_accuracies, contamination_accuracies, density_accuracies = [], [], []

        # Now evaluate each model on stratified KFold test set
        for train_idx, test_idx in skf.split(combined_features, y[:, 0]):
            X_train, X_test = combined_features[train_idx], combined_features[test_idx]
            y_train_peeling, y_test_peeling = y[train_idx, 0], y[test_idx, 0]
            y_train_contamination, y_test_contamination = y[train_idx, 1], y[test_idx, 1]
            y_train_density, y_test_density = y[train_idx, 2], y[test_idx, 2]

            # Fit the models using the train data in this fold
            peeling_search.fit(X_train, y_train_peeling)
            contamination_search.fit(X_train, y_train_contamination)
            density_search.fit(X_train, y_train_density)

            # Best models after search
            peeling_best_model = peeling_search.best_estimator_
            contamination_best_model = contamination_search.best_estimator_
            density_best_model = density_search.best_estimator_

            # Make predictions on the test data of this fold
            peeling_pred = peeling_best_model.predict(X_test)
            contamination_pred = contamination_best_model.predict(X_test)
            density_pred = density_best_model.predict(X_test)

            # Calculate accuracies for this fold
            peeling_accuracy = accuracy_score(y_test_peeling, peeling_pred)
            contamination_accuracy = accuracy_score(y_test_contamination, contamination_pred)
            density_accuracy = accuracy_score(y_test_density, density_pred)

            # Append fold accuracy to respective lists
            peeling_accuracies.append(peeling_accuracy)
            contamination_accuracies.append(contamination_accuracy)
            density_accuracies.append(density_accuracy)

        # Calculate mean accuracies across all folds for the current random state
        mean_peeling_accuracy = np.mean(peeling_accuracies)
        mean_contamination_accuracy = np.mean(contamination_accuracies)
        mean_density_accuracy = np.mean(density_accuracies)

        print(f"Mean Peeling Accuracy with random_state={state}: {mean_peeling_accuracy}")
        print(f"Mean Contamination Accuracy with random_state={state}: {mean_contamination_accuracy}")
        print(f"Mean Density Accuracy with random_state={state}: {mean_density_accuracy}")

        # Track the best models
        if mean_peeling_accuracy > best_accuracies['peeling']:
            best_accuracies['peeling'] = mean_peeling_accuracy
            best_models['peeling'] = peeling_best_model

        if mean_contamination_accuracy > best_accuracies['contamination']:
            best_accuracies['contamination'] = mean_contamination_accuracy
            best_models['contamination'] = contamination_best_model

        if mean_density_accuracy > best_accuracies['density']:
            best_accuracies['density'] = mean_density_accuracy
            best_models['density'] = density_best_model

    # After testing with all random states, print the best models
    print(f"Best Peeling Accuracy: {best_accuracies['peeling']}")
    print(f"Best Contamination Accuracy: {best_accuracies['contamination']}")
    print(f"Best Density Accuracy: {best_accuracies['density']}")

    return best_models['peeling'], best_models['contamination'], best_models['density'], scaler


if __name__ == "__main__":
    # Load the dataset
    rounds = ['round06']  # Example round names

    image_dir_lst = [f'train/{round}_images' for round in rounds]
    csv_lst = [f'train/scoring_{round}.csv' for round in rounds]

    X_path = f'train/X_{"_".join(rounds)}.npy'
    y_path = f'train/y_{"_".join(rounds)}.npy'
    features_path = f'train/features_{"_".join(rounds)}.npy'
    # Load data
    if os.path.exists(X_path) and os.path.exists(y_path) and os.path.exists(features_path):
        print("Loading labeled images and scores from disk...")
        X = np.load(X_path)
        X_rgb = ensure_rgb(X)

        y = np.load(y_path)
        extracted_feature_lst = np.load(features_path)
        #peeling_model, contamination_model, density_model, scaler = train_xgboost(X_rgb, y, extracted_feature_lst, True)
        peeling_model, contamination_model, density_model, scaler = cross_validate_and_tune_model(X_rgb, y, extracted_feature_lst, True)
    else:
        X, y, extracted_feature_lst = load_data(image_dir_lst, csv_lst)
        #peeling_model, contamination_model, density_model, scaler = train_xgboost(ensure_rgb(X), y, extracted_feature_lst, True)
        peeling_model, contamination_model, density_model, scaler = cross_validate_and_tune_model(ensure_rgb(X), y, extracted_feature_lst, True)
