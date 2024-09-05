import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from scripts.preprocess import preprocess_image

def extract_image_features(X):
    """
    Extract features from images using a pre-trained ResNet50 model (or any other CNN).
    """
    # Load the pre-trained ResNet50 model (same as used in training)
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(512, 512, 3))
    features = base_model.predict(preprocess_input(X))
    return features

def preprocess_and_prepare(image_path):
    # Extract the manually extracted features from the image
    labeled_image, extracted_features = preprocess_image(image_path)  # Assuming this function exists

    # Ensure the image is in the right shape for ResNet feature extraction
    if len(labeled_image.shape) == 2:  # If the image is grayscale (512, 512)
        labeled_image = np.expand_dims(labeled_image, axis=-1)  # Add channel dimension (512, 512, 1)
    
    if labeled_image.shape[-1] == 1:  # If the image has 1 channel (grayscale), convert to 3 channels
        labeled_image = np.repeat(labeled_image, 3, axis=-1)  # Convert (512, 512, 1) to (512, 512, 3)

    # Normalize the image data
    labeled_image = labeled_image.astype('float32') / 255.0
    labeled_image = np.expand_dims(labeled_image, axis=0)  # Add batch dimension (1, 512, 512, 3)

    # Ensure extracted features are in the correct shape (if necessary)
    extracted_features = np.array(extracted_features, dtype=np.float32)
    extracted_features = np.expand_dims(extracted_features, axis=0)  # Add batch dimension to match image_features

    return labeled_image, extracted_features

def test_xgboost(image_dir, peeling_model_path='models/xgb_peeling_model.json', 
                 contamination_model_path='models/xgb_contamination_model.json', 
                 density_model_path='models/xgb_density_model.json', 
                 scaler_path='models/xgb_scaler.npy', output_csv='xgb_predictions.csv'):
    
    # Load the saved XGBoost models and scaler
    peeling_model = xgb.Booster()
    peeling_model.load_model(peeling_model_path)

    contamination_model = xgb.Booster()
    contamination_model.load_model(contamination_model_path)

    density_model = xgb.Booster()
    density_model.load_model(density_model_path)

    scaler = np.load(scaler_path, allow_pickle=True).item()

    all_files = os.listdir(image_dir)
    image_paths = [os.path.join(image_dir, file) for file in all_files if file.endswith('.tif')]
    results = []

    for image_path in image_paths:
        # Extract features for the current image
        labeled_image, extracted_features = preprocess_and_prepare(image_path)

        # Extract image features using ResNet50 (or any other feature extractor)
        image_features = extract_image_features(labeled_image)

        # Combine image features and manually extracted features
        combined_features = np.concatenate([image_features, extracted_features], axis=1)

        # Scale combined features
        combined_features = scaler.transform(combined_features)  # Scale features

        # Prepare data for XGBoost (using DMatrix)
        dmatrix = xgb.DMatrix(combined_features)

        # Predict using XGBoost for each output
        peeling_pred = peeling_model.predict(dmatrix)
        contamination_pred = contamination_model.predict(dmatrix)
        density_pred = density_model.predict(dmatrix)

        # The model predicts in the range [0, 1, 2], so we add 1 to get back to [1, 2, 3]
        peeling_pred_class = int(peeling_pred[0]) + 1
        contamination_pred_class = int(contamination_pred[0]) + 1
        density_pred_class = int(density_pred[0]) + 1

        # Append the results
        results.append((os.path.basename(image_path), peeling_pred_class, contamination_pred_class, density_pred_class))

    # Save the predictions to a CSV file
    if results:
        df = pd.DataFrame(results, columns=['Image', 'Peeling', 'Contamination', 'Cell Density'])
        df.to_csv(output_csv, index=False)
        print(f"Predictions saved to {output_csv}")
    else:
        print("No results to save.")

if __name__ == "__main__":
    test_xgboost('data/images')

