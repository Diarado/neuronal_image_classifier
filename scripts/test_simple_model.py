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
    Extract features from images using a pre-trained ResNet50 model.
    Ensure the input is a batch of images before passing to the model.
    """
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(512, 512, 3))
    
    # Print the shape before adding the batch dimension
    print(f"Shape of input before adding batch dimension (if needed): {X.shape}")

    # If X is a single image (512, 512, 3), add a batch dimension to make it (1, 512, 512, 3)
    if len(X.shape) == 3:
        X = np.expand_dims(X, axis=0)
    
    # Print the shape after adding the batch dimension
    print(f"Shape of input after ensuring batch dimension: {X.shape}")

    # Preprocess input and extract features
    features = base_model.predict(preprocess_input(X))
    
    # Print the shape of the extracted features
    print(f"Shape of extracted features: {features.shape}")
    
    return features

def ensure_rgb(X):
    """
    Ensures the input images have 3 channels (RGB).
    If the image is grayscale (single-channel), it duplicates the channel to create an RGB image.
    """
    print(f"Initial shape before ensuring RGB: {X.shape}")
    
    # Check if the image is grayscale (shape should be 2D: height x width)
    if len(X.shape) == 2:  # If the image is grayscale with shape (height, width)
        X = np.expand_dims(X, axis=-1)  # Add a channel dimension (height, width, 1)
    
    # If the image has only 1 channel, repeat the channel to make it RGB
    if X.shape[-1] == 1:  # If the image has 1 channel
        X = np.repeat(X, 3, axis=-1)  # Duplicate to create 3 channels
    
    # Print the shape after ensuring RGB
    print(f"Shape after ensuring RGB: {X.shape}")
    
    return X

def preprocess_and_prepare(image_path):
    """
    Prepare the image for ResNet50 feature extraction and extract manual features.
    """
    # Extract the manually extracted features from the image
    labeled_image, extracted_features = preprocess_image(image_path)  # Assuming this function exists

    # Ensure the image is in the right shape for ResNet feature extraction
    labeled_image = ensure_rgb(labeled_image)

    # Normalize the image data (0 to 1) before passing to ResNet50
    labeled_image = labeled_image.astype('float32') / 255.0

    # Print the shape of the image after normalization
    print(f"Shape of image after normalization: {labeled_image.shape}")

    # Ensure extracted features are in the correct shape (add batch dimension)
    extracted_features = np.array(extracted_features, dtype=np.float32)
    extracted_features = np.expand_dims(extracted_features, axis=0)  # Add batch dimension to match image_features

    # Print the shape of extracted manual features
    print(f"Shape of extracted manual features: {extracted_features.shape}")

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

    # Load the saved scaler (as used during training)
    scaler = np.load(scaler_path, allow_pickle=True).item()

    # Get all image files in the directory
    all_files = os.listdir(image_dir)
    image_paths = [os.path.join(image_dir, file) for file in all_files if file.endswith('.tif')]
    results = []

    # Process each image and make predictions
    for image_path in image_paths:
        print(f"Processing image: {image_path}")
        
        # Extract image features and manual features for the current image
        labeled_image, extracted_features = preprocess_and_prepare(image_path)

        # Extract image features using ResNet50 (or any other feature extractor)
        image_features = extract_image_features(labeled_image)

        # Combine image features and manually extracted features
        combined_features = np.concatenate([image_features, extracted_features], axis=1)

        # Print the shape of combined features before scaling
        print(f"Shape of combined features before scaling: {combined_features.shape}")

        # Scale the combined features using the saved scaler
        combined_features = scaler.transform(combined_features)  # Scale features

        # Print the shape of combined features after scaling
        print(f"Shape of combined features after scaling: {combined_features.shape}")

        # Prepare data for XGBoost (using DMatrix)
        dmatrix = xgb.DMatrix(combined_features)

        # Predict using XGBoost for each output (Peeling, Contamination, and Density)
        peeling_pred = peeling_model.predict(dmatrix)
        contamination_pred = contamination_model.predict(dmatrix)
        density_pred = density_model.predict(dmatrix)

        # The model predicts in the range [0, 1, 2], so we add 1 to get back to [1, 2, 3]
        peeling_pred_class = int(peeling_pred[0]) + 1
        contamination_pred_class = int(contamination_pred[0]) + 1
        density_pred_class = int(density_pred[0]) + 1

        # Dead or empty can be directly determined by extracted features 
        dead_class = bool(extracted_features[0][0])

        # Append the results for this image
        results.append((os.path.basename(image_path), peeling_pred_class, contamination_pred_class, density_pred_class, dead_class))

    # Save the predictions to a CSV file
    if results:
        df = pd.DataFrame(results, columns=['Image', 'Peeling', 'Contamination', 'Cell Density', 'Dead/Empty'])
        df.to_csv(output_csv, index=False)
        print(f"Predictions saved to {output_csv}")
    else:
        print("No results to save.")

if __name__ == "__main__":
    # Specify the image directory for testing
    test_xgboost('data/images')


