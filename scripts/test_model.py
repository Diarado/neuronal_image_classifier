import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model # type: ignore
from scripts.preprocess import preprocess_image

def preprocess_and_prepare(image_path):
    labeled_image, extracted_features = preprocess_image(image_path)
    
    # Ensure the image has a channel dimension (expand if grayscale)
    if len(labeled_image.shape) == 2:
        labeled_image = np.expand_dims(labeled_image, axis=-1)  # Add channel dimension
    
    # If the image has 1 channel, duplicate it to create 3 channels
    if labeled_image.shape[-1] == 1:
        labeled_image = np.repeat(labeled_image, 3, axis=-1)  # Convert 1-channel grayscale to 3-channel

    # Normalize the image data
    labeled_image = labeled_image.astype('float32') / 255.0
    labeled_image = np.expand_dims(labeled_image, axis=0)  # Add batch dimension

    # Ensure extracted features are converted to a NumPy array and have the correct shape
    extracted_features = np.array(extracted_features, dtype=np.float32)
    
    extracted_features = np.expand_dims(extracted_features, axis=0)  # Add batch dimension

    return labeled_image, extracted_features

def test_model(image_dir, model_path='models/classifier_model.keras', output_csv='predictions.csv'):
    model = load_model(model_path)
    
    all_files = os.listdir(image_dir)
    image_paths = [os.path.join(image_dir, file) for file in all_files if file.endswith('.tif')]
    results = []

    for image_path in image_paths:
        labeled_image, extracted_features = preprocess_and_prepare(image_path)
        is_dead = bool(extracted_features[0][0])  # Extract the 'is_dead' feature
        peeling_degree = extracted_features[0][1]
        print(f"Shape of labeled_image: {labeled_image.shape}")
        print(f"Shape of extracted_features: {extracted_features.shape}")

        # Ensure the input sizes match what the model expects
        predictions = model.predict([labeled_image, extracted_features])
        
        # Convert predictions to the most likely class (1, 2, or 3)
        predictions_contamination = np.argmax(predictions[1], axis=1)[0] + 1
        predictions_density = np.argmax(predictions[2], axis=1)[0] + 1

        results.append((os.path.basename(image_path), int(peeling_degree), predictions_contamination, predictions_density, is_dead))
    
    if results:
        df = pd.DataFrame(results, columns=['Image', 'Peeling', 'Contamination', 'Cell Density', 'Empty/Dead'])
        df.to_csv(output_csv, index=False)
        print(f"Predictions saved to {output_csv}")
    else:
        print("No results to save.")

if __name__ == "__main__":
    test_model('data/images')




