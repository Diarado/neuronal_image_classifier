import os
from joblib import load
from scripts.preprocess import preprocess_image
import pandas as pd
import numpy as np

def preprocess_and_flatten(image_path):
    binary_image, is_dead, density = preprocess_image(image_path)
    return binary_image.flatten(), is_dead, density

def test_model(image_dir, model_path='models/classifier_model.pkl', scaler_path='models/scaler.pkl', output_csv='predictions.csv'):

    model = load(model_path)
    scaler = load(scaler_path)
    image_dir = os.path.join(image_dir, 'images')
    # List all files in the images directory for debugging
    all_files = os.listdir(image_dir) 
    # Gather all .tif files
    image_paths = [os.path.join(image_dir, file) for file in all_files if file.endswith('.tif')]
    results = []
    
    for image_path in image_paths:
        labeled_image, is_dead, cell_density = preprocess_and_flatten(image_path)
        num_peeling_pixels = np.sum(labeled_image == 3)
        num_neuron_cells = np.sum(labeled_image == 2)
        
        # Create feature vector
        feature_vector = [num_peeling_pixels, num_neuron_cells, cell_density]
        feature_vector = scaler.transform([feature_vector])  # Normalize the feature vector

        prediction = model.predict(feature_vector)
        results.append((os.path.basename(image_path), prediction[0], cell_density))
    
    if results:
        df = pd.DataFrame(results, columns=['Image', 'Prediction', 'Empty/Dead'])
        df.to_csv(output_csv, index=False)
        print(f"Predictions saved to {output_csv}")
    else:
        print("No results to save.")

if __name__ == "__main__":
    test_model('data')

