import os
from joblib import load
from scripts.preprocess import preprocess_image
import pandas as pd
import numpy as np

def preprocess_and_flatten(image_path):
    binary_image, is_dead, peeling_degree, density = preprocess_image(image_path)
    return binary_image.flatten(), is_dead, peeling_degree, density

def test_model(image_dir, model_path='models/classifier_model.pkl', scaler_path='models/scaler.pkl', output_csv='predictions.csv'):

    model = load(model_path)
    scaler = load(scaler_path)
    image_dir = os.path.join(image_dir, 'images')
    all_files = os.listdir(image_dir)
    image_paths = [os.path.join(image_dir, file) for file in all_files if file.endswith('.tif')]
    results = []

    for image_path in image_paths:
        labeled_image, is_dead, peeling_degree, cell_density = preprocess_and_flatten(image_path)
        num_peeling_pixels = np.sum(labeled_image == 3)
        num_neuron_cells = np.sum(labeled_image == 2)
        
        # Create feature vector
        feature_vector = [num_peeling_pixels, num_neuron_cells, cell_density]
        feature_vector = scaler.transform([feature_vector])  # Normalize the feature vector

        # Predict all four scores
        predictions = model.predict(feature_vector)[0]  # Expecting the model to return four scores
        if peeling_degree != 3:
            predictions[0] = peeling_degree
        # Append all four predicted scores
        results.append((os.path.basename(image_path), predictions[0], predictions[1], predictions[2], is_dead))
    
    if results:
        df = pd.DataFrame(results, columns=['Image', 'Peeling', 'Contamination', 'Cell Density', 'Empty/Dead'])
        df.to_csv(output_csv, index=False)
        print(f"Predictions saved to {output_csv}")
    else:
        print("No results to save.")

if __name__ == "__main__":
    test_model('data')

