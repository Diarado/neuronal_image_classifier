import os
import numpy as np
import pandas as pd
from joblib import load
from tensorflow.keras.models import load_model # type: ignore
from scripts.preprocess import preprocess_image
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.losses import MeanSquaredError # type: ignore


def preprocess_and_flatten(image_path):
    binary_image, is_dead, peeling_degree, density = preprocess_image(image_path)
    return binary_image.flatten(), is_dead, peeling_degree, density

def test_model(image_dir, model_path='models/classifier_model.keras', scaler_path='models/scaler.pkl', output_csv='predictions.csv'):
    mse = MeanSquaredError()
    model = load_model(model_path, custom_objects={'mse': mse})
    scaler = load(scaler_path)
    
    all_files = os.listdir(image_dir)
    image_paths = [os.path.join(image_dir, file) for file in all_files if file.endswith('.tif')]
    results = []

    for image_path in image_paths:
        labeled_image, is_dead, peeling_degree, cell_density = preprocess_and_flatten(image_path)
        num_peeling_pixels = np.sum(labeled_image == 3)
        num_neuron_cells = np.sum(labeled_image == 2)
        
        feature_vector = [num_peeling_pixels, num_neuron_cells, cell_density]
        feature_vector = scaler.transform([feature_vector])

        predictions = model.predict(feature_vector)[0]  
        # predictions = np.clip(predictions, 1, 3)
        if peeling_degree != 3:
            predictions[0] = peeling_degree
        
        results.append((os.path.basename(image_path), predictions[0], predictions[1], predictions[2], is_dead))
    
    if results:
        df = pd.DataFrame(results, columns=['Image', 'Peeling', 'Contamination', 'Cell Density', 'Empty/Dead'])
        df.to_csv(output_csv, index=False)
        print(f"Predictions saved to {output_csv}")
    else:
        print("No results to save.")

if __name__ == "__main__":
    test_model('data/images')
