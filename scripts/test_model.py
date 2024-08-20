import os
from joblib import load
from scripts.preprocess import preprocess_image
import pandas as pd

def preprocess_and_flatten(image_path):
    binary_image, _ = preprocess_image(image_path)
    return binary_image.flatten()

def test_model(image_dir, model_path='models/classifier_model.pkl', output_csv='predictions.csv'):

    model = load(model_path)
    image_dir = os.path.join(image_dir, 'images')
    # List all files in the images directory for debugging
    all_files = os.listdir(image_dir) 
    # Gather all .tif files
    image_paths = [os.path.join(image_dir, file) for file in all_files if file.endswith('.tif')]
    results = []
    
    for image_path in image_paths:

        processed_image = preprocess_and_flatten(image_path)
        prediction = model.predict([processed_image])
        results.append((os.path.basename(image_path), prediction[0]))
    
    if results:
        df = pd.DataFrame(results, columns=['Image', 'Prediction'])
        df.to_csv(output_csv, index=False)
        print(f"Predictions saved to {output_csv}")
    else:
        print("No results to save.")


if __name__ == "__main__":
    test_model('data')


