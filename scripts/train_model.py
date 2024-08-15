import os
from joblib import dump
from scripts.parse_csv_to_dict import parse_csv_to_dict
from scripts.link_images_to_scores import link_images_to_scores
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np

def train_in_batches(X, y, batch_size=100):
    X = np.array([img.flatten() for img in X], dtype=np.float16)  # Flatten each image to 1D
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    for i in range(0, len(X), batch_size):
        X_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size]
        model.fit(X_batch, y_batch)
    
    return model

if __name__ == "__main__":
    image_dir_lst = ['train/round06_images', 'train/round09_images']
    csv_lst = ['train/scoring_round06.csv', 'train/scoring_round09.csv']
    
    X, y = [], []
    
    for image_dir, csv_file in zip(image_dir_lst, csv_lst):
        csv_dict = parse_csv_to_dict(csv_file)
        X_part, y_part = link_images_to_scores(image_dir, csv_dict)
        X.extend(X_part)
        y.extend(y_part)
    
    model = train_in_batches(X, y)  # Use batch training
    
    # Ensure the models directory exists
    os.makedirs('models', exist_ok=True)
    
    dump(model, 'models/classifier_model.pkl')

