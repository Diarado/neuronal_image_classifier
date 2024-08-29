import os
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import gc
from scripts.parse_csv_to_dict import parse_csv_to_dict
from scripts.link_images_to_scores import link_images_to_scores

def train_in_batches(X, y, batch_size=100):
    print("X type:", type(X))
    print("X sample:", X[:1])  # Print the first element to see its structure
    print("X dimensions:", np.array(X).ndim)  # Print the number of dimensions of X

    X = np.array(X, dtype=np.float32)  # Convert X to numpy array, ensuring it's 2D
    scaler = StandardScaler()
    X = scaler.fit_transform(X)  # Normalize the features
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    for i in range(0, len(X), batch_size):
        X_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size]
        model.fit(X_batch, y_batch)
        
        # Free memory after each batch
        del X_batch, y_batch
        gc.collect()
    
    return model, scaler

if __name__ == "__main__":
    image_dir_lst = ['train/round06_images']
    csv_lst = ['train/scoring_round06.csv']
    
    X, y = [], []
    
    for image_dir, csv_file in zip(image_dir_lst, csv_lst):
        csv_dict = parse_csv_to_dict(csv_file)
        X_part, y_part = link_images_to_scores(image_dir, csv_dict)

        X_filtered, y_filtered = [], []
        for img, score in zip(X_part, y_part):
            if score[3] != 1:  # Exclude images with empty/dead = 1
                # Count labeled pixels directly on the preprocessed image
                num_peeling_pixels = np.sum(img == 3)
                num_neuron_cells = np.sum(img == 2)
                
                # Create feature vector correctly as a list of numerical values
                feature_vector = [num_peeling_pixels, num_neuron_cells, score[2]]  # Use the provided cell_density score
                
                X_filtered.append(feature_vector)
                y_filtered.append(score)  # Use the authentic density score from y_part

        # Debugging print statements to inspect X_filtered
        print("X_filtered sample:", X_filtered[:1])  # Print the first element of X_filtered
        print("X_filtered type:", type(X_filtered[0]))  # Print the type of elements in X_filtered
        print("X_filtered shape:", len(X_filtered), len(X_filtered[0]) if X_filtered else 'N/A')

        X.extend(X_filtered)
        y.extend(y_filtered)
        
        # Clear memory after each set of images
        del X_part, y_part
        gc.collect()
    
    model, scaler = train_in_batches(X, y)  # Use batch training
    
    # Ensure the models directory exists
    os.makedirs('models', exist_ok=True)
    
    dump(model, 'models/classifier_model.pkl')
    dump(scaler, 'models/scaler.pkl')  # Save the scaler for later use


