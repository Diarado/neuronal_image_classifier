import os
import numpy as np
import gc
from sklearn.preprocessing import StandardScaler
from scripts.parse_csv_to_dict import parse_csv_to_dict
from scripts.link_images_to_scores import link_images_to_scores
import tensorflow as tf
from joblib import dump
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

# Check if GPU is available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print("Using GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("Using CPU")

def train_in_batches(X, y, batch_size=100, epochs=50, patience=5):
    # Convert X and y to numpy arrays and ensure they are 2D
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Define the model architecture
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X.shape[1],)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(y.shape[1], activation='linear')  # Linear output for regression
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    
    # Train the model
    model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=[early_stopping], verbose=1)
    
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
                num_peeling_pixels = np.sum(img == 3)
                num_neuron_cells = np.sum(img == 2)
                feature_vector = [num_peeling_pixels, num_neuron_cells, score[2]]
                X_filtered.append(feature_vector)
                y_filtered.append(score)
        
        X.extend(X_filtered)
        y.extend(y_filtered)
        
        # Clear memory after each set of images
        del X_part, y_part
        gc.collect()
    
    model, scaler = train_in_batches(X, y)
    
    # Save the model and scaler
    os.makedirs('models', exist_ok=True)
    model.save('models/classifier_model.keras')
    dump(scaler, 'models/scaler.pkl')

