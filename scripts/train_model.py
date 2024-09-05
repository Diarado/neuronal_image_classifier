import os
import numpy as np
import pandas as pd
import gc
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from scripts.parse_csv_to_dict import parse_csv_to_dict
from scripts.link_images_to_scores import link_images_to_scores
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D  # type: ignore
from keras.layers import Concatenate # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.applications import ResNet50 # type: ignore
import tensorflow.keras.backend as K # type: ignore
from tensorflow.keras.mixed_precision import set_global_policy, Policy # type: ignore
from sklearn.model_selection import KFold
from keras.optimizers import Adam # type: ignore

policy = Policy('mixed_float16')
set_global_policy(policy)

print("Is TensorFlow built with CUDA?", tf.test.is_built_with_cuda())  # False
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU'))) # 0 

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs detected: {len(gpus)}")
    for gpu in gpus:
        print(f"- {gpu}")
    # Optionally, set memory growth to avoid allocating all memory at once
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPUs detected, using CPU.")

    
def train_in_batches(X, y, extracted_feature_lst, batch_size=32, epochs=20, patience=5):
    # extracted_feature_lst records [is_dead, peeling_degree, contamination_degree, cell_density] for images in X and y
    # so X[i], y[i], extracted_feature_lst[i] are labled image, pre_assigned score (peeling 1-3, contamination 1-3, cell density 1-3, and dead 0/1),
    # and extracted features for image i, respectively
    
    K.clear_session()
    # Convert X and y to numpy arrays and ensure they are 4D
    X = np.array(X, dtype=np.float32)
    extracted_feature_lst = np.array(extracted_feature_lst, dtype=np.float32)
    # Add a channel dimension if X is 3D (grayscale)
    if len(X.shape) == 3:
        X = np.expand_dims(X, axis=-1)  # Expand to (num_samples, height, width, 1)

    # If X has 1 channel, duplicate it to create 3 channels
    if X.shape[-1] == 1:
        X = np.repeat(X, 3, axis=-1)  # Convert 1-channel grayscale to 3-channel

    # Ensure X has the correct shape
    assert X.shape[-1] == 3, f"Expected 3-channel input, but got shape {X.shape}"

    y = np.array(y, dtype=np.float32)

    # Normalize image data
    X /= 255.0

    image_input = Input(shape=(X.shape[1], X.shape[2], X.shape[3]), name='image_input')  

    # Load a pre-trained ResNet50 model without the top classification layer
    base_model = ResNet50(weights='imagenet', include_top=False, 
                          input_shape=(X.shape[1], X.shape[2], X.shape[3]), input_tensor=image_input)

    # Add custom classification layers on top
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)

    feature_input = Input(shape=(extracted_feature_lst.shape[1],), name='feature_input') 
    print('extracted_feature_lst.shape[1]: ' + str(extracted_feature_lst.shape[1]))

    # Extract individual features from the manual features
    is_dead_input = feature_input[:, 0:1]  # is_dead
    peeling_degree_input = feature_input[:, 1:2]  # peeling_degree
    contamination_degree_input = feature_input[:, 2:3]  # contamination_degree
    cell_density_input = feature_input[:, 3:4]  # cell_density

    # Concatenate image features with relevant manual features for each output head
    peeling_features = Concatenate()([x, peeling_degree_input])
    contamination_features = Concatenate()([x, contamination_degree_input])
    density_features = Concatenate()([x, cell_density_input])

    # Output layers for multi-output classification
    peeling_output = Dense(3, activation='softmax', name='peeling_output', dtype='float32')(peeling_features)
    contamination_output = Dense(3, activation='softmax', name='contamination_output', dtype='float32')(contamination_features)
    density_output = Dense(3, activation='softmax', name='density_output', dtype='float32')(density_features)
    dead_output = Dense(1, activation='sigmoid', name='dead_output', dtype='float32')(is_dead_input)  # Binary classification

    # Define the model with multiple outputs
    model = Model(inputs=[image_input, feature_input], 
                  outputs=[peeling_output, contamination_output, density_output, dead_output])

    # Freeze the layers of the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss={'peeling_output': 'categorical_crossentropy', 
                        'contamination_output': 'categorical_crossentropy',
                        'density_output': 'categorical_crossentropy',
                        'dead_output': 'binary_crossentropy'},
                  metrics={'peeling_output': 'accuracy',
                           'contamination_output': 'accuracy',
                           'density_output': 'accuracy',
                           'dead_output': 'accuracy'})

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

    train_dataset = data_generator(X, y, extracted_feature_lst, batch_size)

    steps_per_epoch = len(X) // batch_size

    model.fit(train_dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, 
              callbacks=[early_stopping], verbose=1)
    
    return model, None  # Scaler is not needed for image data

def convert_to_one_hot(y):
    return {
        'peeling_output': tf.one_hot(tf.cast(y[:, 0] - 1, tf.int32), depth=3),  # Peeling 1-3
        'contamination_output': tf.one_hot(tf.cast(y[:, 1] - 1, tf.int32), depth=3),  # Contamination 1-3
        'density_output': tf.one_hot(tf.cast(y[:, 2] - 1, tf.int32), depth=3),  # Density 1-3
        'dead_output': y[:, 3:4]  # Dead is binary 0 or 1, no need for one-hot
    }

# Data generator function
def data_generator(X, y, extracted_feature_lst, batch_size):
    # Create dataset from images, manual features, and labels
    dataset = tf.data.Dataset.from_tensor_slices(((X, extracted_feature_lst), y))
    dataset = dataset.shuffle(buffer_size=len(X))
    dataset = dataset.batch(batch_size)

    # Map the dataset to convert labels to one-hot encoding
    dataset = dataset.map(lambda inputs, y: (inputs, convert_to_one_hot(y)))

    return dataset


if __name__ == "__main__":
    # to set
    # rounds = ['round06', 'round09', 'round11']
    rounds = ['round06']

    image_dir_lst = [f'train/{round}_images' for round in rounds]
    csv_lst = [f'train/scoring_{round}.csv' for round in rounds]

    # X is labeled images and y is scores
    X, y = [], []

    X_path = f'train/X_{"_".join(rounds)}.npy'
    y_path = f'train/y_{"_".join(rounds)}.npy'
    features_path = f'train/features_{"_".join(rounds)}.npy'
 
    if os.path.exists(X_path) and os.path.exists(y_path) and os.path.exists(features_path):
        print("Loading labeled images and scores from disk...")
        X = np.load(X_path)
        y = np.load(y_path)
        extracted_feature_lst = np.load(features_path)
    else:
        # Initialize empty lists
        X, y, extracted_feature_lst = [], [], []

        # Iterate over the image directories and CSV files
        for image_dir, csv_file in zip(image_dir_lst, csv_lst):
            csv_dict = parse_csv_to_dict(csv_file)
            
            # Link images to scores
            X_part, y_part, extracted_feature_lst_part = link_images_to_scores(image_dir, csv_dict) 
            
            # Optional filtering step: Add logic if needed
            X_filtered, y_filtered, extracted_feature_lst_filtered = [], [], []
            for img, score, features in zip(X_part, y_part, extracted_feature_lst_part):
                if True:  # Placeholder for any filtering logic
                    X_filtered.append(img)
                    y_filtered.append(score)
                    extracted_feature_lst_filtered.append(features)

            # Extend the main lists with filtered data
            X.extend(X_filtered)
            y.extend(y_filtered)
            extracted_feature_lst.extend(extracted_feature_lst_filtered)

            # Memory management after processing each round
            del X_part, y_part, extracted_feature_lst_part, X_filtered, y_filtered, extracted_feature_lst_filtered
            gc.collect()

        if len(X) > 0 and len(y) > 0:
            X = np.array(X).astype('float32')
            y = np.array(y)
            extracted_feature_lst = np.array(extracted_feature_lst)

            # Save data for future use
            print("Saving labeled images, scores, and features to disk...")
            np.save(X_path, X)
            np.save(y_path, y)
            np.save(features_path, extracted_feature_lst)
        else:
            print("No data to save.")

    # Train the model if data is available
    if len(X) > 0 and len(y) > 0 and len(extracted_feature_lst) > 0:
        extracted_feature_lst = np.array(extracted_feature_lst)

        # Perform training (or cross-validation if needed)
        model, _ = train_in_batches(X, y, extracted_feature_lst, batch_size=32)
        
        # Save the trained model
        os.makedirs('models', exist_ok=True)
        model.save('models/classifier_model.keras')
    else:
        print("No data to train on.")