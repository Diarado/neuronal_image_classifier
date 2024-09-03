import os
import numpy as np
import gc
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from scripts.parse_csv_to_dict import parse_csv_to_dict
from scripts.link_images_to_scores import link_images_to_scores
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.applications import ResNet50 # type: ignore
import tensorflow.keras.backend as K # type: ignore
from tensorflow.keras.mixed_precision import set_global_policy, Policy # type: ignore

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

def train_in_batches(X, y, batch_size=32, epochs=50, patience=5):
    K.clear_session()
    # Convert X and y to numpy arrays and ensure they are 4D
    X = np.array(X, dtype=np.float32)

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
    
    # Load a pre-trained ResNet50 model without the top classification layer
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(X.shape[1], X.shape[2], X.shape[3]))

    # Add custom classification layers on top
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)

    # Output layers for multi-output classification
    peeling_output = Dense(3, activation='softmax', name='peeling_output', dtype='float32')(x)  # Ensure float32 for mixed precision
    contamination_output = Dense(3, activation='softmax', name='contamination_output', dtype='float32')(x)  # Ensure float32 for mixed precision
    density_output = Dense(3, activation='softmax', name='density_output', dtype='float32')(x)  # Ensure float32 for mixed precision

    # Define the model with multiple outputs
    model = Model(inputs=base_model.input, outputs=[peeling_output, contamination_output, density_output])

    # Freeze the layers of the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model
    model.compile(optimizer='adam', 
                  loss={'peeling_output': 'categorical_crossentropy', 
                        'contamination_output': 'categorical_crossentropy',
                        'density_output': 'categorical_crossentropy'},
                  metrics={'peeling_output': 'accuracy',
                           'contamination_output': 'accuracy',
                           'density_output': 'accuracy'})

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

    train_dataset = data_generator(X, y, batch_size)

    steps_per_epoch = len(X) // batch_size

    model.fit(train_dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, 
              callbacks=[early_stopping], verbose=1)
    
    return model, None  # Scaler is not needed for image data

def convert_to_one_hot(x, y):
    # Convert the labels to integers
    y = {
        'peeling_output': tf.cast(y['peeling_output'], tf.int32),
        'contamination_output': tf.cast(y['contamination_output'], tf.int32),
        'density_output': tf.cast(y['density_output'], tf.int32),
    }

    # Apply one-hot encoding
    y = {
        'peeling_output': tf.one_hot(y['peeling_output'], depth=3),
        'contamination_output': tf.one_hot(y['contamination_output'], depth=3),
        'density_output': tf.one_hot(y['density_output'], depth=3),
    }
    
    return x, y

def data_generator(X, y, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((X, {
        'peeling_output': y[:, 0] - 1,
        'contamination_output': y[:, 1] - 1,
        'density_output': y[:, 2] - 1
    }))
    dataset = dataset.shuffle(buffer_size=len(X))
    dataset = dataset.batch(batch_size)

    def convert_to_one_hot(x, y):
        # Convert the labels to integers
        y = {
            'peeling_output': tf.cast(y['peeling_output'], tf.int32),
            'contamination_output': tf.cast(y['contamination_output'], tf.int32),
            'density_output': tf.cast(y['density_output'], tf.int32),
        }

        # Apply one-hot encoding
        y = {
            'peeling_output': tf.one_hot(y['peeling_output'], depth=3),
            'contamination_output': tf.one_hot(y['contamination_output'], depth=3),
            'density_output': tf.one_hot(y['density_output'], depth=3),
        }
        
        return x, y

    # Map the conversion function across the dataset
    dataset = dataset.map(convert_to_one_hot)

    return dataset



if __name__ == "__main__":
    # to set
    rounds = ['round06']

    image_dir_lst = [f'train/{round}_images' for round in rounds]
    csv_lst = [f'train/scoring_{round}.csv' for round in rounds]

    # X is labeled images and y is scores
    X, y = [], []

    X_path = f'train/X_{"_".join(rounds)}.npy'
    y_path = f'train/y_{"_".join(rounds)}.npy'
 
    if os.path.exists(X_path) and os.path.exists(y_path):
        print("Loading labeled images and scores from disk...")
        X = np.load(X_path)
        y = np.load(y_path)
    else:
        X, y = [], []
        for image_dir, csv_file in zip(image_dir_lst, csv_lst):
            csv_dict = parse_csv_to_dict(csv_file)
            X_part, y_part = link_images_to_scores(image_dir, csv_dict)

            X_filtered, y_filtered = [], []
            for img, score in zip(X_part, y_part):
                if score[3] != 1:
                    X_filtered.append(img)
                    y_filtered.append(score)
            
            X.extend(X_filtered)
            y.extend(y_filtered)
            
            del X_part, y_part
            gc.collect()

        if len(X) > 0 and len(y) > 0:
            X = np.array(X).astype('float32')
            y = np.array(y)
            
            print("Saving labeled images and scores to disk...")
            np.save(X_path, X)
            np.save(y_path, y)
        else:
            print("No data to save.")
    
    if len(X) > 0 and len(y) > 0:  # Check to ensure data is not empty
        X = np.array(X).astype('float32')
        y = np.array(y)
        
        model, _ = train_in_batches(X, y, batch_size=32)
        
        # Save the model
        os.makedirs('models', exist_ok=True)
        model.save('models/classifier_model.keras')
    else:
        print("No data to train on.")