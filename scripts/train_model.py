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
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set policy for mixed precision
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

print("Is TensorFlow built with CUDA?", tf.test.is_built_with_cuda())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# GPU memory management
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs detected: {len(gpus)}")
    for gpu in gpus:
        print(f"- {gpu}")
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPUs detected, using CPU.")

# Data generator with augmentation but fixed extracted features
def data_generator_with_augmentation(X, y, extracted_feature_lst, batch_size):
    # Augmentation only on image data (not on extracted features)
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Create a generator for augmented images
    image_generator = datagen.flow(X, batch_size=batch_size, shuffle=False)

    def generator():
        while True:
            image_batch = next(image_generator)
            batch_size_actual = image_batch.shape[0]
            feature_batch = extracted_feature_lst[:batch_size_actual]
            label_batch = y[:batch_size_actual]

            # Yield a tuple for inputs instead of a list
            yield ((image_batch, feature_batch), convert_to_one_hot(label_batch))

    # Define the output signature (specifying the types and shapes of the output)
    output_signature = (
        (
            tf.TensorSpec(shape=(None, X.shape[1], X.shape[2], X.shape[3]), dtype=tf.float32),  # Image batch
            tf.TensorSpec(shape=(None, extracted_feature_lst.shape[1]), dtype=tf.float32)  # Extracted features batch
        ),
        {
            'peeling_output': tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
            'contamination_output': tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
            'density_output': tf.TensorSpec(shape=(None, 3), dtype=tf.float32)
        }
    )

    # Create dataset using the generator
    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)

    return dataset



# Convert labels to one-hot encoding
def convert_to_one_hot(y):
    return {
        'peeling_output': tf.one_hot(tf.cast(y[:, 0] - 1, tf.int32), depth=3),  # Peeling 1-3
        'contamination_output': tf.one_hot(tf.cast(y[:, 1] - 1, tf.int32), depth=3),  # Contamination 1-3
        'density_output': tf.one_hot(tf.cast(y[:, 2] - 1, tf.int32), depth=3),  # Density 1-3
    }

# Training function with transfer learning, data augmentation, and regularization
def train_in_batches(X, y, extracted_feature_lst, batch_size=32, epochs=5, patience=5):
    print('original y:')
    print(y[:100])
    K.clear_session()

    # Normalize extracted features using StandardScaler
    scaler = StandardScaler()
    extracted_feature_lst = scaler.fit_transform(extracted_feature_lst)

    # Convert X and y to numpy arrays and ensure they are 4D
    X = np.array(X, dtype=np.float32)
    extracted_feature_lst = np.array(extracted_feature_lst, dtype=np.float32)

    # Add a channel dimension if X is 3D (grayscale)
    if len(X.shape) == 3:
        X = np.expand_dims(X, axis=-1)

    # If X has 1 channel, duplicate it to create 3 channels
    if X.shape[-1] == 1:
        X = np.repeat(X, 3, axis=-1)

    # Normalize image data
    X /= 255.0

    # Input layers for image and extracted features
    image_input = Input(shape=(X.shape[1], X.shape[2], X.shape[3]), name='image_input')
    feature_input = Input(shape=(extracted_feature_lst.shape[1],), name='feature_input')

    # Load a pre-trained ResNet50 model without the top classification layer
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(X.shape[1], X.shape[2], X.shape[3]), input_tensor=image_input)

    # Add custom classification layers on top with dropout and L2 regularization
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)

    # Concatenate image features with manually extracted features
    peeling_degree_input = feature_input[:, 1:2]
    contamination_degree_input = feature_input[:, 2:3]
    cell_density_input = feature_input[:, 3:4]

    peeling_features = Concatenate()([x, peeling_degree_input])
    contamination_features = Concatenate()([x, contamination_degree_input])
    density_features = Concatenate()([x, cell_density_input])

    # Output layers for multi-output classification
    peeling_output = Dense(3, activation='softmax', name='peeling_output')(peeling_features)
    contamination_output = Dense(3, activation='softmax', name='contamination_output')(contamination_features)
    density_output = Dense(3, activation='softmax', name='density_output')(density_features)

    # Define the model
    model = Model(inputs=[image_input, feature_input], outputs=[peeling_output, contamination_output, density_output])

    # Freeze the layers of the base model (ResNet50)
    for layer in base_model.layers[-10:]:
        layer.trainable = False

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss={'peeling_output': 'categorical_crossentropy',
                        'contamination_output': 'categorical_crossentropy',
                        'density_output': 'categorical_crossentropy'},
                  metrics={'peeling_output': 'accuracy',
                           'contamination_output': 'accuracy',
                           'density_output': 'accuracy'})

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

    # Create the data generator with augmentation
    train_dataset = data_generator_with_augmentation(X, y, extracted_feature_lst, batch_size)
    steps_per_epoch = len(X) // batch_size

    class_weight = calculate_class_weights(y)
    # Train the model
    model.fit(train_dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, 
             
              callbacks=[early_stopping], verbose=1)

    return model, scaler  # Return the model and scaler for extracted features

def calculate_class_weights(y):
    print(y)
    # Ensure class labels in y are integers (in case they are strings)
    peeling_scores = y[:, 0].astype(int)  # Cast to int to avoid strings
    print(peeling_scores[:100])
    contamination_scores = y[:, 1].astype(int)
    density_scores = y[:, 2].astype(int)

    # Unique classes for each output
    unique_peeling_classes = np.unique(peeling_scores)
    unique_contamination_classes = np.unique(contamination_scores)
    unique_density_classes = np.unique(density_scores)
    print(unique_peeling_classes)

    # Calculate class weights for each output
    peeling_class_weight = compute_class_weight('balanced', classes=unique_peeling_classes, y=peeling_scores)
    contamination_class_weight = compute_class_weight('balanced', classes=unique_contamination_classes, y=contamination_scores)
    density_class_weight = compute_class_weight('balanced', classes=unique_density_classes, y=density_scores)

    # Create class weight dictionary ensuring integer keys (not strings)
    class_weight = {
        'peeling_output': {int(cls): peeling_class_weight[i] for i, cls in enumerate(unique_peeling_classes)},
        'contamination_output': {int(cls): contamination_class_weight[i] for i, cls in enumerate(unique_contamination_classes)},
        'density_output': {int(cls): density_class_weight[i] for i, cls in enumerate(unique_density_classes)},
    }

    return class_weight



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
        model, scaler = train_in_batches(X, y, extracted_feature_lst)
        # model = train_image_only(X, y)
        # model = train_on_features_only(extracted_feature_lst, y)
        
        # Save the trained model
        os.makedirs('models', exist_ok=True)
        model.save('models/classifier_model.keras')
    else:
        print("No data to train on.")