import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from joblib import dump
from scripts.preprocess import preprocess_image, extract_features

def load_data(image_dir, label_file):
    labels_df = pd.read_csv(label_file)
    X, y = [], []
    
    for _, row in labels_df.iterrows():
        image_path = os.path.join(image_dir, row['filename'])
        processed_image = preprocess_image(image_path)
        features = extract_features(processed_image)
        X.append(list(features.values()))
        y.append(row['category'])  # Adjust 'category' to match your CSV column
    
    return X, y

def train_and_save_model(X, y, output_file):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))
    
    dump(model, output_file)
    print(f"Model saved to {output_file}")

if __name__ == "__main__":
    X, y = load_data('train/round06_images', 'train/scoring_round06.csv')
    train_and_save_model(X, y, 'models/classifier_model.pkl')
