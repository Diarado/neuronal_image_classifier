from scripts.parse_csv_to_dict import parse_csv_to_dict
from scripts.link_images_to_scores import link_images_to_scores
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from joblib import dump
import numpy as np

def train_in_batches(X, y, batch_size=100):
    X = np.array(X, dtype=np.float16)  # Convert X to float16 to reduce memory usage
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    for i in range(0, len(X), batch_size):
        X_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size]
        model.fit(X_batch, y_batch)
    
    return model

if __name__ == "__main__":
    csv_dict = parse_csv_to_dict('train/scoring_round06.csv')
    X, y = link_images_to_scores('train/round06_images', csv_dict)
    
    model = train_in_batches(X, y)  # Use batch training
    dump(model, 'models/classifier_model.pkl')  # Save the trained model
