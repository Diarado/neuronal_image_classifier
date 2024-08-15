from scripts.parse_csv_to_dict import parse_csv_to_dict
from scripts.link_images_to_scores import link_images_to_scores
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from joblib import dump

def train_and_save_model(X, y, output_file):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))
    
    dump(model, output_file)
    print(f"Model saved to {output_file}")

if __name__ == "__main__":
    csv_dict = parse_csv_to_dict('train/scoring_round06.csv')
    X, y = link_images_to_scores('train/round_06images', csv_dict)
    train_and_save_model(X, y, 'models/classifier_model.pkl')
