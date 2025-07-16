from sklearn.ensemble import RandomForestClassifier
from features import extract_features_from_twin
import numpy as np

def twin_to_dataset(twin):
    """
    transform the digital twin into a dataset of features and labels.
    """
    X, y, categories = extract_features_from_twin(twin, source="history")
    return X, y

def train_prediction_model(X, y):
    """
    train the Random Forest model using the features and labels extracted from the twin.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def predict_success(model, twin, category, expected_answer):
    """
    use the trained model to predict the probability of success for a given category and expected answer.
    """
    X, _, categories = extract_features_from_twin(twin, source="history")
    if category in categories:
        idx = categories.index(category)
        features = X[idx].reshape(1, -1)
        prob = model.predict_proba(features)[0][1]
        return prob
    else:
        # if the category is not found, return a default probability
        return 0.5