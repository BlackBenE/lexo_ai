import numpy as np
from collections import defaultdict

def extract_features_from_twin(twin, source="pending_updates"):
    """
    trandform history or pending_updates of the digital twin into a dataset of features and labels.
    """
    data = defaultdict(list)
    for record in twin.get(source, []):
        cat = record["category"]
        data[cat].append(record)
    
    features = []
    labels = []
    categories = []
    for cat, records in data.items():
        total = len(records)
        correct = sum(r["correct"] for r in records)
        avg_diff = np.mean([r["visual_difficulty"] for r in records])
        success_rate = correct / total if total > 0 else 0
        features.append([success_rate, avg_diff, total])
        labels.append(1 if success_rate < 0.7 else 0)
        categories.append(cat)
    return np.array(features), np.array(labels), categories