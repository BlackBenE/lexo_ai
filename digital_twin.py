import os
import json
from datetime import datetime

USERS_DIR = "users"

def load_twin(user_id):
    """ Load the digital twin from a JSON file. If the file does not exist, create a new twin with default values. """

    filepath = os.path.join(USERS_DIR, f"user_{user_id}.json")
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    else:
        return {
            "user_id": user_id,
            "skills": {},        # { "letters": {"correct": 5, "total": 7, "accuracy": 0.71}, ... }
            "confusions": {},    # { "letters→numbers": 2, ... }
            "history": [],       # Liste d'objets answer_record
             "pending_updates": [],
            "last_update_date": None
            
        }

def save_twin(user_id, twin):
    """ Save the digital twin to a JSON file. Creates the directory if it does not exist. """

    os.makedirs(USERS_DIR, exist_ok=True)
    filepath = os.path.join(USERS_DIR, f"user_{user_id}.json")
    with open(filepath, "w") as f:
        json.dump(twin, f, indent=2)

def update_twin(twin, answer_record):
    """ Update the digital twin with a new answer record. This includes updating history, confusions, and skills. """
    
    # Add answer record to history
    twin["history"].append(answer_record)

    # Update confusions
    if not answer_record["correct"]:
        key = f'{answer_record["expected_answer"]}→{answer_record["child_answer"]}'
        twin["confusions"][key] = twin["confusions"].get(key, 0) + 1

    # Update skills
    category = answer_record["category"]
    if category not in twin["skills"]:
        twin["skills"][category] = {"correct": 0, "total": 0, "accuracy": 0.0}

    twin["skills"][category]["total"] += 1
    if answer_record["correct"]:
        twin["skills"][category]["correct"] += 1

    correct = twin["skills"][category]["correct"]
    total = twin["skills"][category]["total"]
    twin["skills"][category]["accuracy"] = round(correct / total, 2)
