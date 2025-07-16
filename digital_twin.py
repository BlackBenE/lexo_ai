import os
import json
from datetime import datetime

USERS_DIR = "users"
UPDATE_INTERVAL_DAYS = 1


def init_twin_fields(twin):
    # inits field defaults if not present
    if "pending_updates" not in twin:
        twin["pending_updates"] = []
    if "last_update_date" not in twin:
        twin["last_update_date"] = None

def apply_pending_updates_if_needed(twin):
    # aaply pending updates if overdue
    if should_update(twin["last_update_date"]):
        print("\n Weekly update of the twin...")
        for record in twin["pending_updates"]:
            update_twin(twin, record)
        twin["pending_updates"] = []
        twin["last_update_date"] = datetime.today().strftime("%Y-%m-%d")
        print(" Twin updated with accumulated data.")
    

def should_update(last_update_str):
    if not last_update_str:
        return True
    last_update = datetime.strptime(last_update_str, "%Y-%m-%d")
    return (datetime.today() - last_update).days >= UPDATE_INTERVAL_DAYS

def load_twin(user_id):
    """ Load the digital twin from a JSON file. If the file does not exist, create a new twin with default values. """
    filepath = os.path.join(USERS_DIR, f"user_{user_id}.json")
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    else:
        return {
            "user_id": user_id,
            "skills": {},
            "confusions": {},
            "history": [],
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
    """Update the digital twin with a new answer record (history, confusions, skills)."""

    twin["history"].append(answer_record)

    if not answer_record["correct"]:
        key = f'{answer_record["expected_answer"]}→{answer_record["child_answer"]}'
        twin["confusions"][key] = twin["confusions"].get(key, 0) + 1

    category = answer_record["category"]
    if category not in twin["skills"]:
        twin["skills"][category] = {"correct": 0, "total": 0, "accuracy": 0.0}

    twin["skills"][category]["total"] += 1
    if answer_record["correct"]:
        twin["skills"][category]["correct"] += 1

    correct = twin["skills"][category]["correct"]
    total = twin["skills"][category]["total"]
    twin["skills"][category]["accuracy"] = round(correct / total, 2)

def apply_pending_updates(twin):
    for record in twin.get("pending_updates", []):
        update_twin(twin, record)
    twin["pending_updates"] = []
    twin["last_update_date"] = datetime.today().strftime("%Y-%m-%d")