import os
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import skimage.transform
from datetime import datetime, timedelta

from digital_twin import load_twin, save_twin, update_twin

# === Config ===
IMAGE_REDUCED_SIZE = 64
IMAGE_DIR = "images"
FOLDERS = ['letters', 'numbers', 'flags', 'animals']
SESSION_SIZE = 5 
UPDATE_INTERVAL_DAYS = 7  

# === Utils ===
def show_image(image):
    plt.imshow(image)
    plt.axis("off")
    plt.show()

def load_dataset(folders):
    filenames, labels, categories = [], [], []
    for folder in folders:
        folder_path = os.path.join(IMAGE_DIR, folder)
        for filename in os.listdir(folder_path):
            if filename.endswith('.png'):
                full_path = os.path.join(folder_path, filename)
                filenames.append(full_path)
                labels.append(os.path.splitext(filename)[0])  # e.g. "A", "3", "dog"
                categories.append(folder)  # e.g. "letters"
    return filenames, labels, categories

def load_images(filenames):
    images = []
    for filename in filenames:
        image = skimage.io.imread(filename)
        image = skimage.transform.resize(image, (IMAGE_REDUCED_SIZE, IMAGE_REDUCED_SIZE))
        images.append(image)
    return np.array(images)

def should_update(last_update_str):
    if not last_update_str:
        return True
    last_update = datetime.strptime(last_update_str, "%Y-%m-%d")
    return (datetime.today() - last_update).days >= UPDATE_INTERVAL_DAYS

# === Main ===
if __name__ == "__main__":
    user_id = input("Enter the child's ID : ").strip()
    twin = load_twin(user_id)

    # Initialization if missing fields
    if "pending_updates" not in twin:
        twin["pending_updates"] = []
    if "last_update_date" not in twin:
        twin["last_update_date"] = None

    # Deferred update (if overdue)
    if should_update(twin["last_update_date"]):
        print("\n Weekly update of the twin...")
        for record in twin["pending_updates"]:
            update_twin(twin, record)
        twin["pending_updates"] = []
        twin["last_update_date"] = datetime.today().strftime("%Y-%m-%d")
        print(" Twin updated with accumulated data.")

    # Data loading
    filenames, expected_answers, categories = load_dataset(FOLDERS)
    images = load_images(filenames)
    print(f"{len(images)} loaded images.")

    # Test session
    for i in range(SESSION_SIZE):
        print(f"\nImage {i + 1}/{SESSION_SIZE}")
        show_image(images[i])
        child_answer = input("The child's answer: ").strip()

        expected = expected_answers[i]
        category = categories[i]
        correct = (child_answer.lower() == expected.lower())

        answer_record = {
            "date": datetime.today().strftime("%Y-%m-%d"),
            "category": category,
            "expected_answer": expected,
            "child_answer": child_answer,
            "correct": correct
        }

        twin["pending_updates"].append(answer_record)
        print(" Good answer!" if correct else f" Bad answer. Correct was: '{expected}'.")

    #  Save the twin with pending updates
    save_twin(user_id, twin)
    print(f"\n Twin for child '{user_id}' saved with pending data.")
