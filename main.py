import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from twin_predictor import twin_to_dataset, train_prediction_model, predict_success
from autoencoder import train_autoencoder, compute_reconstruction_errors
from digital_twin import load_twin, save_twin, update_twin, init_twin_fields, apply_pending_updates_if_needed
from features import extract_features_from_twin
from utils import load_dataset, load_images

# === Config ===
IMAGE_REDUCED_SIZE = 64
IMAGE_DIR = "images"
FOLDERS = ['letters', 'numbers', 'flags', 'animals']
SESSION_SIZE = 15 

# === Utils ===
def show_image(image):
    plt.imshow(image)
    plt.axis("off")
    plt.show()


def run_prediction_analysis(twin):
    X, y = twin_to_dataset(twin)
    if len(X) > 5:
        model = train_prediction_model(X, y)
        category = "animals"
        expected_answer = "dog"
        proba = predict_success(model, twin, category, expected_answer)
        print(f"Probability of success for '{expected_answer}' ({category}) : {proba:.2f}")
    else:
        print("Not yet enough data to train the model. ")

def run_test_session(twin, images, expected_answers, categories, reconstruction_errors):
    for i in range(SESSION_SIZE):
        print(f"\nImage {i + 1}/{SESSION_SIZE}")
        show_image(images[i])
        child_answer = input("The child's answer: ").strip().lower()
        expected = expected_answers[i].lower()
        correct = (child_answer == expected)

        category = categories[i]
        expected = expected_answers[i]
        visual_difficulty = float(reconstruction_errors[i])

        answer_record = {
            "date": datetime.today().strftime("%Y-%m-%d"),
            "category": category,
            "expected_answer": expected,
            "child_answer": child_answer,
            "correct": correct,
            "visual_difficulty": visual_difficulty
        }

        twin["pending_updates"].append(answer_record)
        print(" Good answer!" if correct else f" Bad answer. Correct was: '{expected}'.")
        print(f" → Visual difficulty: {visual_difficulty:.4f}")




def predict_improvements(model, twin):
    X, _, categories = extract_features_from_twin(twin)
    if model is None or len(X) == 0:
        print("No model or no data for prediction.")
        return []
    preds = model.predict(X)
    improvements = [cat for cat, pred in zip(categories, preds) if pred == 1]
    return improvements

def print_teacher_advice(improvements):
    if not improvements:
        print("No improvements needed for next week.")
    else:
        print(" Improvements needed for next week:")
        for cat in improvements:
            print(f"- Work more on the category'{cat}'.")

def main():
    user_id = input("Enter the child's ID : ").strip()
    twin = load_twin(user_id)
    init_twin_fields(twin)
    apply_pending_updates_if_needed(twin)

    filenames, expected_answers, categories = load_dataset(FOLDERS)
    images = load_images(filenames)
    print(f"{len(images)} loaded images.")

    print("\n Training autoencoder to estimate visual difficulty...")
    train_imgs, test_imgs, train_answers, test_answers, train_categories, test_categories = train_test_split(
        images, expected_answers, categories, test_size=0.3, random_state=42
    )

    autoencoder_model = train_autoencoder(train_imgs, epochs=50)
    reconstruction_errors = compute_reconstruction_errors(autoencoder_model, test_imgs)

    run_test_session(twin, test_imgs, test_answers, test_categories, reconstruction_errors)

    save_twin(user_id, twin)
    print(f"\n Twin for child '{user_id}' saved with pending data.")

    X, y = twin_to_dataset(twin)
    if len(X) < 5:
        print("Not enough data to train the prediction model.")
        model = None
    else:
        model = train_prediction_model(X, y)

    run_prediction_analysis(twin)

    improvements = predict_improvements(model, twin)
    print_teacher_advice(improvements)


# === Main ===
if __name__ == "__main__":
    main()


