# Lexo AI

Lexo AI is an educational tool designed to assess and improve children's learning through image-based sessions and digital twin modeling. It uses machine learning models to estimate visual difficulty, track user performance, and provide personalized advice for improvement.

## Features

- **Image-based Sessions:** Supports categories like letters, numbers, flags, and animals.
- **Digital Twin:** Tracks user history, skills, confusions, and pending updates in JSON format.
- **Autoencoder:** Estimates visual difficulty of images using a PyTorch-based autoencoder.
- **Prediction Model:** Uses Random Forest to predict success and suggest improvements.
- **Keras Models:** Includes scripts to train and export autoencoder and classifier models in Keras/TensorFlow.
- **User Data:** Stores user progress and session data in the `users/` directory.

## Project Structure

```
lexo_ai/
├── autoencoder.py           # PyTorch autoencoder for visual difficulty
├── train_keras_models.py    # Keras/TensorFlow model training and export
├── digital_twin.py          # Digital twin management (user history, skills)
├── features.py              # Feature extraction from twin data
├── main.py                  # Main script for running sessions and analysis
├── twin_predictor.py        # Random Forest prediction model
├── utils.py                 # Image loading and preprocessing
├── requirements.txt         # Python dependencies
├── images/                  # Image dataset (letters, numbers, flags, animals)
└── users/                   # User data (JSON files)
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/BlackBenE/lexo_ai.git
   cd lexo_ai
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Additional requirements for Keras/TensorFlow:
   ```bash
   pip install tensorflow
   ```
   For PyTorch:
   ```bash
   pip install torch
   ```
   For scikit-image:
   ```bash
   pip install scikit-image
   ```

## Usage

- **Run the main session:**

  ```bash
  python main.py
  ```

  Follow the prompts to enter a user ID and run a test session.

- **Train Keras models:**
  ```bash
  python train_keras_models.py
  ```
  This will train and export autoencoder and classifier models in both H5 and TFLite formats.

## Data

- **Images:** Place your images in the respective folders under `images/` (`letters/`, `numbers/`, `flags/`, `animals/`).
- **User Data:** User progress is stored in `users/user_<id>.json`.

## How It Works

1. Loads images and user data.
2. Trains an autoencoder to estimate visual difficulty.
3. Runs a test session, records answers and difficulties.
4. Updates the digital twin and saves user progress.
5. Trains a prediction model to analyze performance and suggest improvements.

## License

MIT License

## Author

BlackBenE
