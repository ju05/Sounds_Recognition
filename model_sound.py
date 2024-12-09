import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa
import csv
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
# Load the YAMNet model from TensorFlow Hub
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# Load YAMNet class labels from a CSV file
def load_labels(csv_file):
    labels = []
    with open(csv_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header
        for row in reader:
            labels.append(row[1])  # The second column contains the human-readable label
    return labels

# Function to load the audio file and preprocess it
def load_audio(file_path):
    # Load the audio file using librosa
    waveform, sr = librosa.load(file_path, sr=16000)  # YAMNet expects 16kHz sample rate
    return waveform, sr

# Function to predict the sound category using YAMNet
def predict_sound(file_path, labels):
    # Load and preprocess the audio
    waveform, sample_rate = load_audio(file_path)

    # Run YAMNet model on the waveform
    scores, embeddings, spectrogram = yamnet_model(waveform)

    # Get the top predicted category
    class_scores = scores.numpy().mean(axis=0)  # Average scores over time
    top_class_index = np.argmax(class_scores)

    # Get the corresponding label for the top predicted class
    predicted_label = labels[top_class_index]

    # Print the predicted sound category
    return predicted_label

# Path to your audio file and the labels file
audio_file = dir_path + r'\mixkit-woman-hilarious-laughing-410.wav'  # Replace with your actual audio file path
labels_file = dir_path + r'\yamnet_class_map.csv'  # Replace with your actual labels file path

# Load the labels
labels = load_labels(labels_file)

# Call the function to predict the sound
predict_sound(audio_file, labels)

# Load the labels.csv into a dictionary
def load_labels(file_path):
    labels_dict = {}
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            labels_dict[row['mid']] = row['display_name']
    return labels_dict

# Load the labels dictionary
labels_dict = load_labels(labels_file)

# Predicted sound ID (for example)
predicted_sound_id = predict_sound(audio_file, labels)

# Get the human-readable label for the predicted sound
predicted_label = labels_dict.get(predicted_sound_id, "Unknown sound")

print(f"Predicted sound: {predicted_sound_id} - {predicted_label}")
