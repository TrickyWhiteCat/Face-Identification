import torch
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
from model import InceptionV2, face_recognition
from dataset import YourFaceProcessingClass
import os
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score


# Define the path to the dataset
dataset_path = "/kaggle/input/lfw-dataset/lfw-deepfunneled/lfw-deepfunneled"

# Function to get a list of all image paths in a folder
def get_image_paths(folder):
    return [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith(".jpg")]

# Function to test randomly selected pairs of images
def test_random_pairs(model, face_processor, threshold=0.5, num_tests=100):
    correct_predictions = 0
    true_labels = []
    predicted_labels = []


    for _ in range(num_tests):
        # Select two random folders
        folder1 = random.choice(os.listdir(dataset_path))
        folder2 = random.choice(os.listdir(dataset_path))

        # Skip if the same folder is selected
        if folder1 == folder2:
            continue

        # Get a random image path from each folder
        image1_path = random.choice(get_image_paths(os.path.join(dataset_path, folder1)))
        image2_path = random.choice(get_image_paths(os.path.join(dataset_path, folder2)))

        # Extract faces and get embeddings
        image1_process = face_processor.extract_face(image1_path)
        image2_process = face_processor.extract_face(image2_path)

        image1_pil = Image.fromarray(image1_process)
        image2_pil = Image.fromarray(image2_process)

        transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
        ])

        image1_pil = transform(image1_pil).unsqueeze(0).to(device)
        image2_pil = transform(image2_pil).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            _, _, embedding1 = model(image1_pil)
            _, _, embedding2 = model(image2_pil)

        # Check if the embeddings are of the same person
        is_same_person = face_recognition(embedding1, embedding2, threshold)
        
        # Plot the processed images with folder names
        plt.figure(figsize=(8, 4))

        # Plot Image 1 with folder name
        plt.subplot(1, 2, 1)
        plt.imshow(image1_process)
        plt.title(f"Image 1 - Folder: {folder1}")
        plt.axis("off")

        # Plot Image 2 with folder name
        plt.subplot(1, 2, 2)
        plt.imshow(image2_process)
        plt.title(f"Image 2 - Folder: {folder2}")
        plt.axis("off")

        # Show the plot
        plt.show()

        if is_same_person:
            print(f"They are the same person from folders {folder1} and {folder2}.")
        else:
            print(f"They are different people from folders {folder1} and {folder2}.")

        # Append true label and predicted label
        true_labels.append(1 if folder1 == folder2 else 0)
        predicted_labels.append(1 if is_same_person else 0)

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # Compute accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)

    print("Confusion Matrix:")
    print(cm)
    print("Accuracy:", accuracy)


