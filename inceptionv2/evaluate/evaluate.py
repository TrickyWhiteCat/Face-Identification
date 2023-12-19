import torch
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
import os
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import time
from model import InceptionV2, face_recognition
from dataset import YourFaceProcessingClass


# Define the path to the dataset
dataset_path = "/kaggle/input/lfw-dataset/lfw-deepfunneled/lfw-deepfunneled"

# Function to get a list of all image paths in a folder
def get_image_paths(folder):
    return [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith(".jpg")]

def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size
def write_to_csv(csv_writer, model_name, embedding_size, input_size1, input_size2, transform_time, execution_time, accuracy):
    csv_writer.writerow([model_name, embedding_size, input_size1, input_size2, transform_time, execution_time, accuracy])

    # Define the path to save the CSV file
csv_file_path = "evalute.csv"

# Function to test randomly selected pairs of images
def test_random_pairs1(model, face_processor, threshold=0.5, num_tests=100):
    correct_predictions = 0
    true_labels = []
    predicted_labels = []

    total_execution_time = 0
    total_transformation_time = 0
    with open(csv_file_path, mode='w', newline='') as csv_file:
        # Create a CSV writer
        csv_writer = csv.writer(csv_file)

        # Write the header row
        csv_writer.writerow(['Model', 'Typed', 'Embedding Size', 'Input Size 1', 'Input Size 2', 'Speed Transform', 'Execution Time', 'Accuracy'])

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

            image_size1 = get_image_size(image1_path)
            image_size2 = get_image_size(image2_path)
            # Extract faces and get embeddings
            image1_process = face_processor.extract_face(image1_path)
            image2_process = face_processor.extract_face(image2_path)

            # Measure execution time for face recognition
            start_time = time.time()

            image1_pil = Image.fromarray(image1_process)
            image2_pil = Image.fromarray(image2_process)

            transform = transforms.Compose([
                    transforms.Resize((112, 112)),
                    transforms.ToTensor(),
                ])

                # Measure transformation time
            transform_start_time = time.time()
            image1_pil = transform(image1_pil).unsqueeze(0).to(device)
            image2_pil = transform(image2_pil).unsqueeze(0).to(device)
            transform_end_time = time.time()
            transformation_time = transform_end_time - transform_start_time
            total_transformation_time += transformation_time

            model.eval()
            with torch.no_grad():
                _, _, embedding1 = model(image1_pil)
                _, _, embedding2 = model(image2_pil)

                embedding_size1 = embedding1.size(-1)
                embedding_size2 = embedding2.size(-1)
                if embedding_size1 == embedding_size2:
                    embedding_size = embedding_size1
                else:
                    embedding_size = 'ERROR'

            end_time = time.time()
            execution_time = end_time - start_time
            total_execution_time += execution_time

                # Check if the embeddings are of the same person
            is_same_person = face_recognition(embedding1, embedding2, threshold)


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

            # Calculate average execution time per test
        avg_execution_time = total_execution_time / num_tests
        avg_transformation_time = total_transformation_time / num_tests

        avg_execution_time = round(avg_execution_time, 5)
        avg_transformation_time = round(avg_transformation_time, 5)

        print("Confusion Matrix:")
        print(cm)
        print("Accuracy:", round(accuracy, 5))
        print(f"Average Execution Time per Test: {avg_execution_time} seconds")
        print(f"Average Transformation Time per Test: {avg_transformation_time} seconds")
        write_to_csv(csv_writer, 'InceptionV2', 'Different People', embedding_size, image_size1, image_size2, avg_transformation_time, avg_execution_time, accuracy)
def test_same_pairs_random1(model, face_processor, threshold=0.9, num_tests=100):
    true_labels = []
    predicted_labels = []

    total_execution_time = 0
    total_transformation_time = 0

    # Open the CSV file in append mode
    with open(csv_file_path, mode='a', newline='') as csv_file:
        # Create a CSV writer
        csv_writer = csv.writer(csv_file)

        for _ in range(num_tests):
            folder = random.choice(os.listdir(dataset_path))
            image_paths = get_image_paths(os.path.join(dataset_path, folder))

            if len(image_paths) < 2:
                continue

            # Randomly select two different images from the folder
            image1_path, image2_path = random.sample(image_paths, 2)

            image_size1 = get_image_size(image1_path)
            image_size2 = get_image_size(image2_path)

            # Extract faces and get embeddings
            image1_process = face_processor.extract_face(image1_path)
            image2_process = face_processor.extract_face(image2_path)

            # Measure execution time for face recognition
            start_time = time.time()

            image1_pil = Image.fromarray(image1_process)
            image2_pil = Image.fromarray(image2_process)

            transform = transforms.Compose([
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
            ])

            # Measure transformation time
            transform_start_time = time.time()
            image1_pil = transform(image1_pil).unsqueeze(0).to(device)
            image2_pil = transform(image2_pil).unsqueeze(0).to(device)
            transform_end_time = time.time()
            transformation_time = transform_end_time - transform_start_time
            total_transformation_time += transformation_time

            model.eval()
            with torch.no_grad():
                _, _, embedding1 = model(image1_pil)
                _, _, embedding2 = model(image2_pil)

                embedding_size1 = embedding1.size(-1)
                embedding_size2 = embedding2.size(-1)
                if embedding_size1 == embedding_size2:
                    embedding_size = embedding_size1
                else:
                    embedding_size = 'ERROR'

            end_time = time.time()
            execution_time = end_time - start_time
            total_execution_time += execution_time

            # Check if the embeddings are of the same person
            is_same_person = face_recognition(embedding1, embedding2, threshold)

            if is_same_person:
                print(f"They are the same person from folder {folder}.")
            else:
                print(f"They are different people from folder {folder}.")

            # Append true label and predicted label
            true_labels.append(1)
            predicted_labels.append(1 if is_same_person else 0)
        # Compute confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)

    # Compute accuracy
        accuracy = accuracy_score(true_labels, predicted_labels)

    # Calculate average execution time per test
        avg_execution_time = total_execution_time / num_tests
        avg_transformation_time = total_transformation_time / num_tests

        avg_execution_time = round(avg_execution_time, 5)
        avg_transformation_time = round(avg_transformation_time, 5)

        print("Confusion Matrix:")
        print(cm)
        print("Accuracy:", round(accuracy, 5))
        print(f"Average Execution Time per Test: {avg_execution_time} seconds")
        print(f"Average Transformation Time per Test: {avg_transformation_time} seconds")

            # Write a row to the CSV file
        write_to_csv(csv_writer, 'InceptionV2','Same Person', embedding_size, image_size1, image_size2, transformation_time, execution_time, accuracy)



loaded_inception_model = InceptionV2()
loaded_inception_model.load_state_dict(torch.load('inceptionv2/result/InceptionV2_model.pth'))

face_processor = YourFaceProcessingClass()
threshold = 4

loaded_inception_model.eval() 
# Call the test_same_pairs_random function
test_same_pairs_random1(loaded_inception_model, face_processor, threshold=threshold, num_tests=1000)

# Call the test_random_pairs function
test_random_pairs1(loaded_inception_model, face_processor, threshold=threshold, num_tests=1000)
