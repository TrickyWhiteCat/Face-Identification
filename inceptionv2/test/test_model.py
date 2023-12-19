import torch
from model import face_recognition
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader

def test_model(model, test_loader, threshold=0.5):
    model.eval()
    correct_predictions = 0
    total_samples = 0
    y_true = []
    y_pred = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move the model to the device
    model = model.to(device)

    for idx, (img_anchor, img_positive, img_negative, labels) in enumerate(test_loader):
        # Unpack the tensors from the tuple
        anchor = img_anchor[0].to(device)
        positive = img_positive[0].to(device)
        negative = img_negative[0].to(device)

        # Set the model to evaluation mode
        with torch.no_grad():
            embeddings_anchor = model(anchor.unsqueeze(0))  # Add batch dimension
            embeddings_positive = model(positive.unsqueeze(0))
            embeddings_negative = model(negative.unsqueeze(0))

        # Assuming you have a function to check face recognition (similar to what you implemented)
        is_same_person = face_recognition(embeddings_anchor[0], embeddings_positive[0], threshold)

        # Check if the model's prediction is correct
        if is_same_person:
            correct_predictions += 1  # Consider only when is_same_person is True
            y_pred.append(1)
        else:
            y_pred.append(0)
        total_samples += 1

        y_true.append(1)

    accuracy = correct_predictions / total_samples
    print("correct_predictions: ", correct_predictions)
    print("total_samples:", total_samples)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)
