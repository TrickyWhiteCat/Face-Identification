import torch

def face_recognition(embedding1, embedding2, threshold=0.5):
    distance = torch.norm(embedding1 - embedding2, p=2)

    print("SIMILARITIES:", distance)
    return distance < threshold