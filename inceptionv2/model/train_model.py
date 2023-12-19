from model import InceptionV2, TripletLoss
import os
import torch
from torch import nn, optim
import wandb
from torchvision.transforms import ToPILImage
import numpy as np



def train_model(data_loader, valid_loader, save_file="inception_model.pth", wandb_project=None, train_transform=None, early_stop_patience=3, epoch = 10):
    embeddings_file="saved_embeddings.npy"
    
    model = InceptionV2().to(device)
    if os.path.exists(save_file):
        print(f"Loading pretrained weights from {save_file}")
        model.load_state_dict(torch.load(save_file))
        
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = TripletLoss()

    if wandb_project:
        wandb.login(key="b9575849263a9312a73f76d71d270c8751628e10")
        wandb.init(project=wandb_project)
        wandb.watch(model)

    best_validation_loss = float('inf')
    patience = early_stop_patience
    epoch_no_improve = 0
    saved_embeddings = [] 

    for epoch in range(epoch):
        total_loss = 0.0
        validation_loss = 0.0

        model.train()

        # Inside the training loop
        for idx, (img_anchor, img_positive, img_negative, labels) in enumerate(data_loader):
            anchor, positive, negative = img_anchor.to(device), img_positive.to(device), img_negative.to(device)
            optimizer.zero_grad()
            # Model returns (x, aux, embedding)
            outputs_anchor, aux_anchor, embedding_anchor = model(anchor)
            outputs_positive, aux_positive, embedding_positive = model(positive)
            outputs_negative, aux_negative, embedding_negative = model(negative)

            loss = criterion(embedding_anchor, embedding_positive, embedding_negative)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
                        
            if idx % 10 == 0:
                to_pil_image = ToPILImage()
                pil_image = to_pil_image(img_anchor[0].cpu())  # Convert the tensor to a PIL Image
                sample_transformed_image = train_transform(pil_image)  # Apply transform to the original PIL Image
                wandb.log({"example_transformed_image": [wandb.Image(sample_transformed_image, caption="Transformed Image")]})
        
        # Validation loop
        model.eval()
        with torch.no_grad():
            for val_idx, (val_anchor, val_positive, val_negative, val_labels) in enumerate(valid_loader):
                val_anchor, val_positive, val_negative = val_anchor.to(device), val_positive.to(device), val_negative.to(device)

                val_outputs_anchor, val_aux_anchor,val_embeddings_anchor = model(val_anchor)
                val_outputs_positive, val_aux_positive, val_embeddings_positive = model(val_positive)
                val_outputs_negative, val_aux_negative, val_embeddings_negative = model(val_negative)

                val_loss = criterion(val_embeddings_anchor, val_embeddings_positive, val_embeddings_negative)
                validation_loss += val_loss.item()

        validation_loss /= len(valid_loader)
        
        # Log loss values
        wandb.log({"Training Loss": total_loss, "Validation Loss": validation_loss})
        # Save the embeddings to a file
        np.save(embeddings_file, np.array(saved_embeddings))
        
        # Save model if validation loss is the best so far
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(model.state_dict(), save_file)
            epoch_no_improve = 0
        else:
            epoch_no_improve += 1

        print(f"Epoch {epoch + 1}, Training Loss: {total_loss}, Validation Loss: {validation_loss}")

        if epoch_no_improve >= patience:
            print(f"No improvement in validation loss for {patience} consecutive epochs. Early stopping.")
            break
    print(f"Model is saved to {save_file}.")
