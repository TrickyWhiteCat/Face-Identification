from model import InceptionV2, TripletLoss
import os
import torch
from torch import nn, optim
import wandb
from torchvision.transforms import ToPILImage
import numpy as np



def train_model(data_loader, valid_loader, save_file="inception_model.pth", wandb_project=None, train_transform=None, early_stop_patience=3, epoch = 10, resume_checkpoint=None):
    embeddings_file="saved_embeddings.npy"
    
    model = InceptionV2().to(device)
    if os.path.exists(save_file):
        print(f"Loading pretrained weights from {save_file}")
        model.load_state_dict(torch.load(save_file))
        
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = TripletLoss()
    
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    saved_embeddings = [] 
    
    if wandb_project:
        wandb.login(key="b9575849263a9312a73f76d71d270c8751628e10")
        run = wandb.init(project=wandb_project, name='InceptionV2', resume='allow', id=resume_checkpoint)
        wandb.watch(model)
        if run.id != 'new':
            # Load checkpoint and retrieve necessary information
            checkpoint_path = run.restore(name=resume_checkpoint)
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            # Set the initial epoch for training
            epoch_no_improve = checkpoint.get('epoch_no_improve', 0)
            best_validation_loss = checkpoint.get('best_validation_loss', float('inf'))

            print(f"Resuming training from epoch {start_epoch}, with best validation loss: {best_validation_loss}")
        else:
            start_epoch = 0
            best_validation_loss = float('inf')
            epoch_no_improve = 0
    

    else:
        start_epoch = 0
        best_validation_loss = float('inf')
        epoch_no_improve = 0

    patience = early_stop_patience

    for epoch in range(start_epoch, epoch):
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
            saved_embeddings.append(embedding_anchor.detach().cpu().numpy())
            saved_embeddings.append(embedding_positive.detach().cpu().numpy())
            saved_embeddings.append(embedding_negative.detach().cpu().numpy())

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
                        
            if idx % 10 == 0:
                to_pil_image = ToPILImage()
                pil_image = to_pil_image(img_anchor[0].cpu())  # Convert the tensor to a PIL Image
                sample_transformed_image = train_transform(pil_image)  # Apply transform to the original PIL Image
                wandb.log({"example_transformed_image": [wandb.Image(sample_transformed_image, caption="Transformed Image")]})
        total_loss /= len(data_loader)
        if wandb_project:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'epoch_no_improve': epoch_no_improve,
                'best_validation_loss': best_validation_loss
            }, f'checkpoint_{epoch}.pth')

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
        scheduler.step()
        # Log loss values
        wandb.log({"Training Loss": total_loss, "Validation Loss": validation_loss})
        # Save the embeddings to a file
        np.save(embeddings_file, np.vstack(saved_embeddings))

        
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
        if wandb_project:
            wandb.log({"learning_rate": optimizer.param_groups[0]['lr']})
    print(f"Model is saved to {save_file}.")
