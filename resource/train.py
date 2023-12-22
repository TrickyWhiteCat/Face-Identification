import torch

def train(num_epochs,model,train_dataloader,val_dataloader,optimizer,patience,file_name):
    criterion=torch.nn.TripletMarginLoss()
    train_loss_saving=[]
    val_loss_saving=[]
    best_val_loss = float('inf')
    no_improvement_counter = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in range(num_epochs):
        print('start epoch', epoch + 1)
        model.train()  
        counting =0.2
        for batch_idx, (anchor,positive,negative) in enumerate(train_dataloader):
            anchor,positive,negative= anchor.to(device),positive.to(device),negative.to(device)
            optimizer.zero_grad()
            anchor_out = model(anchor)
            positive_out=model(positive)
            negative_out=model(negative)
            loss = criterion(anchor_out,positive_out,negative_out)  
            loss.backward() 
            optimizer.step() 

            if (batch_idx + 1) % (len(train_dataloader) // 5) == 0:
                print('training process complete',counting,'is:', loss.item())
                counting+=0.2
        train_loss_saving.append(loss.item())



        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            val_loss=0.0
            for anchor,positive,negative in val_dataloader:
                anchor,positive,negative = anchor.to(device),positive.to(device),negative.to(device)
                anchor_out = model(anchor) 
                positive_out=model(positive)
                negative_out=model(negative)
                val_loss = criterion(anchor,positive,negative)
                val_loss += criterion(anchor_out, positive_out, negative_out).item()
            
        val_loss /= len(val_dataloader)
        val_loss_saving.append(val_loss)


        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}")
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_counter = 0
            torch.save(model.state_dict(), file_name)
        else:
            no_improvement_counter += 1
            if no_improvement_counter >= patience:
                print(f"No improvement in validation loss for {patience} epochs. Early stopping.")
                break
    return train_loss_saving,val_loss_saving
        
        
