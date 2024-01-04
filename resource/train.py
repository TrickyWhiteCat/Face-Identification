import torch
from pytorch_metric_learning import losses,miners
def train(num_epochs,model,train_dataloader,val_dataloader,learning_rate,margin,patience,file_name):
    train_loss_saving=[]
    val_loss_saving=[]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    no_improvement_counter = 0
    best_val_loss = float('inf')
    criterion = losses.TripletMarginLoss(margin=margin)
    mining=miners.TripletMarginMiner(margin=margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    gamma = 0.95
    schedular=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma)
    for epoch in range(num_epochs):
        print('start epoch', epoch + 1)
        model.train()  # Set the model to training mode
        counting =0
        for batch_idx, (input, target) in enumerate(train_dataloader):
            input,target=input.to(device),target.to(device)
            optimizer.zero_grad()  
            output = model(input)  
            miner=mining(output,target)
            loss = criterion(output,target,miner)  
            loss.backward() 
            optimizer.step() 

            if (batch_idx + 1) % (len(train_dataloader) // 5) == 0:
                print('training process done ',counting,'is:', loss.item())
                counting+=0.2
        train_loss_saving.append(loss)
        schedular.step()

        model.eval() 
        with torch.no_grad():
            val_loss=0.0
            for batch_idx, (input, target) in enumerate(train_dataloader):
                input,target=input.to(device),target.to(device)
                output = model(input)  # Forward pass
                val_loss += criterion(output,target).item()
        val_loss /= len(val_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {loss.item()}, Validation Loss: {val_loss}")
        val_loss_saving.append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_counter = 0
            file_name='model'+f"_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), file_name)
        else:
            no_improvement_counter += 1
            if no_improvement_counter >= patience:
                print(f"No improvement in validation loss for {patience} epochs. Early stopping.")
                break
    return train_loss_saving,val_loss_saving
        
        
