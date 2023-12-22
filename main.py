from resource import train,dataset_handling, vgg16
from torch.utils.data import DataLoader,random_split
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
#parameter
my_model=vgg16.VGG16_NET()
epochs=20
batch_size=64
optimizer=torch.optim.Adam(my_model.parameters(),lr=0.01)
file_name='model.pth'
subset_data_link="E:\deep learning dataset\ms1m-retinaface-t1\imgs_subset"
# Define transformations including normalization
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
# dataset=torchvision.datasets.ImageFolder(root=subset_data_link,transform=transform)
dataset=dataset_handling.MyDataset(root=subset_data_link,transform=transform)

# Define the size of each subset (e.g., 80% train, 10% validation, 10% test)
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
val_size = int(0.1 * dataset_size)
test_size = dataset_size - train_size - val_size

# Use random_split to create the subsets
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders for each subset
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # No need to shuffle for validation
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # No need to shuffle for testing

train_loss,val_loss=train.train(
    num_epochs=epochs,
    model=my_model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    optimizer=optimizer,
    patience=3,
    file_name=file_name
)
#plot result
plt.plot(train_loss,val_loss)
epochs = range(1, len(train_loss) + 1)
plt.plot(epochs, train_loss, label='train loss')
plt.plot(epochs, val_loss, label='validation loss')

plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(epochs)  # Set x-ticks to start from 1

plt.legend()  # Display legend

plt.show()