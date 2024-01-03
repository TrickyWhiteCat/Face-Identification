from resource import train,vgg16
from pytorch_metric_learning import samplers
from torchvision import transforms,datasets
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
#parameter
my_model=vgg16.VGG16_NET()
epochs=80
n_samples=4
batch_size=64
file_name='model.pth'
subset_data_link="E:\deep learning dataset\ms1m-retinaface-t1\imgs_subset"

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

dataset=datasets.ImageFolder(subset_data_link,transform=transform)
labels=dataset.targets
train_labels=labels[:int(len(dataset)*0.8)]
valid_labels=labels[int(len(dataset)*0.8):]

dataset_size = len(dataset)
train_indices = list(range(int(len(dataset) * 0.8)))
val_indices = list(range(int(len(dataset) * 0.8), len(dataset)))

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

# Create DataLoaders for each subset
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=samplers.MPerClassSampler(train_labels, m=n_samples, length_before_new_iter=len(train_dataset), batch_size=batch_size))
val_dataloader = DataLoader(val_dataset, batch_size=batch_size,sampler=samplers.MPerClassSampler(valid_labels, m=n_samples, length_before_new_iter=len(train_dataset), batch_size=batch_size)) 

train_loss,val_loss=train.train(
    num_epochs=epochs,
    model=my_model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    learning_rate=0.01,
    margin=0.5,
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