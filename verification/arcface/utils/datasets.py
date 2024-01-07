import torch

class SubsetWithTransforms(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        super().__init__()
        self.subset = subset
        self.transform = transform
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
    def __len__(self):
        return len(self.subset)
    
