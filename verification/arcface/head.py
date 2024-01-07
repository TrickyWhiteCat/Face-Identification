from torch import nn

class ArcFaceEmbeddingHead(nn.Module):
    def __init__(self, embedding_size, in_features, dropout=0.2, last_batchnorm=True):
        super().__init__()
        self.batchnorm1 = nn.BatchNorm1d(in_features)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_features, embedding_size, bias=True)
        self.features = nn.BatchNorm1d(embedding_size) if last_batchnorm else nn.Identity()
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

    def forward(self, x):
        x = self.batchnorm1(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.features(x)
        return nn.functional.normalize(x, dim=-1)