import torch
import torch.nn as nn
import torch.nn.functional as F


class HandwritingEncoder(nn.Module):
  """Encoder CNN di base per estrarre feature dallo stile di scrittura."""

  def __init__(self, embedding_dim: int =128) -> None:
    super(HandwritingEncoder, self).__init__()

    self.features = nn.Sequential(
        # Blocco Convoluzionale 1
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),  # Dimezza le dimensioni spaziali
        # Blocco Convoluzionale 2
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        # Blocco Convoluzionale 3
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),  # Riduce a 1x1 per ogni canale
    )

    self.fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(128, embedding_dim),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.features(x)
    x = self.fc(x)
    # Normalizzazione L2 lungo la dimensione delle feature (embeddings di una dimensione più piccoli)
    x = F.normalize(x, p=2, dim=1)
    return x


class SiameseNetwork(nn.Module):
  """Siamese Network che condivide gli stessi pesi per confrontare due immagini."""

  def __init__(self, embedding_dim: int = 128) -> None:
    super(SiameseNetwork, self).__init__()
    self.encoder = HandwritingEncoder(embedding_dim=embedding_dim)

  def forward_one(self, x: torch.Tensor) -> torch.Tensor:
    return self.encoder(x)

  def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Ottiene gli embedding per entrambe le immagini in parallelo
    output1 = self.forward_one(input1)
    output2 = self.forward_one(input2)
    return output1, output2