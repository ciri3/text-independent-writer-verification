import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):

    """Contrastive Loss per Siamese Networks.
    
      - Label = 1.0 se stesso writer (spinge gli embedding vicini)
      - Label = 0.0 se writer diversi (spinge gli embedding ad almeno una distanza
      pari a 'margin')
      """
    
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)

        loss_positive = label * torch.pow(euclidean_distance, 2)

        loss_negative = (1.0 - label) * torch.pow(
            torch.clamp(self.margin - euclidean_distance, min=0.0), 2
        )

        return torch.mean(loss_positive + loss_negative)