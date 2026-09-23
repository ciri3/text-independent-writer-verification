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
    super(ContrastiveLoss, self).__init__()
    self.margin = margin

  def forward(self, output1, output2, label):
    # Distanza euclidea tra i due embedding
    euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)

    # Loss per coppie positive (stesso autore)
    loss_positive = (1.0 - label) * torch.pow(euclidean_distance, 2)

    # Loss per coppie negative (autori diversi)
    loss_negative = label * torch.pow(
        torch.clamp(self.margin - euclidean_distance, min=0.0), 2
    )

    # Media totale della loss
    loss_contrastive = torch.mean(loss_positive + loss_negative)
    return loss_contrastive