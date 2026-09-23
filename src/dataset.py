import random
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class IAMWriterPairsDataset(Dataset):

  def __init__(self, data_dir="data", transform=None):
    self.data_dir = Path(data_dir)
    self.transform = transform

    # 1. Parsing automatico delle cartelle
    records = []
    for writer_folder in self.data_dir.iterdir():
      if writer_folder.is_dir():
        writer_id = writer_folder.name
        for img_path in writer_folder.glob("*.png"):
          records.append({"image_path": str(img_path), "writer_id": writer_id})

    self.df = pd.DataFrame(records)
    self.writers = self.df["writer_id"].unique().tolist()

    # Mappa writer -> lista di immagini
    self.writer_to_images = (
        self.df.groupby("writer_id")["image_path"].apply(list).to_dict()
    )

    # 2. Generazione delle coppie positive/negative
    self.pairs = self._generate_pairs()

  def _generate_pairs(self):
    pairs = []
    for writer_id, img_paths in self.writer_to_images.items():
      # Coppie positive (stesso writer)
      if len(img_paths) >= 2:
        for i in range(len(img_paths)):
          for j in range(i + 1, len(img_paths)):
            pairs.append((img_paths[i], img_paths[j], 1.0))

      # Coppie negative (writer diversi)
      for img_path in img_paths:
        other_writer = random.choice([w for w in self.writers if w != writer_id])
        other_img_path = random.choice(self.writer_to_images[other_writer])
        pairs.append((img_path, other_img_path, 0.0))

    random.shuffle(pairs)
    return pairs

  def __len__(self):
    return len(self.pairs)

  def __getitem__(self, idx):
    img_path_a, img_path_b, label = self.pairs[idx]

    img_a = Image.open(img_path_a).convert("L")
    img_b = Image.open(img_path_b).convert("L")

    if self.transform:
      img_a = self.transform(img_a)
      img_b = self.transform(img_b)

    return img_a, img_b, torch.tensor(label, dtype=torch.float32)