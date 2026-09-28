import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

from src.dataset import IAMDataset
from src.transforms import ResizeHeight
from src.split import split_by_writer
from src.siamese_dataset import SiameseDataset
from src.collate import pad_collate
from src.losses import ContrastiveLoss
from src.models.siamese import SiameseNetwork


def main():
    BATCH_SIZE = 32
    EPOCHS = 1
    train = 4000
    val = 1000
    test = 1000
    LEARNING_RATE = 0.001
    EMBEDDING_DIM = 128
    MARGIN = 1.0

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print(f"Utilizzo device: {device}")

    transform = Compose([
      ResizeHeight(64),
      ToTensor()
    ])

    dataset = IAMDataset(
      data_dir="data",
      granularity="words",
      transform=transform
    )

    train_indices, val_indices, test_indices = split_by_writer(dataset)

    train_dataset = SiameseDataset(
      dataset,
      train_indices,
      number_of_pairs=train,
    )

    val_dataset = SiameseDataset(
      dataset,
      val_indices,
      number_of_pairs=val,
      fixed=True,
      seed=42
    ) 

    test_dataset = SiameseDataset(
      dataset,
      test_indices,
      number_of_pairs=test,
      fixed=True,
      seed=42
    )

    # le coppie sono gia generate in modo random, quindi non serve shuffle=True, in piu così manteniamo equilibrio tra le classi
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=pad_collate
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=pad_collate
    )    

    test_loader = DataLoader(
      test_dataset,
      batch_size=BATCH_SIZE,
      shuffle=False,
      collate_fn=pad_collate
    ) 

    print(f"Immagini training: {len(train_indices)}")
    print(f"Coppie per epoca: {len(train_dataset)}")
    print(f"Batch per epoca: {len(train_loader)}")

    model = SiameseNetwork(embedding_dim=EMBEDDING_DIM).to(device)
    criterion = ContrastiveLoss(margin=MARGIN)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    for epoch in range(EPOCHS):
      model.train()
      running_loss = 0.0

      for batch_idx, batch in enumerate(train_loader):
          img1 = batch["image1"].to(device)
          img2 = batch["image2"].to(device)
          labels = batch["label"].to(device)
          optimizer.zero_grad()
          output1, output2 = model(img1, img2)

          loss = criterion(output1, output2, labels)
          loss.backward()
          optimizer.step()
          running_loss += loss.item()
  
      epoch_loss = running_loss / len(train_loader)

      model.eval()
      val_running_loss = 0.0
      with torch.no_grad():
        for batch in val_loader:
          img1 = batch["image1"].to(device)
          img2 = batch["image2"].to(device)
          labels = batch["label"].to(device)
          output1, output2 = model(img1, img2)
          loss = criterion(output1, output2, labels)
          val_running_loss += loss.item()

      val_loss = val_running_loss / len(val_loader)
      if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")

      print(f"Epoch {epoch + 1}/{EPOCHS} - Train loss: {epoch_loss:.4f} - Val loss: {val_loss:.4f}")

    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()
    test_running_loss = 0.0

    with torch.no_grad():
        for batch in test_loader:
            img1 = batch["image1"].to(device)
            img2 = batch["image2"].to(device)
            labels = batch["label"].to(device)

            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, labels)
            test_running_loss += loss.item()

    test_loss = test_running_loss / len(test_loader)
    print(f"Test loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()