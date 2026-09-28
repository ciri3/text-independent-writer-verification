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
from src.evaluation import collect_distances, find_best_threshold, calculate_metrics

def main():
    BATCH_SIZE = 32
    EPOCHS = 15
    TRAIN_PAIRS = 32768
    VAL_PAIRS = 4096
    TEST_PAIRS = 8192
    LEARNING_RATE = 0.001
    EMBEDDING_DIM = 128
    MARGIN = 1.0

    IMAGE_HEIGHT = 64

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print(f"Utilizzo device: {device}")

    transform = Compose([
      ResizeHeight(IMAGE_HEIGHT),
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
      number_of_pairs=TRAIN_PAIRS,
    )

    val_dataset = SiameseDataset(
      dataset,
      val_indices,
      number_of_pairs=VAL_PAIRS,
      fixed=True,
      seed=42
    ) 

    test_dataset = SiameseDataset(
      dataset,
      test_indices,
      number_of_pairs=TEST_PAIRS,
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
        torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "val_loss": val_loss,
        "batch_size": BATCH_SIZE,
        "train_pairs": TRAIN_PAIRS,
        "val_pairs": VAL_PAIRS,
        "test_pairs": TEST_PAIRS,
        "learning_rate": LEARNING_RATE,
        "embedding_dim": EMBEDDING_DIM,
        "margin": MARGIN,
        "image_height": IMAGE_HEIGHT
        }, "best_model.pth")

      print(f"Epoch {epoch + 1}/{EPOCHS} - Train loss: {epoch_loss:.4f} - Val loss: {val_loss:.4f}")

    #model.load_state_dict(torch.load("best_model.pth", map_location=device))
    checkpoint = torch.load("best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Caricato modello migliore: epoca {checkpoint['epoch']}, val loss: {checkpoint['val_loss']:.4f}")
    model.eval()

    val_distances, val_labels = collect_distances(model, val_loader, device)
    best_threshold, best_accuracy = find_best_threshold(val_distances, val_labels)

    print(f"Best threshold: {best_threshold:.4f} - Validation accuracy: {best_accuracy:.4f}")
    torch.save({
        "distances": val_distances,
        "labels": val_labels,
        "threshold": best_threshold,
        "accuracy": best_accuracy
    }, "validation_results.pt")

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
    test_distances, test_labels = collect_distances(model, test_loader, device)
    test_metrics = calculate_metrics(test_distances, test_labels, best_threshold)
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1: {test_metrics['f1']:.4f}")
    print(f"TP: {test_metrics['true_positive']} - TN: {test_metrics['true_negative']} - FP: {test_metrics['false_positive']} - FN: {test_metrics['false_negative']}")

    torch.save({
    "distances": test_distances,
    "labels": test_labels,
    "threshold": best_threshold,
    "loss": test_loss,
    "metrics": test_metrics
    }, "test_results.pt")


if __name__ == "__main__":
    main()