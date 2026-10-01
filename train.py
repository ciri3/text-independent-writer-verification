import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

from src.dataset import IAMDataset
from src.transforms import ResizeAndPad
from src.split import split_by_writer
from src.siamese_dataset import SiameseDataset
from src.collate import pad_collate
from src.losses import ContrastiveLoss
from src.models.siamese import SiameseNetwork
from src.evaluation import collect_distances, find_best_threshold, calculate_metrics

import time
from pathlib import Path
from src.logger import Logger

def main():

    # Configurazione cartella di output
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    log = Logger(output_dir) 
    print(f"File di output salvati in: {output_dir.resolve()}")
    
    BATCH_SIZE = 32
    EPOCHS = 40
    TRAIN_PAIRS = 32768
    VAL_PAIRS = 4096
    TEST_PAIRS = 8192
    LEARNING_RATE = 0.001
    EMBEDDING_DIM = 128
    MARGIN = 1.0

    IMAGE_HEIGHT = 64
    IMAGE_MAX_WIDTH = 320

    log.log(
        f"Parametri di addestramento:\n"
        f"batch_size={BATCH_SIZE}\n"
        f"epochs={EPOCHS}\n"
        f"train_pairs={TRAIN_PAIRS}\n"
        f"val_pairs={VAL_PAIRS}\n"
        f"test_pairs={TEST_PAIRS}\n"
        f"learning_rate={LEARNING_RATE}\n"
        f"embedding_dim={EMBEDDING_DIM}\n"
        f"margin={MARGIN}\n"
        f"image_height={IMAGE_HEIGHT}\n"
        f"image_max_width={IMAGE_MAX_WIDTH}\n"
        + "-" * 40
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    log.log(f"Utilizzo device: {device}")

    transform = Compose([
      ResizeAndPad(height=IMAGE_HEIGHT, max_width=IMAGE_MAX_WIDTH),
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

    log.log(f"Immagini training: {len(train_indices)}")
    log.log(f"Coppie per epoca: {len(train_dataset)}")
    log.log(f"Batch per epoca: {len(train_loader)}")

    model = SiameseNetwork(embedding_dim=EMBEDDING_DIM).to(device)
    criterion = ContrastiveLoss(margin=MARGIN)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    total_start_time = time.time()    # Istante partenza training
    epoch_times = []
    for epoch in range(EPOCHS):
      epoch_start_time = time.time()  # Istante partenza epoca
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

      epoch_duration = time.time() - epoch_start_time
      epoch_times.append(epoch_duration)
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
        }, "outputs/best_model.pth")

      log.log(f"Epoch {epoch + 1}/{EPOCHS}: {epoch_duration:.2f}s - Train loss: {epoch_loss:.4f} - Val loss: {val_loss:.4f}")

    total_duration = time.time() - total_start_time
    avg_epoch_duration = sum(epoch_times) / len(epoch_times)
    log.log(f"Training completato in {total_duration / 60:.2f}min, con una durata media per epoca di {avg_epoch_duration:.2f}s\n"+"-"*40)

    #model.load_state_dict(torch.load("best_model.pth", map_location=device))
    checkpoint = torch.load("outputs/best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    log.log(f"Caricato modello migliore: epoca {checkpoint['epoch']}, val loss: {checkpoint['val_loss']:.4f}")
    model.eval()

    val_distances, val_labels = collect_distances(model, val_loader, device)
    best_threshold, best_accuracy = find_best_threshold(val_distances, val_labels)

    log.log(f"Best threshold: {best_threshold:.4f} - Validation accuracy: {best_accuracy:.4f}")
    torch.save({
        "distances": val_distances,
        "labels": val_labels,
        "threshold": best_threshold,
        "accuracy": best_accuracy
    }, "outputs/validation_results.pt")

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
    log.log(f"Test loss: {test_loss:.4f}")
    test_distances, test_labels = collect_distances(model, test_loader, device)
    test_metrics = calculate_metrics(test_distances, test_labels, best_threshold)
    log.log(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    log.log(f"Precision: {test_metrics['precision']:.4f}")
    log.log(f"Recall: {test_metrics['recall']:.4f}")
    log.log(f"F1: {test_metrics['f1']:.4f}")
    log.log(f"TP: {test_metrics['true_positive']} - TN: {test_metrics['true_negative']} - FP: {test_metrics['false_positive']} - FN: {test_metrics['false_negative']}")

    torch.save({
    "distances": test_distances,
    "labels": test_labels,
    "threshold": best_threshold,
    "loss": test_loss,
    "metrics": test_metrics
    }, "outputs/test_results.pt")

    print(f"-"*40+f"\nTutti i file e i log salvati correttamente in: {output_dir}")
    log.close()


if __name__ == "__main__":
    main()