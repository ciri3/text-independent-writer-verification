import random
import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, RandomAffine

from src.dataset import IAMDataset
from src.transforms import ResizeAndPad, NormalizeContrast
from src.split import split_by_writer
from src.siamese_dataset import SiameseDataset
from src.collate import pad_collate
from src.losses import ContrastiveLoss
from src.models.siamese import SiameseNetwork
from src.evaluation import collect_distances, find_best_threshold, evaluate_and_plot_embeddings
from src.run_manager import create_run_dir

import time
from pathlib import Path
from src.logger import Logger



#bisogna usare un seed per standirzzare tutti i random del programma

def main():

    # ============================================================
    #  CONFIGURAZIONE
    # ============================================================

    #seed generale 
    SEED = 42

    PLOT_INTERVAL = 8
    PLOT_NUM_WRITERS = 10
    PLOT_SAMPLES_PER_WRITER = 30
    PLOT_SAMPLE_SEED = 42 #determina quali immagine venongono selezionate per il plot


    CHECKPOINT_PATH = None

    SPLIT_SEED = 42         # Seed per la divisione train/val/test 
    VAL_PAIRS_SEED = 42

    #RESUME_TRAINING = True # Se True, riprende l'addestramento dal checkpoint salvato
    BATCH_SIZE = 32
    EPOCHS = 60             
    TRAIN_PAIRS = 32768
    VAL_PAIRS = 4096

    LEARNING_RATE = 0.001  
    EMBEDDING_DIM = 128
    MARGIN = 1.0

    IMAGE_HEIGHT = 64
    IMAGE_MAX_WIDTH = 320

    # ============================================================
    #  RANDOM SEED
    # ============================================================

    random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # ============================================================
    #  OUTPUT E CHECKPOINT
    # ============================================================

    # Configurazione cartella di output
    output_dir = create_run_dir(base_dir="outputs")  # Crea una nuova cartella di run in outputs
    plots_dir = output_dir / "plots"
    plots_dir.mkdir()


    # File prodotti dalla nuova run
    output_model_filename = output_dir / "best_model.pth"
    val_results_filename = output_dir / "validation_results.pt"
    log_filename = "training_log.txt"

    checkpoint_path = None

    if CHECKPOINT_PATH is not None:
      checkpoint_path = Path(CHECKPOINT_PATH)

      if not checkpoint_path.exists():
          raise FileNotFoundError( f"Checkpoint non trovato: {checkpoint_path}")

    # ============================================================
    #  LOGGER E DEVICE
    # ============================================================

    log = Logger(output_dir, filename=log_filename)       # salvataggio nel file di log specifico
    print(f"File di output salvati in: {output_dir.resolve()}")

    if checkpoint_path is None:
      log.log("Parent checkpoint: none (training from scratch)")
    else:
      log.log(f"Parent checkpoint: {checkpoint_path}")

    log.log(
        f"Parametri di addestramento:\n"
        f"seed={SEED}\n"
        f"batch_size={BATCH_SIZE}\n"
        f"epochs={EPOCHS}\n"
        f"train_pairs={TRAIN_PAIRS}\n"
        f"val_pairs={VAL_PAIRS}\n"
        f"split_seed={SPLIT_SEED}\n"
        f"val_pairs_seed={VAL_PAIRS_SEED}\n"
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

    # ============================================================
    #  DATASET E SPLIT
    # ============================================================

    base_transform = Compose([
      NormalizeContrast(cutoff=1),
      ResizeAndPad(height=IMAGE_HEIGHT, max_width=IMAGE_MAX_WIDTH)
    ])

    dataset = IAMDataset(
      data_dir="data",
      granularity="words",
      transform=base_transform
    )

    train_indices, val_indices, _ = split_by_writer(dataset, seed=SPLIT_SEED)

    # ============================================================
    #  SIAMESE DATASETS E DATALOADERS
    # ============================================================

    #shear=1
    train_transform = Compose([
      RandomAffine(degrees=3, translate=(0.03, 0.08), scale=(0.95, 1.05),fill=255),
      ToTensor()
    ])

    eval_transform = Compose([
        ToTensor()  
    ])

    train_dataset = SiameseDataset(
      dataset,
      train_indices,
      number_of_pairs=TRAIN_PAIRS,
      transform=train_transform
    )

    val_dataset = SiameseDataset(
      dataset,
      val_indices,
      number_of_pairs=VAL_PAIRS,
      fixed=True,
      seed=VAL_PAIRS_SEED, 
      transform=eval_transform
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

    log.log(f"Immagini training: {len(train_indices)}")
    log.log(f"Coppie per epoca: {len(train_dataset)}")
    log.log(f"Batch per epoca: {len(train_loader)}")

    # ============================================================
    #  MODELLO, LOSS E OPTIMIZER
    # ============================================================

    model = SiameseNetwork(embedding_dim=EMBEDDING_DIM).to(device)
    criterion = ContrastiveLoss(margin=MARGIN)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    start_epoch = 0

    # ============================================================
    #  CARICAMENTO CHECKPOINT (OPZIONALE)
    # ============================================================

    if checkpoint_path is not None:
      log.log(f"Ripresa del training dal checkpoint: {checkpoint_path}")

      checkpoint = torch.load(checkpoint_path, map_location=device)

      model.load_state_dict(checkpoint["model_state_dict"])

      if "optimizer_state_dict" in checkpoint:
          optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
          
          for param_group in optimizer.param_groups:
            param_group["lr"] = LEARNING_RATE
          log.log(
              "Stato dell'optimizer caricato con successo. "
              f"Learning rate impostato a {LEARNING_RATE}.")
      else:
          log.log(
              "Stato dell'optimizer non trovato. "
              "Verrà utilizzato il nuovo optimizer."
          )

      start_epoch = checkpoint["epoch"]
      best_val_loss = checkpoint["val_loss"]

      log.log(
          f"Ripresa dall'epoca {start_epoch + 1} "
          f"(Val Loss precedente: {best_val_loss:.4f})"
      )

    else:
        log.log("Training da zero: nessun checkpoint caricato.")

    # ============================================================
    #  TRAINING E VALIDAZIONE
    # ============================================================

    total_start_time = time.time()    # Istante partenza training
    epoch_times = []

    end_epoch = start_epoch + EPOCHS
    for epoch in range(start_epoch,end_epoch):
      epoch_start_time = time.time()  # Istante partenza epoca
      model.train()
      running_loss = 0.0

      # si potrebbe riscrivere come 
      #for batch in train_loader:
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
      
      should_save_plot = ((epoch + 1) == 1 or (epoch + 1) % PLOT_INTERVAL == 0 or (epoch + 1) == end_epoch)       

      if should_save_plot:
        evaluate_and_plot_embeddings(
            base_dataset=dataset, 
            val_indices=val_indices, 
            model=model, 
            device=device, 
            epoch=epoch + 1, 
            output_dir=plots_dir,
            num_writers=PLOT_NUM_WRITERS,
            samples_per_writer=PLOT_SAMPLES_PER_WRITER,
            sample_seed=PLOT_SAMPLE_SEED,
            save_plot=True # t-SNE si disegna solo se should_save_plot è True (lento)
        )
      

      if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
          "epoch": epoch + 1,
          "model_state_dict": model.state_dict(),
          "optimizer_state_dict": optimizer.state_dict(), # Salvataggio stato optimizer
          "val_loss": val_loss,

          "parent_checkpoint": CHECKPOINT_PATH,

          "seed": SEED,

          "batch_size": BATCH_SIZE,
          "train_pairs": TRAIN_PAIRS,
          "val_pairs": VAL_PAIRS,
          "learning_rate": LEARNING_RATE,
          "embedding_dim": EMBEDDING_DIM,
          "margin": MARGIN,
          "image_height": IMAGE_HEIGHT,
          "image_max_width": IMAGE_MAX_WIDTH,

          "split_seed": SPLIT_SEED,
          "val_pairs_seed": VAL_PAIRS_SEED
        }, output_model_filename)

      log.log(f"Epoch {epoch + 1}/{end_epoch}: {epoch_duration:.2f}s - Train loss: {epoch_loss:.4f} - Val loss: {val_loss:.4f}")

    total_duration = time.time() - total_start_time
    avg_epoch_duration = sum(epoch_times) / len(epoch_times)
    log.log(f"Training completato in {total_duration / 60:.2f}min, con una durata media per epoca di {avg_epoch_duration:.2f}s\n"+"-"*40)

    # ============================================================
    #  CARICAMENTO DEL MIGLIOR MODELLO DELLA RUN
    # ============================================================

    if not output_model_filename.exists():
      log.log(
          "Nessun nuovo modello ha migliorato il checkpoint di partenza. "
          "La run termina senza salvare un nuovo modello."
      )
      log.close()
      return

    checkpoint = torch.load(output_model_filename, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    log.log(f"Caricato modello migliore: epoca {checkpoint['epoch']}, val loss: {checkpoint['val_loss']:.4f}")
    model.eval()

    best_plot_path = plots_dir / f"embedding_space_epoch_{checkpoint['epoch']}.png"

    #creiamo il grafico del miglior modello
    if not best_plot_path.exists():
      evaluate_and_plot_embeddings(
      base_dataset=dataset,
      val_indices=val_indices,
      model=model,
      device=device,
      epoch=checkpoint["epoch"],
      output_dir=plots_dir,
      num_writers=PLOT_NUM_WRITERS,
      samples_per_writer=PLOT_SAMPLES_PER_WRITER,
      sample_seed=PLOT_SAMPLE_SEED,
      save_plot=True
      )

    # ============================================================
    #  CALCOLO THRESHOLD SUL VALIDATION SET
    # ============================================================

    val_distances, val_labels = collect_distances(model, val_loader, device)
    best_threshold, best_accuracy = find_best_threshold(val_distances, val_labels)

    log.log(f"Best threshold: {best_threshold:.4f} - Validation accuracy: {best_accuracy:.4f}")
    torch.save({
        "distances": val_distances,
        "labels": val_labels,
        "threshold": best_threshold,
        "accuracy": best_accuracy
    }, val_results_filename)

    print(f"-"*40+f"\nTutti i file e i log salvati correttamente in: {output_dir}")
    log.close()


if __name__ == "__main__":
    main()