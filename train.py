from torch.utils.data import DataLoader
from torchvision import transforms
from src.dataset import IAMWriterPairsDataset
from src.losses import ContrastiveLoss
from src.models.siamese import SiameseNetwork
import torch
import torch.optim as optim


def main():
  # 1. Configurazione dei parametri di base
  BATCH_SIZE = 32
  EPOCHS = 5
  LEARNING_RATE = 0.001
  EMBEDDING_DIM = 128
  MARGIN = 1.0

  # Usiamo la GPU se disponibile, altrimenti CPU (funziona ovunque senza rompersi)
  device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu")
  print(f"Utilizzo device: {device}")

  # 2. Trasformazioni sulle immagini (Resize a 128x128 e conversione in tensore)
  transform = transforms.Compose([
      transforms.Resize((128, 128)),
      transforms.ToTensor(),
      transforms.Normalize(
          mean=[0.5], std=[0.5]
      ),  # Normalizzazione standard in [-1, 1]
  ])

  # 3. Inizializzazione del Dataset e del DataLoader
  print("Caricamento dataset e generazione coppie...")
  dataset = IAMWriterPairsDataset(data_dir="data", transform=transform)
  dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
  print(f"Totale coppie generate: {len(dataset)}")

  # 4. Inizializzazione Modello, Loss e Optimizer
  model = SiameseNetwork(embedding_dim=EMBEDDING_DIM).to(device)
  criterion = ContrastiveLoss(margin=MARGIN)
  optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

  # 5. Ciclo di Training
  print("Avvio del training...")
  for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for batch_idx, (img1, img2, labels) in enumerate(dataloader):
      img1 = img1.to(device)
      img2 = img2.to(device)
      labels = labels.to(device).unsqueeze(
          1
      )  # Adatta la forma per la loss [batch_size, 1]

      # Zero dei gradienti
      optimizer.zero_grad()

      # Forward pass attraverso la Siamese Network
      output1, output2 = model(img1, img2)

      # Calcolo della Loss
      loss = criterion(output1, output2, labels)

      # Backward pass e ottimizzazione
      loss.backward()
      optimizer.step()

      running_loss += loss.item()

      if batch_idx % 10 == 0:
        print(
            f"Epoch [{epoch+1}/{EPOCHS}], Batch [{batch_idx}/{len(dataloader)}]"
            f" - Loss: {loss.item():.4f}"
        )

    epoch_loss = running_loss / len(dataloader)
    print(f"==> Fine Epoch {epoch+1} - Loss Media: {epoch_loss:.4f}\n")

  # 6. Salvataggio dei pesi finali del modello
  torch.save(model.state_dict(), "siamese_writer_model.pth")
  print(
      "Training completato! Modello salvato con successo in"
      " 'siamese_writer_model.pth'."
  )


if __name__ == "__main__":
  main()