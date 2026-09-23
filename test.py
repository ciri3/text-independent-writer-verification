from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from src.models.siamese import SiameseNetwork
from src.dataset import IAMWriterPairsDataset


def main():
    # 1. Configurazione del device (GPU o CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Utilizzo device per il test: {device}")

    # 2. Caricamento del modello con i pesi salvati
    model_path = "siamese_writer_model.pth"
    if not Path(model_path).exists():
        print(f"Errore: File dei pesi '{model_path}' non trovato. Esegui prima 'train.py'.")
        return

    embedding_dim = 128
    model = SiameseNetwork(embedding_dim=embedding_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Modalità valutazione
    print(f"Modello caricato correttamente da '{model_path}'.")

    # 3. Trasformazioni identiche a quelle usate nel training
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # 4. Caricamento del dataset di test
    print("Caricamento campioni dal dataset...")
    dataset = IAMWriterPairsDataset(data_dir="data", transform=transform)

    # 5. Estrapoliamo non solo i tensori ma anche i percorsi reali e la label dal dataset di coppie
    # (Nota: modifichiamo un attimo l'accesso per recuperare i path testuali)
    sample_idx = 0  # Puoi cambiare indice per testare coppie diverse (es. 0, 1, 5, 10...)
    img_path_a, img_path_b, label = dataset.pairs[sample_idx]
    
    # Carichiamo le immagini originali per l'inferenza
    img_a = Image.open(img_path_a).convert("L")
    img_b = Image.open(img_path_b).convert("L")

    # Applichiamo le trasformazioni e aggiungiamo la dimensione del batch [1, C, H, W]
    img1_tensor = transform(img_a).unsqueeze(0).to(device)
    img2_tensor = transform(img_b).unsqueeze(0).to(device)

    # 6. Inferenza
    with torch.no_grad():
        out1, out2 = model(img1_tensor, img2_tensor)
        distance = F.pairwise_distance(out1, out2).item()

    # 7. Stampa dei risultati e dei percorsi per la verifica
    print("\n" + "="*40)
    print("REPORT DI TEST - VERIFICA SCRITTORE")
    print("="*40)
    print(f"Immagine A: {img_path_a}")
    print(f"Immagine B: {img_path_b}")
    print("-"*40)
    print(f"Etichetta Reale: {'Stesso Autore (1.0)' if label == 1.0 else 'Autori Diversi (0.0)'}")
    print(f"Distanza Euclidea: {distance:.4f}")
    
    # Soglia di decisione di esempio
    threshold = 0.5 
    prediction = "Stesso Autore" if distance < threshold else "Autori Diversi"
    print(f"Verdetto del Modello (soglia={threshold}): {prediction}")
    print("="*40)


if __name__ == "__main__":
    main()