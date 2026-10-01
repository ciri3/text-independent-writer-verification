from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader

from src.models.siamese import SiameseNetwork
from src.dataset import IAMDataset
from src.split import split_by_writer
from src.siamese_dataset import SiameseDataset
from src.evaluation import collect_distances, calculate_metrics
from src.transforms import ResizeAndPad
from src.collate import pad_collate

def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )
    print(f"Utilizzo device per il test: {device}")

    # Percorsi dei file generati dal training
    output_dir = Path("outputs")
    model_path = output_dir / "best_model.pth"
    val_results_path = output_dir / "validation_results.pt"

    if not model_path.exists():
        print(f"Errore: File dei pesi '{model_path}' non trovato. Esegui prima 'train.py'.")
        return

    # Caricamento del modello migliore
    embedding_dim = 128
    model = SiameseNetwork(embedding_dim=embedding_dim).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval() # Modalità valutazione
    print(f"Modello caricato correttamente da '{model_path}' (Epoca {checkpoint['epoch']}).")

    # Recupero della soglia ottimale salvata in validazione
    best_threshold = 0
    if val_results_path.exists():
        val_data = torch.load(val_results_path, map_location=device)
        best_threshold = val_data["threshold"]
        print(f"Soglia ottimale caricata dal validation set: {best_threshold:.4f}")
    else:
        print(f"Avviso: 'validation_results.pt' non trovato\nChiusura...")
        return

    # Aggiornamento delle trasformazioni con ResizeAndPad
    image_height = checkpoint.get("image_height", 64)
    image_max_width = 320
    
    transform = Compose([
        ResizeAndPad(height=image_height, max_width=image_max_width),
        ToTensor()
    ])

    # Ricostruzione esatta del Test Dataset (stesso split e stesso seed del training)
    print("Caricamento dataset e configurazione test set...")
    dataset = IAMDataset(data_dir="data", granularity="words", transform=transform)
    _, _, test_indices = split_by_writer(dataset)
    
    test_pairs_count = checkpoint.get("test_pairs", 8192)
    test_dataset = SiameseDataset(
        dataset,
        test_indices,
        number_of_pairs=test_pairs_count,
        fixed=True,
        seed=42
    )

    # Creazione del DataLoader di test per le metriche globali
    batch_size = checkpoint.get("batch_size", 32)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pad_collate
    )

    print("="*40)
    print("INFERENZA su test set - VERIFICA SCRITTORE")
    print("\nStampa di alcune coppie di test ...\n"+"-"*40)
    num_samples_to_print = 5
    total_samples = len(test_dataset)
    step = total_samples // num_samples_to_print
    sample_indices = [i * step for i in range(num_samples_to_print)]
    for idx in sample_indices:
        sample = test_dataset[idx]
    
        img1_tensor = sample["image1"].unsqueeze(0).to(device)
        img2_tensor = sample["image2"].unsqueeze(0).to(device)
        label = sample["label"]

        # Inferenza
        with torch.no_grad():
            out1, out2 = model(img1_tensor, img2_tensor)
            distance = F.pairwise_distance(out1, out2).item()

        # Stampa dei risultati e dei percorsi per la verifica visiva
        print(f"Indice coppia test: {idx}")
        print(f"Label: {'stesso autore (1)' if label == 1.0 else 'autori diversi (0)'}")
        print(f"Distanza Euclidea: {distance:.4f}")
        
        prediction = "stesso autore" if distance < best_threshold else "autori diversi"
        print(f"Verdetto modello (soglia ottimale={best_threshold:.4f}): {prediction}")
        print("-"*40)

    print("Metriche di riferimento per intero test set:")
    test_distances, test_labels = collect_distances(model, test_loader, device)
    test_metrics = calculate_metrics(test_distances, test_labels, best_threshold)
    print(f"- Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"- Precision: {test_metrics['precision']:.4f}")
    print(f"- Recall: {test_metrics['recall']:.4f}")
    print(f"- F1: {test_metrics['f1']:.4f}")
    print(f"- TP: {test_metrics['true_positive']} - TN: {test_metrics['true_negative']} - FP: {test_metrics['false_positive']} - FN: {test_metrics['false_negative']}")


if __name__ == "__main__":
    main()