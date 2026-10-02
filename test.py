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
from src.logger import Logger

def main():
    # Flag per scegliere quale versione testare:
    # True -> testa il modello aggiornato dal fine-tuning (best_model_updated.pth)
    # False -> testa il modello originale delle prime 40 epoche (best_model.pth)
    USE_UPDATED_MODEL = True

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Percorsi dinamici dei file in base al flag
    if USE_UPDATED_MODEL:
        model_path = output_dir / "best_model_updated.pth"
        val_results_path = output_dir / "validation_results_updated.pt"
        test_results_path = output_dir / "inference_test_results_updated.pt"
        log_filename = "inference_results_updated.txt"
    else:
        model_path = output_dir / "best_model.pth"
        val_results_path = output_dir / "validation_results.pt"
        test_results_path = output_dir / "inference_test_results.pt"
        log_filename = "inference_results.txt"

    # Inizializziamo il Logger per salvare sia a schermo che su file
    log = Logger(output_dir, filename=log_filename)

    device = torch.device(
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )
    log.log(f"Utilizzo device per il test: {device}")
    log.log(f"Modalità test: {'MODELLO AGGIORNATO (Fine-Tuning)' if USE_UPDATED_MODEL else 'MODELLO BASE (Originale)'}")

    if not model_path.exists():
        log.log(f"Errore: File dei pesi '{model_path}' non trovato. Esegui prima il training corrispondente.")
        return

    # Caricamento del modello migliore
    embedding_dim = 128
    model = SiameseNetwork(embedding_dim=embedding_dim).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval() # Modalità valutazione
    log.log(f"Modello caricato correttamente da '{model_path}' (Epoca {checkpoint['epoch']}).")

    # Recupero della soglia ottimale salvata in validazione
    best_threshold = 0
    if val_results_path.exists():
        val_data = torch.load(val_results_path, map_location=device)
        best_threshold = val_data["threshold"]
        log.log(f"Soglia ottimale caricata dal validation set: {best_threshold:.4f}")
    else:
        log.log(f"Avviso: 'validation_results.pt' non trovato\nChiusura...")
        log.close()
        return

    # Aggiornamento delle trasformazioni con ResizeAndPad
    image_height = checkpoint.get("image_height", 64)
    image_max_width = 320
    
    transform = Compose([
        ResizeAndPad(height=image_height, max_width=image_max_width),
        ToTensor()
    ])

    # Ricostruzione esatta del Test Dataset (stesso split e stesso seed del training)
    log.log("Caricamento dataset e configurazione test set...")
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

    log.log("="*40)
    log.log(f"INFERENZA su test set - VERIFICA SCRITTORE ({'MODELLO AGGIORNATO' if USE_UPDATED_MODEL else 'MODELLO BASE'})")
    log.log("\nStampa di alcune coppie di test ...\n"+"-"*40)
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
        log.log(f"Indice coppia test: {idx}")
        log.log(f"Label: {'stesso autore (1)' if label == 1.0 else 'autori diversi (0)'}")
        log.log(f"Distanza Euclidea: {distance:.4f}")
        
        prediction = "stesso autore" if distance < best_threshold else "autori diversi"
        log.log(f"Verdetto modello (soglia ottimale={best_threshold:.4f}): {prediction}")
        log.log("-"*40)

    log.log("Metriche di riferimento per intero test set:")
    test_distances, test_labels = collect_distances(model, test_loader, device)
    test_metrics = calculate_metrics(test_distances, test_labels, best_threshold)
    log.log(f"- Test accuracy: {test_metrics['accuracy']:.4f}")
    log.log(f"- Precision: {test_metrics['precision']:.4f}")
    log.log(f"- Recall: {test_metrics['recall']:.4f}")
    log.log(f"- F1: {test_metrics['f1']:.4f}")
    log.log(f"- TP: {test_metrics['true_positive']} - TN: {test_metrics['true_negative']} - FP: {test_metrics['false_positive']} - FN: {test_metrics['false_negative']}")

    # Salvataggio strutturato dei risultati in formato .pt (come in train.py)
    torch.save({
        "distances": test_distances,
        "labels": test_labels,
        "threshold": best_threshold,
        "metrics": test_metrics
    }, test_results_path)

    log.log(f"\nRisultati strutturati salvati in: {test_results_path}")
    log.log(f"Log testuale salvato tramite Logger in: outputs/{log_filename}")
    log.close()


if __name__ == "__main__":
    main()