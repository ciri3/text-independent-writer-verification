from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader

from src.models.siamese import SiameseNetwork
from src.dataset import IAMDataset
from src.split import split_by_writer
from src.siamese_dataset import SiameseDataset
from src.evaluation import (
    collect_distances,
    calculate_metrics,
    calculate_disaggregated_metrics,
    calculate_eer_and_plot_roc,
)
from src.transforms import ResizeAndPad
from src.collate import pad_collate
from src.logger import Logger

def main():

    # ============================================================
    # CONFIGURAZIONE
    # ============================================================

    CHECKPOINT_PATH = "outputs/run_XXX/best_model.pth"

    TEST_PAIRS = 8192
    TEST_PAIRS_SEED = 42

    # ============================================================
    # PERCORSI
    # ============================================================

    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint non trovato: {checkpoint_path}"
        )
    output_dir = checkpoint_path.parent

    val_results_path = output_dir / "validation_results.pt"
    test_results_path = output_dir / "test_results.pt"
    log_filename = "test_log.txt"

    # ============================================================
    # LOGGER E DEVICE
    # ============================================================

    # Inizializziamo il Logger per salvare sia a schermo che su file
    log = Logger(output_dir, filename=log_filename)

    device = torch.device(
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )
    log.log(f"Utilizzo device per il test: {device}")

    # ============================================================
    # CARICAMENTO MODELLO
    # ============================================================

    checkpoint = torch.load(checkpoint_path, map_location=device)
    embedding_dim = checkpoint["embedding_dim"]

    model = SiameseNetwork(embedding_dim=embedding_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval() # Modalità valutazione
    log.log(f"Modello caricato correttamente da '{checkpoint_path}' (Epoca {checkpoint['epoch']}).")

    # ============================================================
    # CARICAMENTO THRESHOLD
    # ============================================================

    # Recupero della soglia ottimale salvata in validazione
    if not val_results_path.exists():
        log.log(f"Validation results non trovati: {val_results_path}")
        log.close()
        return
    val_data = torch.load(val_results_path, map_location=device)
    best_threshold = val_data["threshold"]
    log.log(
        f"Soglia ottimale caricata dal validation set: "
        f"{best_threshold:.4f}"
    )

    # ============================================================
    # PREPROCESSING E DATASET
    # ============================================================

    # Aggiornamento delle trasformazioni con ResizeAndPad
    image_height = checkpoint["image_height"]
    image_max_width = checkpoint["image_max_width"]
    split_seed = checkpoint["split_seed"]
    
    transform = Compose([
        ResizeAndPad(height=image_height, max_width=image_max_width),
        ToTensor()
    ])

    # Ricostruzione esatta del Test Dataset (stesso split e stesso seed del training)
    #log.log("Caricamento dataset e configurazione test set...")
    dataset = IAMDataset(data_dir="data", granularity="words", transform=transform)

    _, _, test_indices = split_by_writer(dataset, seed=split_seed)

    # ============================================================
    # TEST DATASET E DATALOADER
    # ============================================================
    
    test_dataset = SiameseDataset(
        dataset,
        test_indices,
        number_of_pairs=TEST_PAIRS,
        fixed=True,
        seed=TEST_PAIRS_SEED
    )
    
    batch_size = checkpoint["batch_size"]

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pad_collate
    )

    
    # ============================================================
    # CONTROLLO QUALITATIVO
    # ============================================================
    
    log.log("=" * 40)
    log.log("CONTROLLO QUALITATIVO SU ALCUNE COPPIE")
    log.log("Analisi di 5 coppie campione:")
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


    # ============================================================
    # VALUTAZIONE COMPLETA SUL TEST SET
    # ============================================================

    test_distances, test_labels = collect_distances(model, test_loader, device)
    test_metrics = calculate_metrics(test_distances, test_labels, best_threshold)
    # calcolo EER e salvataggio della curva ROC nella cartella di run del test
    test_eer = calculate_eer_and_plot_roc(test_distances, test_labels, output_dir)

    # calcolo metriche disaggregate sulle 4 categorie
    test_pair_types = [
        test_dataset._generate_pair(i)[2] if not test_dataset.fixed
        else test_dataset._fixed_pairs[i][2]
        for i in range(len(test_dataset))
    ]
    disaggregated_test = calculate_disaggregated_metrics(
        test_distances, test_labels, test_pair_types, best_threshold
    )

    log.log("=" * 40)
    log.log("VALUTAZIONE COMPLETA SUL TEST SET")
    log.log(f"Numero totale coppie nel test set: {len(test_dataset)}")
    log.log(f"Seed coppie di test: {TEST_PAIRS_SEED}")
    log.log(f"Soglia applicata: {best_threshold:.4f}")

    log.log(f"- Test accuracy: {test_metrics['accuracy']:.4f}")
    log.log(f"- Precision: {test_metrics['precision']:.4f}")
    log.log(f"- Recall: {test_metrics['recall']:.4f}")
    log.log(f"- F1: {test_metrics['f1']:.4f}")
    log.log(f"- EER (Equal Error Rate): {test_eer:.4f}")
    log.log(f"- TP: {test_metrics['true_positive']} - TN: {test_metrics['true_negative']} - FP: {test_metrics['false_positive']} - FN: {test_metrics['false_negative']}")

    log.log("=" * 40)
    log.log("PRESTAZIONI PER CATEGORIA DI COPPIA:")
    log.log(f"{'Categoria':<35} | {'Accuracy':<10} | {'Dist. Media':<12} | {'Campioni':<8}")
    log.log("-" * 72)
    for cat_name, cat_m in disaggregated_test.items():
        log.log(
            f"{cat_name:<35} | {cat_m['accuracy']:.4f}     | "
            f"{cat_m['mean_distance']:.4f}      | {cat_m['total_samples']:<8}"
        )
    log.log("=" * 40)

    # ============================================================
    # SALVATAGGIO RISULTATI
    # ============================================================

    # Salvataggio strutturato dei risultati in formato .pt (come in train.py)
    torch.save({
        "distances": test_distances,
        "labels": test_labels,
        "threshold": best_threshold,
        "metrics": test_metrics,
        "eer": test_eer,
        "disaggregated_metrics": disaggregated_test
    }, test_results_path)

    log.log(f"\nRisultati strutturati salvati in: {test_results_path}")
    log.log(f"Log testuale salvato tramite Logger in: {output_dir / log_filename}")
    log.close()


if __name__ == "__main__":
    main()