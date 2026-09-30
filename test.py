from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor
from PIL import Image

from src.models.siamese import SiameseNetwork
from src.dataset import IAMDataset
from src.transforms import ResizeHeight
from src.split import split_by_writer
from src.siamese_dataset import SiameseDataset
from src.evaluation import find_best_threshold, collect_distances, calculate_metrics
from src.collate import pad_collate
from torch.utils.data import DataLoader

def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )
    print(f"Utilizzo device per il test: {device}")

    # Percorsi dei file generati dal training
    output_dir = Path("outputs")
    #model_path = output_dir / "best_model.pth"
    model_path = Path("best_model.pth")
    #val_results_path = output_dir / "validation_results.pt"
    val_results_path = Path("validation_results.pt")

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
    ''' ------------ da reinserire quando train eseguito completo -------------
    best_threshold = 0.5  # Valore di fallback di sicurezza
    if val_results_path.exists():
        val_data = torch.load(val_results_path, map_location=device)
        best_threshold = val_data["threshold"]
        print(f"Soglia ottimale caricata dal validation set: {best_threshold:.4f}")
    else:
        print(f"Avviso: 'validation_results.pt' non trovato. Uso soglia di default: {best_threshold}")
    --------------------------------------------------------------------------
    '''

    # Trasformazioni coerenti con il training (ResizeHeight dinamico)
    image_height = checkpoint.get("image_height", 64)
    transform = Compose([
        ResizeHeight(image_height),
        ToTensor()
    ])

    # Ricostruzione esatta del Test Dataset (stesso split e stesso seed del training)
    print("Caricamento dataset e configurazione test set...")
    dataset = IAMDataset(data_dir="data", granularity="words", transform=transform)
    _, val_indices, test_indices = split_by_writer(dataset)

    # ------------- da cancellare quando train eseguito completo -------------
    val_dataset = SiameseDataset(dataset, val_indices, number_of_pairs=4096, fixed=True, seed=42)
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=pad_collate
    )
    val_distances, val_labels = collect_distances(model, val_loader, device)
    best_threshold, best_accuracy = find_best_threshold(val_distances, val_labels)
    print(f"Best thresh: {best_threshold}, best accuracy: {best_accuracy}")
    # ------------------------------------------------------------------------
    
    test_pairs_count = checkpoint.get("test_pairs", 8192)
    test_dataset = SiameseDataset(
        dataset,
        test_indices,
        number_of_pairs=test_pairs_count,
        fixed=True,
        seed=42
    )

    print("="*40)
    print("INFERENZA - VERIFICA SCRITTORE")
    # Estrazione di una coppia di esempio dal test set
    sample_idx = [4, 79, 215, 343, 555]  # Puoi cambiare indice per testare coppie diverse (es. 0, 1, 5, 10...)
    print(f"Coppie test selezionate: {sample_idx}")
    print("-"*40)
    for idx in sample_idx:
        sample = test_dataset[idx]
    
        # Applicazione trasformazioni e aggiunta della dimensione del batch [1, C, H, W]
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


if __name__ == "__main__":
    main()