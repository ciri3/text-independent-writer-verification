import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torchvision.transforms.functional import to_tensor
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from typing import Any



def collect_distances(
    model: torch.nn.Module, 
    data_loader: DataLoader, 
    device: torch.device
    ) -> tuple[list[float], list[int]]:
    """
        Esegue l'inferenza sul DataLoader e raccoglie la lista delle distanze euclidee
        e delle corrispondenti label reali.
    """

    model.eval()

    all_distances = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            img1 = batch["image1"].to(device)
            img2 = batch["image2"].to(device)
            labels = batch["label"]

            output1, output2 = model(img1, img2)
            distances = F.pairwise_distance(output1, output2)

            all_distances.extend(distances.cpu().tolist())
            all_labels.extend(labels.tolist())

    return all_distances, all_labels

def find_best_threshold(distances: list[float], labels: list[int]) -> tuple[float, float]:
    """Trova la soglia di distanza che massimizza l'accuracy sul set fornito."""

    best_threshold = None
    best_accuracy = 0.0
    
    sorted_distances = sorted(set(distances))

    thresholds = []
    for i in range(len(sorted_distances) - 1):
        threshold = (sorted_distances[i] + sorted_distances[i + 1]) / 2
        thresholds.append(threshold)

    for threshold in thresholds:
        correct = 0

        for distance, label in zip(distances, labels):
            prediction = 1 if distance < threshold else 0

            if prediction == label:
                correct += 1

        accuracy = correct / len(labels)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return best_threshold, best_accuracy

def calculate_metrics(distances: list[float], labels: list[float], threshold: float) -> dict[str, Any]:
    """Calcola le metriche classiche di classificazione (Accuracy, Precision, Recall, F1, Matrice di Confusione)."""
    
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    for distance, label in zip(distances, labels):
        prediction = 1 if distance < threshold else 0

        if prediction == 1 and label == 1:
            true_positive += 1
        elif prediction == 0 and label == 0:
            true_negative += 1
        elif prediction == 1 and label == 0:
            false_positive += 1
        else:
            false_negative += 1

    total = len(labels)
    accuracy = (true_positive + true_negative) / total
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "true_positive": true_positive,
    "true_negative": true_negative,
    "false_positive": false_positive,
    "false_negative": false_negative
    }



def evaluate_and_plot_embeddings(
    base_dataset: Any, 
    val_indices: list[int], 
    model: torch.nn.Module, 
    device: torch.device, 
    epoch: int, 
    output_dir: Path | str,
    num_writers: int = 10, 
    samples_per_writer: int = 30, 
    sample_seed: int = 42, 
    save_plot: bool = False
    ) -> None:
    """
    Estrae gli embedding delle immagini di validazione e, se save_plot=True,
    salva la loro visualizzazione t-SNE in formato PNG.
    """
    model.eval()
    
    # Creiamo un DataLoader temporaneo per scorrere le singole immagini di validazione
    # (riutilizzando le stesse trasformazioni del dataset)
    val_subset = Subset(base_dataset, val_indices)

    writer_counts = {}
    for sample in val_subset:
        writer_id = sample["writer_id"]
        if writer_id not in writer_counts:
            writer_counts[writer_id] = 0
        writer_counts[writer_id] += 1

    #print(f"Numero di writer: {len(writer_counts)}")
    #print(f"Minimo immagini per writer: {min(writer_counts.values())}")
    #print(f"Massimo immagini per writer: {max(writer_counts.values())}")

    eligible_writers = [
    writer_id
    for writer_id, count in writer_counts.items()
    if count >= samples_per_writer
    ]

    #print(f"Writer con almeno {samples_per_writer} immagini: "f"{len(eligible_writers)}")

    actual_num_writers = min(num_writers, len(eligible_writers))

    if actual_num_writers == 0:
        raise ValueError(
            f"Nessun writer possiede almeno {samples_per_writer} immagini."
        )

    if actual_num_writers < num_writers:
        print(
            f"Richiesti {num_writers} writer, ma solo "
            f"{actual_num_writers} soddisfano il requisito. "
            f"Verranno utilizzati tutti quelli disponibili."
        )

    rng = np.random.default_rng(sample_seed)

    selected_writers = rng.choice(
        eligible_writers,
        size=actual_num_writers,
        replace=False
    )

    #print(f"Writer selezionati: {selected_writers}")

    selected_indices = []

    for writer_id in selected_writers:

        writer_indices = []

        for index in val_indices:
            sample = base_dataset[index]

            if sample["writer_id"] == writer_id:
                writer_indices.append(index)

        chosen_indices = rng.choice(
            writer_indices,
            size=samples_per_writer,
            replace=False
        )

        selected_indices.extend(chosen_indices)
        
    val_subset = Subset(base_dataset, selected_indices)
    
    # Scegliamo un sottoinsieme se ci sono troppe immagini, per velocizzare t-SNE
    #if len(val_subset) > max_samples:
    #    # Prendiamo un campione fisso o casuale
    #    rng = np.random.default_rng(sample_seed)
    #    indices = rng.choice(len(val_subset), max_samples, replace=False)
    #    val_subset = Subset(base_dataset, [val_indices[i] for i in indices])

    eval_loader = DataLoader(
        val_subset,
        batch_size=32,
        shuffle=False,
        collate_fn=lambda b: {
            "image": torch.stack([to_tensor(x["image"]) for x in b]),
            "writer_id": [x["writer_id"] for x in b]
        }
    )

    embeddings_list = []
    labels_list = []
    
    with torch.no_grad():
        for batch in eval_loader:
            imgs = batch["image"].to(device)
            writers = batch["writer_id"]
            
            # Estrazione embedding (usando forward_once o l'encoder della Siamese)
            #if hasattr(model, "forward_once"):
            #    embs = model.forward_once(imgs)
            #else:
                # Se non ha forward_once, passiamo l'immagine due volte o adattiamo
            #    embs, _ = model(imgs, imgs)

            embs = model.forward_one(imgs)

            embeddings_list.append(embs.cpu().numpy())
            labels_list.extend(writers)
            
    X = np.vstack(embeddings_list)
    y = np.array(labels_list)
    
    # Riduzione dimensionale con t-SNE per la visualizzazione 2D
    # t-SNE e salvataggio grafico eseguiti SOLO se save_plot è True
    if save_plot:
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        X_2d = tsne.fit_transform(X)
        
        unique_writers = list(set(y))
        writer_to_int = {w: i for i, w in enumerate(unique_writers)}
        y_numeric = np.array([writer_to_int[w] for w in y])

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_numeric, cmap='tab10', alpha=0.6, s=15)
        plt.title(f"Spazio degli Embedding - Epoca {epoch}")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.colorbar(scatter, label="Writer Clusters")
        plt.grid(True, linestyle="--", alpha=0.5)
        
        plot_path = Path(output_dir) / f"embedding_space_epoch_{epoch}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

def calculate_disaggregated_metrics(
    distances: list[float],
    labels: list[float],
    pair_types: list[int],
    threshold: float,
    ) -> dict[str, dict[str, float]]:
    """
        Calcola le metriche di accuratezza separate per ciascuna delle 4 categorie di coppia:
        - same_writer_different_text
        - same_writer_same_text
        - different_writer_different_text
        - different_writer_same_text
    """
    categories = {
        0: "same_writer_different_text",
        1: "same_writer_same_text",
        2: "different_writer_different_text",
        3: "different_writer_same_text",
    }
    
    results = {}
    
    for type_id, category in categories.items():
        # estrazione distanze e label per categoria corrente
        cat_distances = [d for d, t in zip(distances, pair_types) if t == type_id]
        cat_labels = [l for l, t in zip(labels, pair_types) if t == type_id]

        # se non ci sono campioni per la categoria, salta il calcolo delle metriche
        if not cat_labels:
            continue

        # conta quanti campioni correttamente classificati in questa categoria
        correct = sum(
            1 for dist, label in zip(cat_distances, cat_labels)
            if (1 if dist < threshold else 0) == label
        )
        accuracy = correct / len(cat_labels)                                    # accuracy
        mean_dist = float(np.mean(cat_distances)) if cat_distances else 0.0     # distanza media
        
        results[category] = {
            "accuracy": accuracy,
            "mean_distance": mean_dist,
            "total_samples": len(cat_labels),
        }
        
    return results

def calculate_eer_and_plot_roc(
        distances: list[float],
        labels: list[float],
        output_path: Path | str,
    ) -> float:
    """Calcola l'Equal Error Rate (EER) e salva il grafico della Curva ROC."""

    # Convertiamo le distanze in score di similarità (più è piccola la distanza, più sono simili)
    # perché la funzione roc di scikit assume valori più alti come più probabili positivi (noi contrario)
    scores = [-d for d in distances]
    
    fpr, tpr, thresholds = roc_curve(labels, scores)    # False Pos. Rate e True Pos. Rate
    fnr = 1 - tpr                                       # False Neg. Rate
    
    # punto in cui FPR "=" FNR
    eer_threshold_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = float(fpr[eer_threshold_idx])
    
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (EER = {eer:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FAR)")
    plt.ylabel("True Positive Rate (1 - FRR)")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plt.savefig(Path(output_path) / "roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    return eer