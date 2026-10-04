import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset
from pathlib import Path



def collect_distances(model, data_loader, device):
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

def find_best_threshold(distances, labels):
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

def calculate_metrics(distances, labels, threshold):
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



def evaluate_and_plot_embeddings(base_dataset, val_indices, model, device, epoch, output_dir,    num_writers=10, samples_per_writer=30, sample_seed=42, save_plot=False):
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

    print(f"Numero di writer: {len(writer_counts)}")
    print(f"Minimo immagini per writer: {min(writer_counts.values())}")
    print(f"Massimo immagini per writer: {max(writer_counts.values())}")

    eligible_writers = [
    writer_id
    for writer_id, count in writer_counts.items()
    if count >= samples_per_writer
]

    print(
        f"Writer con almeno {samples_per_writer} immagini: "
        f"{len(eligible_writers)}"
    )

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

    print(f"Writer selezionati: {selected_writers}")

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
            "image": torch.stack([x["image"] for x in b]),
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