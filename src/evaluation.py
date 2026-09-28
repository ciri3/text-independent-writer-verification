import torch
import torch.nn.functional as F


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