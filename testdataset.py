from src.dataset import IAMDataset
from src.transforms import ResizeHeight
from src.split import split_by_writer
from src.siamese_dataset import SiameseDataset
from collections import Counter
from torchvision.transforms import Compose, ToTensor


# Definiamo le trasformazioni da applicare alle immagini
transform = Compose([
    ResizeHeight(64),
    ToTensor()
])


# Creiamo il dataset
dataset = IAMDataset(
    data_dir="data",
    granularity="words",
    transform=transform
)


# Controlliamo il numero di campioni
print("Numero campioni:", len(dataset))


# Controlliamo i primi 10 campioni
for i in range(10):
    sample = dataset[i]

    print(
        i,
        sample["text"],
        sample["image"].shape,
        sample["writer_id"]
    )

train_indices, val_indices, test_indices = split_by_writer(dataset)

print("Totale:", len(dataset))
print("Train:", len(train_indices))
print("Validation:", len(val_indices))
print("Test:", len(test_indices))

print(
    "Somma corretta:",
    len(train_indices) + len(val_indices) + len(test_indices) == len(dataset)
)

train_writers = {dataset.words[i]["writer_id"] for i in train_indices}
val_writers = {dataset.words[i]["writer_id"] for i in val_indices}
test_writers = {dataset.words[i]["writer_id"] for i in test_indices}

print("Writer train:", len(train_writers))
print("Writer validation:", len(val_writers))
print("Writer test:", len(test_writers))

print("Train-Val separati:", train_writers.isdisjoint(val_writers))
print("Train-Test separati:", train_writers.isdisjoint(test_writers))
print("Val-Test separati:", val_writers.isdisjoint(test_writers))

train_siamese = SiameseDataset(dataset, train_indices, number_of_pairs=100000)

print("Writer nel SiameseDataset:", len(train_siamese.writer_ids))
print("Numero coppie per epoca:", len(train_siamese))


# Testiamo __getitem__
for i in range(8):
    pair = train_siamese[i]

    print(
        i,
        "label:", pair["label"],
        "shape1:", pair["image1"].shape,
        "shape2:", pair["image2"].shape
    )


# Testiamo singolarmente i quattro tipi di coppia
methods = [
    train_siamese._same_writer_different_text,
    train_siamese._same_writer_same_text,
    train_siamese._different_writer_different_text,
    train_siamese._different_writer_same_text
]

for method in methods:
    index1, index2 = method()

    sample1 = dataset.words[index1]
    sample2 = dataset.words[index2]

    print("\n", method.__name__)
    print("Writer:", sample1["writer_id"], "|", sample2["writer_id"])
    print("Testo:", sample1["text"], "|", sample2["text"])


    

writer_counts = Counter()

for _ in range(10000):
    index1, index2 = train_siamese._same_writer_different_text()

    writer_id = dataset.words[index1]["writer_id"]
    writer_counts[writer_id] += 1

counts = list(writer_counts.values())

print("\nTest bilanciamento writer:")
print("Writer utilizzati:", len(writer_counts))
print("Min:", min(counts))
print("Max:", max(counts))
print("Media:", sum(counts) / len(counts))