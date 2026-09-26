import random

def split_by_writer(dataset, train_ratio=0.70, val_ratio=0.15, seed=42):

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1")

    if dataset.granularity == "words":
        samples = dataset.words
    elif dataset.granularity == "lines":
        samples = dataset.lines
    elif dataset.granularity == "both":
        samples = dataset.lines + dataset.words
    else:
        raise ValueError(f"Invalid granularity: {dataset.granularity}")
        
    writer_ids = set()
    for sample in samples:
        writer_ids.add(sample["writer_id"])

    writer_ids = list(writer_ids)
    writer_ids = sorted(writer_ids)

    random.seed(seed)
    random.shuffle(writer_ids)

    num_writers = len(writer_ids)

    num_train = int(num_writers * train_ratio)
    num_val = int(num_writers * val_ratio)
    train_writers = writer_ids[:num_train]
    val_writers = writer_ids[num_train:num_train + num_val]
    test_writers = writer_ids[num_train + num_val:]

    #lista di autori per ogni split
    #train_writers  = quali AUTORI appartengono al training
    train_writers = set(train_writers)
    val_writers = set(val_writers)
    test_writers = set(test_writers)

    #lista di indici per ogni split
    #train_indices  = quali CAMPIONI/IMMAGINI appartengono al training
    train_indices = []
    val_indices = []
    test_indices = []

    for index, sample in enumerate(samples):
        writer_id = sample["writer_id"]

        if writer_id in train_writers:
            train_indices.append(index)
        elif writer_id in val_writers:
            val_indices.append(index)
        elif writer_id in test_writers:
            test_indices.append(index)

    return train_indices, val_indices, test_indices