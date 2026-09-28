import random
from torch.utils.data import Dataset

class SiameseDataset(Dataset):
    def __init__(self, base_dataset, indices, number_of_pairs=100000):

        self.base_dataset = base_dataset
        self.indices = indices
        self.number_of_pairs = number_of_pairs

        # Selezioniamo i metadati in base alla granularità del dataset
        if self.base_dataset.granularity == "words":
            samples = self.base_dataset.words
        elif self.base_dataset.granularity == "lines":
            samples = self.base_dataset.lines
        elif self.base_dataset.granularity == "both":
            samples = self.base_dataset.lines + self.base_dataset.words
        else:
            raise ValueError(f"Invalid granularity: {self.base_dataset.granularity}")
        
        #self.writer_to_indices = {}
        #self.text_to_indices = {}
        self.writer_to_text_indices = {}
        
        for index in self.indices:

            sample = samples[index]
            writer_id = sample["writer_id"]
            text = sample["text"].lower()

            if writer_id not in self.writer_to_text_indices:
                self.writer_to_text_indices[writer_id] = {}
            if text not in self.writer_to_text_indices[writer_id]:
                self.writer_to_text_indices[writer_id][text] = []
            self.writer_to_text_indices[writer_id][text].append(index)

        # Lista di tutti i writer disponibili in questo split
        self.writer_ids = list(self.writer_to_text_indices.keys())

    def _same_writer_different_text(self):
        while True:
            writer_id = random.choice(self.writer_ids)
            texts = list(self.writer_to_text_indices[writer_id].keys())
            if len(texts) >= 2:
                break

        text1, text2 = random.sample(texts, 2)
        index1 = random.choice(self.writer_to_text_indices[writer_id][text1])
        index2 = random.choice(self.writer_to_text_indices[writer_id][text2])

        return index1, index2

    def _same_writer_same_text(self):

        while True:
            writer_id = random.choice(self.writer_ids)

            valid_texts = []

            for text, indices in self.writer_to_text_indices[writer_id].items():
                if len(indices) >= 2:
                    valid_texts.append(text)

            if len(valid_texts) > 0:
                break

        text = random.choice(valid_texts)

        index1, index2 = random.sample(self.writer_to_text_indices[writer_id][text], 2)

        return index1, index2


    def _different_writer_different_text(self):
        while True:
            writer1, writer2 = random.sample(self.writer_ids, 2)
            text1 = random.choice(
                list(self.writer_to_text_indices[writer1].keys())
            )
            valid_texts2 = [
                text
                for text in self.writer_to_text_indices[writer2].keys()
                if text != text1
            ]
            if len(valid_texts2) > 0:
                break

        text2 = random.choice(valid_texts2)

        index1 = random.choice(self.writer_to_text_indices[writer1][text1])
        index2 = random.choice(self.writer_to_text_indices[writer2][text2])

        return index1, index2

    def _different_writer_same_text(self):
        while True:
            writer1, writer2 = random.sample(self.writer_ids, 2)

            texts1 = set(self.writer_to_text_indices[writer1].keys())
            texts2 = set(self.writer_to_text_indices[writer2].keys())

            common_texts = list(texts1.intersection(texts2))

            if len(common_texts) > 0:
                break

        text = random.choice(common_texts)

        index1 = random.choice(self.writer_to_text_indices[writer1][text])
        index2 = random.choice(self.writer_to_text_indices[writer2][text])

        return index1, index2

    def __len__(self):
        return self.number_of_pairs

    def __getitem__(self, idx):

        pair_type = idx % 4

        if pair_type == 0:
            index1, index2 = self._same_writer_different_text()
            label = 1
        elif pair_type == 1:
            index1, index2 = self._same_writer_same_text()
            label = 1
        elif pair_type == 2:
            index1, index2 = self._different_writer_different_text()
            label = 0
        else:
            index1, index2 = self._different_writer_same_text()
            label = 0

        sample1 = self.base_dataset[index1]
        sample2 = self.base_dataset[index2]

        return {
            "image1": sample1["image"],
            "image2": sample2["image"],
            "label": label
        }