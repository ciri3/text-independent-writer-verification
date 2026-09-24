from pathlib import Path
from PIL import Image
import xml.etree.ElementTree as ET

from torch.utils.data import Dataset


class IAMDataset(Dataset):
    

    def __init__(self, data_dir="data", granularity="lines"):
        self.granularity = granularity
        self.data_dir = Path(data_dir)

        self.lines_dir = self.data_dir / "lines"
        self.words_dir = self.data_dir / "words"
        self.xml_dir = self.data_dir / "xml"

        self.lines = []
        self.words = []
        self._load_metadata()

    def __len__(self):
        if self.granularity == "lines":
            return len(self.lines)
        elif self.granularity == "words":
            return len(self.words)
        elif self.granularity == "both":
            return len(self.lines) + len(self.words)
        else:
            raise ValueError(f"Invalid granularity: {self.granularity}")

    def __getitem__(self, idx):
        if self.granularity == "lines":
            metadata = self.lines[idx]
        elif self.granularity == "words":
            metadata = self.words[idx]
        elif self.granularity == "both":
            if idx < len(self.lines):
                metadata = self.lines[idx]
            else:
                metadata = self.words[idx-len(self.lines)]
        else:
            raise ValueError(f"Invalid granularity: {self.granularity}")
        
        image = Image.open(metadata["image_path"])

        #ancora restituisce un'immagine ma deve restituire un tensore
        return {
            "image": image,
            "id": metadata["id"],
            "writer_id": metadata["writer_id"],
            "text": metadata["text"],
        }

        
    def _load_metadata(self):
        for xml_path in self.xml_dir.glob("*.xml"):

            tree = ET.parse(xml_path)
            root = tree.getroot()
            writer_id = root.get("writer-id")

            handwritten_part = root.find("handwritten-part")

            for line in handwritten_part.findall("line"):

                line_id = line.get("id")
                line_text = line.get("text")

                parts = line_id.split("-")
                line_image_path = self.lines_dir / parts[0] / "-".join(parts[:2]) / f"{line_id}.png"

                line_sample = {
                    "id": line_id,
                    "writer_id": writer_id,
                    "text": line_text,
                    "image_path": line_image_path,
                }

                self.lines.append(line_sample)
                for word in line.findall("word"):

                    word_id = word.get("id")
                    word_text = word.get("text")

                    parts = word_id.split("-")
                    word_image_path = self.words_dir / parts[0] / "-".join(parts[:2]) / f"{word_id}.png"   

                    word_sample = {
                        "id": word_id,
                        "writer_id": writer_id,
                        "text": word_text,
                        "image_path": word_image_path,
                  }

                    self.words.append(word_sample)
