from pathlib import Path

from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor

from src.models.siamese import SiameseNetwork
from src.transforms import ResizeAndPad


class WriterVerifier:

    def __init__(
        self,
        model_path="models/best_model.pth",
        threshold=0.5520,
    ):
        self.threshold = threshold

        # Seleziona automaticamente il miglior device disponibile
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        # Stesso preprocessing utilizzato durante training/test
        self.transform = Compose([
            ResizeAndPad(height=64, max_width=320),
            ToTensor()
        ])

        # Stessa architettura utilizzata durante il training
        self.model = SiameseNetwork(
            embedding_dim=128
        ).to(self.device)

        # Caricamento checkpoint
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Modello non trovato: {model_path}"
            )

        checkpoint = torch.load(
            model_path,
            map_location=self.device
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        # Disattiva il comportamento specifico del training
        self.model.eval()

    def _load_image(self, image_path):

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Immagine non trovata: {image_path}"
            )

        # Identico a IAMDataset
        image = Image.open(image_path).convert("L")

        # Resize + padding + conversione in Tensor
        image = self.transform(image)

        # [C, H, W] -> [1, C, H, W]
        image = image.unsqueeze(0)

        return image.to(self.device)

    def compare(self, image1_path, image2_path):

        image1 = self._load_image(image1_path)
        image2 = self._load_image(image2_path)

        # Inferenza: non servono gradienti
        with torch.no_grad():

            embedding1, embedding2 = self.model(
                image1,
                image2
            )

            distance = F.pairwise_distance(
                embedding1,
                embedding2
            ).item()

        same_writer = distance < self.threshold

        return {
            "distance": distance,
            "threshold": self.threshold,
            "same_writer": same_writer,
            "prediction": (
                "stesso autore"
                if same_writer
                else "autori diversi"
            )
        }