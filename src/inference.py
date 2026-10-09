from pathlib import Path
from typing import Any

from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor

from src.models.siamese import SiameseNetwork
from src.transforms import ResizeAndPad, NormalizeContrast



class WriterVerifier:
    """Classe per eseguire inferenza in tempo reale su una coppia di immagini."""

    def __init__(
        self,
        model_path: str | Path = "models/best_model.pth"
    ) -> None:

        # Seleziona automaticamente il miglior device disponibile
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        

        # Caricamento checkpoint
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Modello non trovato: {model_path}")

        checkpoint = torch.load(
            model_path,
            map_location=self.device,
            weights_only=True
        )

        # Stesso preprocessing utilizzato durante training/test
        self.transform = Compose([
            NormalizeContrast(),
            ResizeAndPad(height=checkpoint["image_height"],max_width=checkpoint["image_max_width"]),
            ToTensor()
        ])

         # Recupera la configurazione dal checkpoint
        embedding_dim = checkpoint["embedding_dim"]
        self.threshold = checkpoint["threshold"]

        # Stessa architettura utilizzata durante il training
        self.model = SiameseNetwork(
            embedding_dim=embedding_dim
        ).to(self.device)


        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )
        # Disattiva il comportamento specifico del training
        self.model.eval()


    def _load_image(self, image_path: str | Path) -> torch.Tensor:

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

    def compare(self, image1_path: str | Path, image2_path: str | Path) -> dict[str, Any]:

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