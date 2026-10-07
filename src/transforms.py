from PIL import Image, ImageOps
from typing import Any


class ResizeAndPad:
    """
        Ridimensiona l'immagine mantenendo l'aspect ratio e la inserisce al centro
        di un canvas di altezza e larghezza fisse con padding bianco (255).
    """

    def __init__(self, height: int = 64, max_width: int = 320) -> None:
        self.height = height
        self.max_width = max_width

    def __call__(self, image: Image.Image) -> Image.Image:
        old_width, old_height = image.size

        scale = min(
            self.height / old_height,
            self.max_width / old_width
        )

        new_height = round(old_height * scale)
        new_width = round(old_width * scale) 

        resized_image = image.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS
        )

        canvas = Image.new(
            image.mode,
            (self.max_width, self.height),
            color=255
        )

        left = (self.max_width - new_width) // 2
        top = (self.height - new_height) // 2

        canvas.paste(resized_image, (left, top))
        return canvas

class NormalizeContrast:
    """Aplica l'autocontrasto all'immagine tagliando una percentuale fissa dei pixel estremi."""
    
    #il cutoff è un parametro che determina la percentuale di pixel più chiari e più scuri da tagliare prima di calcolare il contrasto. Un valore più alto di cutoff rimuove più pixel estremi, aumentando il contrasto dell'immagine risultante.
    def __init__(self, cutoff: int = 1) -> None:

        self.cutoff = cutoff

    def __call__(self, image: Image.Image) -> Image.Image:
        return ImageOps.autocontrast(
            image,
            cutoff=self.cutoff
        )