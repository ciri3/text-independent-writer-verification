from PIL import Image, ImageOps


class ResizeAndPad:
    def __init__(self, height = 64, max_width = 320):
        self.height = height
        self.max_width = max_width

    def __call__(self, image):
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

    #il cutoff è un parametro che determina la percentuale di pixel più chiari e più scuri da tagliare prima di calcolare il contrasto. Un valore più alto di cutoff rimuove più pixel estremi, aumentando il contrasto dell'immagine risultante.
    def __init__(self, cutoff=1):

        self.cutoff = cutoff

    def __call__(self, image):
        return ImageOps.autocontrast(
            image,
            cutoff=self.cutoff
        )