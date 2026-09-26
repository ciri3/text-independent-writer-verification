from PIL import Image

class ResizeHeight:
    def __init__(self, height):
        self.height = height

    def __call__(self, image):
        width, height = image.size
        old_width = width
        old_height = height
        new_height = self.height
        new_width = round(old_width * (new_height / old_height)) 
        return  image.resize((new_width, new_height), Image.Resampling.LANCZOS)
