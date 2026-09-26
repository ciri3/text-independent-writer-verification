import torch
import torch.nn.functional as F

def pad_collate(batch):
    images = [sample["image"] for sample in batch]

    max_width = max(image.shape[2] for image in images)

    padded_images = []
    for image in images:
        padding_width = max_width - image.shape[2]

        padded_image = F.pad(image, (0, padding_width, 0, 0))

        padded_images.append(padded_image)

    images = torch.stack(padded_images)