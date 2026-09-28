import torch
import torch.nn.functional as F

def pad_collate(batch):
    images1 = [sample["image1"] for sample in batch]
    images2 = [sample["image2"] for sample in batch]

    max_width = max(image.shape[2] for image in images1 + images2)

    padded_images1 = []
    padded_images2 = []

    for image in images1:
        padding_width = max_width - image.shape[2]

        padded_image = F.pad(image, (0, padding_width, 0, 0))

        padded_images1.append(padded_image)

    images1 = torch.stack(padded_images1)

    for image in images2:
        padding_width = max_width - image.shape[2]

        padded_image = F.pad(image, (0, padding_width, 0, 0))

        padded_images2.append(padded_image)

    images2 = torch.stack(padded_images2)

    labels = torch.tensor([sample["label"] for sample in batch], dtype=torch.float32)

    return {
    "image1": images1,
    "image2": images2,
    "label": labels
}