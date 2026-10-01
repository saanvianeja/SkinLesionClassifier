"""Image transforms. Eval pipeline matches the original CNN inference path."""

from torchvision import transforms

from skinlesion.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def eval_transforms() -> transforms.Compose:
    """Resize 224, tensor, ImageNet normalize — same as original val/inference."""
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def train_transforms() -> transforms.Compose:
    """Training augmentations from the original train_real_isic.py."""
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
