"""ImageNet-normalized 224×224 CXR transforms."""

from __future__ import annotations

from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def eval_transform(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Lambda(lambda im: im.convert("RGB")),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
