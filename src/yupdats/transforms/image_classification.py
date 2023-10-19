import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as tf

from .normalizations import *

__all__ = [
    "image_classification_transforms",
]


def numpy_image_to_tensor(
        numpy_image: np.ndarray,
) -> torch.Tensor:
    assert len(numpy_image.shape) == 3, \
        f"Input must be 3D (H*W*C), got {len(numpy_image.shape)}D"

    return torch.from_numpy(numpy_image).permute(2, 0, 1)


default_transforms = {
    "CIFAR10": nn.Identity(),

    "CIFAR100": nn.Identity(),

    "ImageNet": tf.Compose([
        tf.CenterCrop(256),
        tf.Resize(224),
    ]),
}

image_classification_transforms = {
    # CIFAR10
    "CIFAR10": tf.Compose([
        default_transforms["CIFAR10"],
        tf.ToTensor(),
    ]),
    "CIFAR10_uint8": tf.Compose([
        default_transforms["CIFAR10"],
        numpy_image_to_tensor,
    ]),
    "CIFAR10+norm": tf.Compose([
        default_transforms["CIFAR10"],
        tf.ToTensor(),
        normalize["CIFAR10"],
    ]),

    # CIFAR100
    "CIFAR100": tf.Compose([
        default_transforms["CIFAR100"],
        tf.ToTensor(),
    ]),
    "CIFAR100_uint8": tf.Compose([
        default_transforms["CIFAR100"],
        numpy_image_to_tensor,
    ]),
    "CIFAR100+norm": tf.Compose([
        default_transforms["CIFAR100"],
        tf.ToTensor(),
        normalize["CIFAR100"],
    ]),

    # ImageNet
    "ImageNet": tf.Compose([
        default_transforms["ImageNet"],
        tf.ToTensor(),
    ]),
    "ImageNet_uint8": tf.Compose([
        default_transforms["ImageNet"],
        tf.PILToTensor(),
    ]),
    "ImageNet+norm": tf.Compose([
        default_transforms["ImageNet"],
        tf.ToTensor(),
        normalize["ImageNet"],
    ]),
}
