from typing import Callable, Optional

import os

from .base import ImageClassificationDataset

__all__ = [
    "Caltech101",
    "Caltech256",
    "CIFAR10",
    "CIFAR100",
    "Country211",
    "FashionMNIST",
    "ImageNet",
    "MNIST",
]


class Caltech101(ImageClassificationDataset):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        from torchvision.datasets import Caltech101 as Dataset

        super().__init__(
            name="Caltech-101",
            transform=transform,
            target_transform=target_transform,
        )

        dataset = Dataset(
            root=root,
            target_type="category",
            transform=None,
            target_transform=None,
            download=download,
        )

        path_and_target = self.make_dataset(
            directory=os.path.join(
                root,
                "caltech101",
                "101_ObjectCategories",
            ),
            extensions=["jpg", ],
        )

        # remove category "BACKGROUND_Google" (class 0)
        path_and_target = [
            (path_to_data, target) for path_to_data, target in path_and_target
            if "BACKGROUND_Google" not in path_to_data
        ]

        # target must start from 0
        if path_and_target[0][-1] == 1:
            path_and_target = [
                (path_to_data, target - 1) for path_to_data, target in path_and_target
            ]

        self.initialize(
            data=path_and_target,
            targets=[target for _, target in path_and_target],
            class_labels=dataset.categories,
        )


class Caltech256(ImageClassificationDataset):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        from torchvision.datasets import Caltech256 as Dataset

        super().__init__(
            name="Caltech-256",
            transform=transform,
            target_transform=target_transform,
        )

        dataset = Dataset(
            root=root,
            transform=None,
            target_transform=None,
            download=download,
        )

        path_and_target = self.make_dataset(
            directory=os.path.join(
                root,
                "caltech256",
                "256_ObjectCategories",
            ),
            extensions=["jpg", ],
        )

        self.initialize(
            data=path_and_target,
            targets=[target for _, target in path_and_target],
            class_labels=dataset.categories,
        )


class CIFAR10(ImageClassificationDataset):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        from torchvision.datasets import CIFAR10 as Dataset
        from .class_labels.cifar10 import class_labels

        super().__init__(
            name="CIFAR-10",
            transform=transform,
            target_transform=target_transform,
        )

        dataset = Dataset(
            root=root,
            train=train,
            transform=None,
            target_transform=None,
            download=download,
        )

        self.initialize(
            data=dataset.data,
            targets=dataset.targets,
            class_labels=class_labels,
        )


class CIFAR100(ImageClassificationDataset):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        from torchvision.datasets import CIFAR100 as Dataset
        from .class_labels.cifar100 import class_labels

        super().__init__(
            name="CIFAR-100",
            transform=transform,
            target_transform=target_transform,
        )

        dataset = Dataset(
            root=root,
            train=train,
            transform=None,
            target_transform=None,
            download=download,
        )

        self.initialize(
            data=dataset.data,
            targets=dataset.targets,
            class_labels=class_labels,
        )


class Country211(ImageClassificationDataset):
    def __init__(
            self,
            root: str,
            split: str = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        from torchvision.datasets import Country211 as Dataset

        super().__init__(
            name="Country211",
            transform=transform,
            target_transform=target_transform,
        )

        dataset = Dataset(
            root=root,
            split=split,
            transform=None,
            target_transform=None,
            download=download,
        )

        self.initialize(
            data=dataset.imgs,
            targets=dataset.targets,
            class_labels=dataset.classes,
        )


class FashionMNIST(ImageClassificationDataset):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        from torchvision.datasets import FashionMNIST as Dataset
        from .class_labels.fashionmnist import class_labels

        super().__init__(
            name="Fashion-MNIST",
            transform=transform,
            target_transform=target_transform,
        )

        dataset = Dataset(
            root=root,
            train=train,
            transform=None,
            target_transform=None,
            download=download,
        )

        self.initialize(
            data=dataset.data,
            targets=dataset.targets,
            class_labels=class_labels,
        )


class ImageNet(ImageClassificationDataset):
    def __init__(
            self,
            root: str,
            split: str = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        from torchvision.datasets import ImageNet as Dataset
        from .class_labels.imagenet import class_labels

        super().__init__(
            name="ImageNet",
            transform=transform,
            target_transform=target_transform,
        )

        try:
            _ = Dataset(
                root=root,
                split=split,
                transform=None,
                target_transform=None,
                loader=None,
            )
        except RuntimeError:
            pass

        path_and_target = self.make_dataset(
            directory=os.path.join(
                root,
                split,
            ),
            extensions=["jpeg", ],
        )

        self.initialize(
            data=path_and_target,
            targets=[target for _, target in path_and_target],
            class_labels=class_labels,
        )


class MNIST(ImageClassificationDataset):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        from torchvision.datasets import MNIST as Dataset
        from .class_labels.mnist import class_labels

        super().__init__(
            name="MNIST",
            transform=transform,
            target_transform=target_transform,
        )

        dataset = Dataset(
            root=root,
            train=train,
            transform=None,
            target_transform=None,
            download=download,
        )

        self.initialize(
            data=dataset.data,
            targets=dataset.targets,
            class_labels=class_labels,
        )
