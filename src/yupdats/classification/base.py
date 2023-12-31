from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
from torchvision.datasets import DatasetFolder

__all__ = [
    "ImageClassificationDataset",
]


class ImageClassificationDataset:
    default_channel = "RGB"

    def __init__(
            self,
            name: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        self.name = name
        self.transform = transform
        self.target_transform = target_transform

        # must be initialized
        self.data = None
        self.targets = None
        self.class_labels = None

    def __getitem__(
            self,
            index: int,
    ) -> Tuple[Any, Any]:
        assert self.data is not None and self.targets is not None

        # if type(data) == List[Tuple[path_to_data: str, target: int]]
        if isinstance(self.data, list):
            path, target = self.data[index]
            data = Image.open(path).convert(self.default_channel)

        else:
            data = self.data[index]
            target = self.targets[index]

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

    def __len__(
            self,
    ) -> int:
        return len(self.data)

    def __str__(
            self,
    ) -> str:
        return self.name

    @staticmethod
    def make_dataset(
            directory: str,
            extensions: List[str],
    ) -> List[Tuple[str, int]]:
        datasetfolder = DatasetFolder(
            root=directory,
            loader=None,
            extensions=extensions,
            transform=None,
            target_transform=None,
            is_valid_file=None,
        )

        _, class_to_idx = datasetfolder.find_classes(directory)

        path_and_target = datasetfolder.make_dataset(
            directory=directory,
            class_to_idx=class_to_idx,
            extensions=extensions,
            is_valid_file=None,
        )

        return path_and_target

    # MUST BE DONE
    def initialize(
            self,
            data: Union[List[Tuple[str, int]], np.ndarray, torch.Tensor],
            targets: Union[List[int], np.ndarray, torch.Tensor],
            class_labels: Union[Dict[int, str], List[str]],
    ) -> None:
        # initialize
        self.data = data
        self.targets = targets
        self.class_labels = class_labels

        # if type(targets) == list,
        # type(targets) <- torch.Tensor
        if isinstance(self.targets, list):
            self.targets = torch.Tensor(self.targets)

        self.targets = self.targets.to(torch.int64)

    def data_and_targets_of_class_c(
            self,
            c: int,
    ) -> Tuple[Any, Any]:
        assert self.data is not None and self.targets is not None

        indices = torch.arange(len(self))[self.targets == c]

        data_c = []
        targets_c = []

        for i in indices:
            data, target = self[i]

            if isinstance(data, np.ndarray):
                data_c.append(np.expand_dims(data, axis=0))

            elif isinstance(data, torch.Tensor):
                data_c.append(data.unsqueeze(dim=0))

            else:
                raise TypeError(
                    f"data type must be ndarray or Tensor, got {type(data)}"
                )

            if isinstance(target, int):
                target = torch.Tensor([target, ]).to(torch.int64)[0]

            targets_c.append(target.unsqueeze(dim=0))

        if isinstance(data, np.ndarray):
            data_c = np.concatenate(data_c, axis=0)

        else:
            data_c = torch.cat(data_c, dim=0)

        targets_c = torch.cat(targets_c, dim=0)

        return data_c, targets_c

    def mean_and_std_of_data(
            self,
    ) -> Tuple[Any, Any]:
        assert self.data is not None
        assert not isinstance(self.data, list)

        data = self.data

        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)

        data = data.type(torch.float32)

        mean = data.mean(axis=(0, 1, 2))
        std = data.std(axis=(0, 1, 2))

        return mean, std

    def fragment(
            self,
            pct: float,
            random_pick: bool = False,
            keep_class_balance: bool = False,
            name: Optional[str] = None,
            verbose: Optional[bool] = False,
    ):
        from .fragment import _fragment

        return _fragment(self, pct, random_pick, keep_class_balance, name, verbose)
