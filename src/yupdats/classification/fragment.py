from typing import Any, List, Optional, Tuple, Union

import numpy as np
import random
import torch
import tqdm

from .base import ImageClassificationDataset

__all__ = [
    "_fragment",
]


def data_and_targets_of_class_c(
        data: Union[List[Tuple[str, int]], np.ndarray, torch.Tensor],
        targets: torch.Tensor,
        c: int,
) -> Tuple[Any, Any]:
    assert data is not None and targets is not None

    indices = torch.arange(len(data))[targets == c]

    data_c = []
    targets_c = []

    for i in indices:
        if isinstance(data, list):
            data_c.append(data[i])

        elif isinstance(data, np.ndarray):
            data_c.append(np.expand_dims(data[i], axis=0))

        elif isinstance(data, torch.Tensor):
            data_c.append(data[i].unsqueeze(dim=0))

        else:
            raise

        target = targets[i]

        if isinstance(target, int):
            target = torch.Tensor([target, ]).to(torch.int64)[0]

        targets_c.append(target.unsqueeze(dim=0))

    if isinstance(data, np.ndarray):
        data_c = np.concatenate(data_c, axis=0)

    elif isinstance(data, torch.Tensor):
        data_c = torch.cat(data_c, dim=0)

    targets_c = torch.cat(targets_c, dim=0)

    return data_c, targets_c


def frag(
        data: Union[List[Tuple[str, int]], np.ndarray, torch.Tensor],
        targets: Union[List[int], np.ndarray, torch.Tensor],
        pct: float,
        random_pick: bool = False,
) -> Tuple[
    Union[List[Tuple[str, int]], np.ndarray, torch.Tensor],
    Union[List[int], np.ndarray, torch.Tensor],
]:
    num_data = len(data)
    num_frag = int(len(data) * pct)

    if random_pick:
        indices = random.sample(range(num_data), num_frag)

        if isinstance(data, list):
            return [data[i] for i in indices], [targets[i] for i in indices]

        elif isinstance(data, np.ndarray):
            return data[indices], targets[indices]

        elif isinstance(data, torch.Tensor):
            assert isinstance(targets, torch.Tensor)

            return data[indices], targets[indices]

        else:
            raise

    else:
        return data[:num_frag], targets[:num_frag]


def cat(
        original: Union[None, List[Tuple[str, int]], np.ndarray, torch.Tensor],
        new: Union[List[Tuple[str, int]], np.ndarray, torch.Tensor],
) -> Union[List[Tuple[str, int]], np.ndarray, torch.Tensor]:
    if original is None:
        return new

    else:
        assert type(original) == type(new)

        if isinstance(original, list):
            return original + new

        elif isinstance(original, np.ndarray):
            return np.concatenate([original, new], axis=0)

        elif isinstance(original, torch.Tensor):
            return torch.cat([original, new], dim=0)

        else:
            raise


def _fragment(
        dataset: ImageClassificationDataset,
        pct: float,
        random_pick: bool = False,
        keep_class_balance: bool = False,
        name: Optional[str] = None,
        verbose: Optional[bool] = False,
) -> ImageClassificationDataset:
    assert 0 < pct < 1

    frag_dataset = ImageClassificationDataset(
        name=name if name is not None else dataset.name,
        transform=dataset.transform,
        target_transform=dataset.target_transform,
    )

    if keep_class_balance:
        frag_data = None
        frag_targets = None

        num_class = len(dataset.class_labels)

        for c in tqdm.trange(
                num_class,
                desc=f"{pct * 100:.2f}% fragment of {dataset.name}",
                disable=not verbose,
        ):
            data_c, targets_c = data_and_targets_of_class_c(dataset.data, dataset.targets, c)

            frag_data_c, frag_targets_c = frag(data_c, targets_c, pct, random_pick)

            frag_data = cat(frag_data, frag_data_c)
            frag_targets = cat(frag_targets, frag_targets_c)

    else:
        data = dataset.data
        targets = dataset.targets

        frag_data, frag_targets = frag(data, targets, pct, random_pick)

    frag_dataset.initialize(frag_data, frag_targets, dataset.class_labels)

    return frag_dataset
