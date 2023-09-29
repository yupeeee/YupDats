from typing import Tuple

import torch

__all__ = [
    "PCA",
]


class PCA:
    def __init__(
            self,
            use_cuda: bool = False,
    ) -> None:
        self.use_cuda = use_cuda

        self.components = None
        self.singular_values = None

    def __call__(self, data: torch.Tensor, ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.run(data)

        return self.components, self.singular_values

    def run(self, data: torch.Tensor, ) -> None:
        n, d = data.shape
        data = data.to(torch.device("cuda" if self.use_cuda else "cpu"))

        # mean centering
        centered_data = data - data.mean(dim=0, keepdim=True)

        # covariance matrix
        cov_mat = centered_data.T @ centered_data / n

        # EVD (using SVD)
        components, singular_values, _ = torch.linalg.svd(cov_mat)

        self.components = components.detach().cpu()
        self.singular_values = singular_values.detach().cpu()

    def proj_to_components(self, data: torch.Tensor, ) -> torch.Tensor:
        assert self.components is not None, \
            "To project data to components, run PCA first"

        return data @ self.components
