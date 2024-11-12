from typing import Optional

import torch


class ImageAverage(torch.nn.Module):
    def forward(self, x: torch.Tensor, image_id: torch.Tensor) -> torch.Tensor:
        """
        Average x features by image_id.

        Args:
            x (torch.Tensor): A float tensor of shape (n_refln, n_feature).
            image_id (torch.Tensor): An int tensor of shape (n_refln).

        Returns:
            xout (torch.Tensor): A float tensor of shape (n_images, n_feature).
        """
        n_images = image_id.max() + 1
        n_features = x.shape[-1]
        idx = torch.tile(image_id[:, None], (1, n_features))
        xout = torch.zeros((n_images, n_features))
        xout.scatter_add_(dim=0, index=idx, src=x)
        n_reflns_per_image = torch.bincount(image_id)
        xout /= n_reflns_per_image[:, None]
        return xout
