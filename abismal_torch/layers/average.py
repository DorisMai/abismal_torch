import torch


class ImageAverage(torch.nn.Module):
    def forward(
        self, x: torch.Tensor, image_indices: torch.Tensor, counts: torch.Tensor
    ) -> torch.Tensor:
        """
        Average x along images and across mc samples.

        Args:
            x (torch.Tensor): Source value tensor of shape (n_reflns, n_dim).
            image_indices (torch.Tensor): A tensor of shape (n_reflns,) that contains the
                image index (range 0 to n_images-1) for each reflection in the batch.
            counts (torch.Tensor): A tensor of shape (n_images,) that contains the
                number of reflections in each image.
        Returns:
            averaged (torch.Tensor): A tensor of shape (n_images, n_dim).
        """
        n_images = counts.size(0)
        _, n_dim = x.shape
        _averaged = x.new_zeros((n_images, n_dim))
        idx = image_indices[:, None].expand(-1, n_dim)
        _averaged.scatter_add_(dim=0, index=idx, src=x)
        _averaged /= counts[:, None]
        return _averaged
