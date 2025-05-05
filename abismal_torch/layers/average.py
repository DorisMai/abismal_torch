import torch


class ImageAverage(torch.nn.Module):
    def forward(
        self, x: torch.Tensor, image_id: torch.Tensor) -> torch.Tensor:
        """
        Average x along images and across mc samples.

        Args:
            x (torch.Tensor): Source value tensor of shape (n_reflns, n_dim).
            image_id (torch.Tensor): A tensor of shape (n_reflns,) that contains the
                image index for each reflection.

        Returns:
            averaged (torch.Tensor): A tensor of shape (n_images, n_dim).
            counts (torch.Tensor): A tensor of shape (n_images,) that contains the
                number of reflections in each image.
        """
        unique_image_ids, unique_indices, counts = torch.unique(image_id, return_inverse=True, return_counts=True)
        n_images = unique_image_ids.size(0)
        _, n_dim = x.shape
        _averaged = x.new_zeros((n_images, n_dim))
        idx = unique_indices[:, None].expand(-1, n_dim)
        _averaged.scatter_add_(dim=0, index=idx, src=x)
        _averaged /= counts[:, None]
        return _averaged, unique_indices, counts