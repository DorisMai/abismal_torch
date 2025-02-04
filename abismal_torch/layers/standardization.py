from typing import Optional

import torch
from torch.nn.parameter import UninitializedBuffer


class Standardization(torch.nn.Module):
    def __init__(
        self,
        center: Optional[bool] = True,
        decay: Optional[float] = 0.999,
        epsilon: Optional[float] = 1e-6,
        **kwargs,
    ):
        """
        Naive standardization layer for testing purposes.
        """
        super().__init__(**kwargs)
        self.decay = decay
        self.center = center
        self.epsilon = epsilon

    def _debiased_mean_variance(self, data):
        # TODO: tfs.moving_mean_variance_zero_debiased
        var, mean = torch.var_mean(data, dim=0)
        return mean, var

    @property
    def mean(self):
        mean, _ = self._debiased_mean_variance()
        return mean

    @property
    def var(self):
        _, var = self._debiased_mean_variance()
        return var

    @property
    def std(self):
        s = torch.sqrt(self.var)
        return torch.clamp(s, self.epsilon, torch.inf)

    def forward(self, data):
        mean, var = self._debiased_mean_variance(data)
        std = torch.clamp(torch.sqrt(var), min=self.epsilon)
        if self.center:
            return (data - mean) / std
        return data / std


class WelfordStandardization(torch.nn.modules.batchnorm._NormBase):
    def __init__(
        self,
        num_features,
        epsilon: Optional[float] = 1e-6,
        center: Optional[bool] = True,
        track_running_stats: Optional[bool] = True,
    ):
        """
        Standardization layer using Welford's algorithm to track running mean and variance.
        Extends PyTorch's _NormBase class for better compatibility with the framework.

        Args:
            num_features (int): Number of features in the input data (C from an expected input
                of size (N, C, *))
            epsilon (float, optional): Epsilon for numerical stability. Default: 1e-6
            center (bool, optional): If True, subtract the running mean from the input.
                Default: True
            track_running_stats (bool, optional): Should always be True. Intended to be False
                for LazyWelfordStandardization instances.

        Attributes:
            running_mean (torch.Tensor): Running mean of the input data.
            running_var (torch.Tensor): Running biased variance of the input data.
            num_batches_tracked (torch.Tensor): Note that this will be a misnomer here. We are actually
                tracking the number of samples seen so far.
        """
        super().__init__(
            num_features,
            eps=epsilon,
            momentum=None,
            affine=False,
            track_running_stats=track_running_stats,
        )
        self.center = center

    @property
    def mean(self):
        return self.running_mean

    @property
    def var(self):
        """unbiased variance"""
        return (
            self.running_var
            * self.num_batches_tracked
            / (self.num_batches_tracked - 1).clamp(min=1)
        )

    @property
    def std(self):
        return torch.sqrt(self.var.clamp(min=self.eps))

    def _update_running_stats(self, batch_mean, batch_var, batch_size):
        """
        Update the running mean and variance using Welford's algorithm. Helpful references:
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        https://jonisalonen.com/2013/deriving-welfords-method-for-computing-variance/
        https://www.johndcook.com/blog/standard_deviation/
        https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/running_mean_std.py
        """
        M_a = self.running_var * self.num_batches_tracked
        M_b = batch_var * batch_size
        total_samples = self.num_batches_tracked + batch_size
        delta = batch_mean - self.running_mean
        self.running_mean = self.running_mean + delta * batch_size / total_samples
        self.running_var = (
            M_a + M_b + delta**2 * batch_size * self.num_batches_tracked / total_samples
        ) / total_samples
        self.num_batches_tracked = total_samples

    def standardize(self, x: torch.Tensor) -> torch.Tensor:
        if self.center:
            return (x - self.running_mean) / self.std
        return x / self.std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(x)

        if self.training and self.track_running_stats:
            # Calculate batch statistics
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            batch_size = x.size(0)
            self._update_running_stats(batch_mean, batch_var, batch_size)
        else:
            batch_mean = self.running_mean
            batch_var = self.running_var
        x = self.standardize(x)
        return x

    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError(f"expected 2D input (got {input.dim()}D input)")


class LazyWelfordStandardization(
    torch.nn.modules.lazy.LazyModuleMixin, WelfordStandardization
):
    """
    A WelfordStandardization layer with lazy initialization of the num_features argument
    that is inferred from the ``input.size(1)``. The attributes that will be lazily initialized
    are `running_mean` and `running_var`. See `WelfordStandardization` for more details on
    the arguments and attributes.
    """

    cls_to_become = WelfordStandardization

    def __init__(self, epsilon=1e-5, center=True):
        super().__init__(
            # track_running_stats is hardcoded to False to avoid creating tensors that will
            # soon be overwritten.
            0,
            epsilon,
            center,
            track_running_stats=False,
        )
        self.track_running_stats = True
        self.running_mean = UninitializedBuffer()
        self.running_var = UninitializedBuffer()
        self.num_batches_tracked = torch.tensor(0, dtype=torch.long)

    def reset_parameters(self) -> None:
        if not self.has_uninitialized_params() and self.num_features != 0:
            super().reset_parameters()

    def initialize_parameters(self, input) -> None:  # type: ignore
        if self.has_uninitialized_params():
            self.num_features = input.shape[1]
            if self.track_running_stats:
                self.running_mean.materialize((self.num_features,))
                self.running_var.materialize((self.num_features,))
            self.reset_parameters()
