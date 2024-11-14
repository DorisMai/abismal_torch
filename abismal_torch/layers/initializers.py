import math
from typing import Optional

import torch
import torch.nn as nn


class VarianceScalingNormalInitializer(nn.Module):
    def __init__(
        self,
        mode: Optional[str] = "fan_avg",
        gain: Optional[float] = math.sqrt(0.1),
        low: Optional[float] = -2.0,
        high: Optional[float] = 2.0,
        generator: Optional[torch.Generator] = None,
        **kwargs
    ):
        """
        Initialize weights using variance scaling method. Currently only supports truncated
        normal distribution of mean 0 and standard deviation (std).
        ```
        std = gain * sqrt(1 / fan)
        ```
        where fan depends on the mode.

        Args:
            mode (str, optional): Mode for variance scaling. Can be 'fan_in', 'fan_out',
                or 'fan_avg'. Defaults to 'fan_avg'.
            gain (float, optional): Scaling factor for the weights. If None, keyword
                arguments for the nn.init.calculate_gain() function should be provided
                to calculate gain. Defaults to 0.1.
            low (float, optional): Lower std bound for the truncated normal distribution.
                If None, normal distribution is used.
            high (float, optional): Upper std bound for the truncated normal distribution.
                If None, normal distribution is used.
        """
        super().__init__()
        if mode not in ["fan_in", "fan_out", "fan_avg"]:
            raise ValueError(
                "Invalid mode: {}. Expected 'fan_in', 'fan_out', or 'fan_avg'.".format(
                    mode
                )
            )
        self.mode = mode
        if gain is None:
            self.gain = nn.init.calculate_gain(**kwargs)
        else:
            self.gain = gain
        self.low, self.high = low, high
        self.generator = generator
        self.fan = None

    def __call__(self, tensor: torch.Tensor) -> None:
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
        if self.mode == "fan_in":
            self.fan = fan_in
        elif self.mode == "fan_out":
            self.fan = fan_out
        else:
            self.fan = 0.5 * (tensor.size(0) + tensor.size(1))

        std = self.gain * math.sqrt(1.0 / self.fan)
        mean = 0.0
        if self.low is None or self.high is None:  # normal distribution
            print("normal distribution of mean {} and std {}".format(mean, std))
            nn.init.normal_(tensor, mean, std, generator=self.generator)
            print(tensor)
        else:  # truncated normal distribution
            print("truncated normal distribution")
            nn.init.trunc_normal_(
                tensor,
                mean,
                std,
                self.low * std,
                self.high * std,
                generator=self.generator,
            )
