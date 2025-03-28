import torch


class NormalLikelihood(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, Ipred: torch.Tensor, Iobs: torch.Tensor, SigIobs: torch.Tensor
    ) -> torch.Tensor:
        if len(Iobs.shape) == 1:
            Iobs = Iobs[:, None]
        if len(SigIobs.shape) == 1:
            SigIobs = SigIobs[:, None]

        p = torch.distributions.Normal(Iobs, SigIobs)
        ll = p.log_prob(Ipred)
        return ll
