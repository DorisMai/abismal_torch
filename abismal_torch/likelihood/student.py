import torch


class StudentTLikelihood(torch.nn.Module):
    def __init__(self, dof: float):
        super().__init__()
        self.dof = dof

    def forward(
        self, Ipred: torch.Tensor, Iobs: torch.Tensor, SigIobs: torch.Tensor
    ) -> torch.Tensor:
        if len(Iobs.shape) == 1:
            Iobs = Iobs[:, None]
        if len(SigIobs.shape) == 1:
            SigIobs = SigIobs[:, None]

        p = torch.distributions.StudentT(self.dof, Iobs, SigIobs)
        ll = p.log_prob(Ipred)
        return ll
