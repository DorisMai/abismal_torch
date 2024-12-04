import math
from typing import Optional, Sequence

import torch
from rs_distributions import distributions as rsd

from abismal_torch.prior.base import DistributionBase, PriorBase
from abismal_torch.symmetry import ReciprocalASUCollection, ReciprocalASUGraph


class GenericWilsonDistribution(DistributionBase):
    def __init__(
        self,
        centric: torch.Tensor,
        multiplicity: torch.Tensor,
        normalization_sigma: Optional[float | torch.Tensor] = 1.0,
        single_wilson: Optional[bool] = True,
        correlation: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Generic Wilson distribution for structure factor amplitudes.
        For single Wilson distribution,
        ```
        Wilson(F_h) =
            FoldedNormal(0, sqrt(multiplicity * normalization_sigma)) #centric
            Rice(0., sqrt(0.5 * multiplicity * normalization_sigma)   #acentric
        ```
        is equivalent to
        ```
        Wilson(F_h) =
            HalfNormal(sqrt(multiplicity * normalization_sigma)) #centric
            Weibull(sqrt(0.5 * multiplicity * normalization_sigma), 2)   #acentric
        ```
        For Double Wilson distribution,
        ```
        DoubleWilson(F_h) =
            FoldedNormal(r_h * z_Pa(h), sqrt(multiplicity * normalization_sigma * (1 - r^2))) #centric
            Rice(r_h * z_Pa(h), sqrt(0.5 * multiplicity * normalization_sigma * (1 - r^2)))   #acentric
        ```
        where `r` is the correlation coefficient between parent and child rasus and `z_Pa(h)`
        is a sample from the parent distribution.

        Args:
            centric (torch.Tensor): A tensor of shape (n_refln,).
            multiplicity (torch.Tensor): A tensor of shape (n_refln,).
            normalization_sigma (float | torch.Tensor, optional): Normalization sigma value. This represents
                the average intensity stratified by a measure like resolution. If this is a tensor,
                it must be of shape (n_refln,). Defaults to 1.0.
            single_wilson (bool, optional): If True, a single Wilson distribution is used for all reflections.
                If False, a parent-child Wilson distribution is used for all reflections. Defaults to True.
            correlation (torch.Tensor, optional): A tensor of shape (n_refln,). Defaults to None, which
                is equivalent to a tensor of (n_refln,) zeros. Only used if single_wilson is False.
        """
        self.centric = centric
        self.multiplicity = multiplicity
        self.normalization_sigma = normalization_sigma
        self.single_wilson = single_wilson
        if self.single_wilson or correlation is None:
            self.correlation = torch.zeros_like(self.centric)
        else:
            self.correlation = correlation

        self._loc = self.correlation
        self._scale = torch.sqrt(
            self.multiplicity
            * (1.0 - torch.square(self.correlation))
            * self.normalization_sigma
        )

    def p_centric(
        self, z: Optional[float | torch.Tensor] = 1.0
    ) -> torch.distributions.Distribution:
        """
        Structure factor amplitude distribution for centric reflections.
        """
        return rsd.FoldedNormal(self._loc * z, self._scale)

    def p_acentric(
        self, z: Optional[float | torch.Tensor] = 1.0
    ) -> torch.distributions.Distribution:
        """
        Structure factor amplitude distribution for acentric reflections.
        """
        return rsd.Rice(self._loc * z, math.sqrt(0.5) * self._scale)

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute the log probability of the Wilson distribution depending on whether it is
        a single Wilson distribution or a parent-child Wilson distribution.

        Args:
            z (torch.Tensor): A tensor of shape (mc_samples, n_reflections).

        Returns:
            torch.Tensor: A tensor of same shape as z containing the log probabilities.
        """
        if self.single_wilson:
            return torch.where(
                self.centric,
                self.p_centric().log_prob(z),
                self.p_acentric().log_prob(z),
            )
        else:
            return torch.where(
                self.centric,
                self.p_centric(z).log_prob(z),
                self.p_acentric(z).log_prob(z),
            )


class MultiWilsonDistribution(DistributionBase):
    def __init__(
        self,
        is_root: torch.Tensor,
        parent_reflection_ids: torch.Tensor,
        centric: torch.Tensor,
        multiplicity: torch.Tensor,
        normalization_sigma: Optional[float | torch.Tensor] = 1.0,
        correlation: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Wrapper class to handle mixture of single Wilson distribution for root nodes
        and Double Wilson distribution otherwise.

        Args:
            is_root (torch.Tensor): A boolean tensor of shape (n_refln,).
            parents_reflection_ids (torch.Tensor): A tensor of shape (n_refln,).

        Attributes:
            root_wilson (GenericWilsonDistribution): Single Wilson distribution for all nodes.
            parent_wilson (GenericWilsonDistribution): Double Wilson distribution for non-root nodes.
        """
        self.is_root = is_root
        self.parent_reflection_ids = parent_reflection_ids
        self.root_wilson = GenericWilsonDistribution(
            centric, multiplicity, normalization_sigma
        )
        self.parent_wilson = GenericWilsonDistribution(
            centric,
            multiplicity,
            normalization_sigma,
            single_wilson=False,
            correlation=correlation,
        )

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z (torch.Tensor): A tensor of shape (mc_samples, n_reflections).

        Returns:
            torch.Tensor: A tensor of same shape as z containing the log probabilities.
        """
        z_root = z
        z_parent = z[:, self.parent_reflection_ids]
        logprob_root = self.root_wilson.log_prob(z_root)
        logprob_parent = self.parent_wilson.log_prob(z_parent)
        return torch.where(self.is_root, logprob_root, logprob_parent)


class WilsonPrior(PriorBase):
    def __init__(
        self,
        rac: ReciprocalASUCollection,
        normalization_sigma: Optional[float | torch.Tensor] = 1.0,
        **kwargs
    ) -> None:
        """
        Single Wilson prior for structure factor amplitudes.

        Args:
            rac (ReciprocalASUCollection): ReciprocalASUCollection instance.
            normalization_sigma (float | torch.Tensor, optional): Normalization sigma value.
                See GenericWilsonDistribution. If this is a tensor, it must be of shape
                (rac_size,). Defaults to 1.0.
        """
        super().__init__(**kwargs)
        self.rac = rac
        self.normalization_sigma = normalization_sigma

    def distribution(
        self, rasu_id: Optional[torch.Tensor] = None, hkl: Optional[torch.Tensor] = None
    ) -> GenericWilsonDistribution:
        """
        Args:
            rasu_id (torch.Tensor, optional): RASU IDs to get distribution for. Defaults to None,
                implicitly assumes rac.rasu_ids.
            hkl (torch.Tensor, optional): Miller indices to get distribution for. Only
                used when rasu_id is not None and must have same length with rasu_id.

        Returns:
            p (GenericWilsonDistribution): Single Wilson distribution for the specified number of
            reflections, which is rac_size if rasu_id is None and len(rasu_id) otherwise.
        """
        normalization_sigma = self.normalization_sigma
        if rasu_id is None:
            centric = self.rac.centric
            multiplicity = self.rac.multiplicity
            rasu_id = self.rac.rasu_ids
        else:
            centric = self.rac.gather(self.rac.centric, rasu_id, hkl)
            multiplicity = self.rac.gather(self.rac.multiplicity, rasu_id, hkl)
            if not isinstance(normalization_sigma, float):
                normalization_sigma = self.rac.gather(normalization_sigma, rasu_id, hkl)
        p = GenericWilsonDistribution(centric, multiplicity, normalization_sigma)
        return p


class MultiWilsonPrior(torch.nn.Module):
    def __init__(
        self,
        rag: ReciprocalASUGraph,
        correlation: Sequence[float],
        normalization_sigma=1.0,
        **kwargs
    ) -> None:
        """
        Multi-Wilson prior for structure factor amplitudes from a ReciprocalASUGraph.
        Args:
            rag (ReciprocalASUGraph): ReciprocalASUGraph instance.
            correlation (Sequence[float]): User supplied correlation coefficients between
                parent-child asus. Use 0 for root nodes.
            normalization_sigma (float | torch.Tensor): Normalization sigma value. See
                GenericWilsonDistribution.
        """
        super().__init__(**kwargs)
        self.rag = rag
        self.correlation = torch.tensor(correlation)
        self.normalization_sigma = normalization_sigma

    def distribution(
        self, rasu_id: Optional[torch.Tensor] = None, hkl: Optional[torch.Tensor] = None
    ) -> MultiWilsonDistribution:
        """
        Args:
            rasu_id (torch.Tensor, optional): RASU IDs to get distribution for. Defaults to None,
                implicitly assumes rag.rasu_ids.
            hkl (torch.Tensor, optional): Miller indices to get distribution for. Only
                used when rasu_id is not None and must have same length with rasu_id.

        Returns:
            p (MultiWilsonDistribution): MultiWilsonDistribution instance for the specified number of
            reflections, which is rac_size if rasu_id is None and len(rasu_id) otherwise.
        """
        if rasu_id is None:
            centric = self.rag.centric
            multiplicity = self.rag.multiplicity
            root = self.rag.is_root
            parent_reflection_ids = self.rag.parent_reflection_ids
            correlation = self.correlation[self.rag.rasu_ids]
        else:
            centric = self.rag.gather(self.rag.centric, rasu_id, hkl)
            multiplicity = self.rag.gather(self.rag.multiplicity, rasu_id, hkl)
            root = self.rag.gather(self.rag.is_root, rasu_id, hkl)
            parent_reflection_ids = self.rag.gather(
                self.rag.parent_reflection_ids, rasu_id, hkl
            )
            correlation = self.correlation[rasu_id]
            if not isinstance(self.normalization_sigma, float):
                self.normalization_sigma = self.rag.gather(
                    self.normalization_sigma, rasu_id, hkl
                )
        p = MultiWilsonDistribution(
            root,
            parent_reflection_ids,
            centric,
            multiplicity,
            self.normalization_sigma,
            correlation,
        )
        return p
