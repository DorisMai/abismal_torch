from typing import Optional

import gemmi
import numpy as np
import torch
from reciprocalspaceship.decorators import cellify, spacegroupify
from reciprocalspaceship.utils import apply_to_hkl, generate_reciprocal_asu


class ReciprocalASU(torch.nn.Module):
    @cellify
    @spacegroupify
    def __init__(
        self,
        cell: gemmi.UnitCell,
        spacegroup: gemmi.SpaceGroup,
        dmin: float,
        anomalous: Optional[bool] = True,
        device: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Base Layer that maps observed reflections to the reciprocal asymmetric unit (rasu).

        Args:
            cell (gemmi.UnitCell): Unit cell parameters.
            spacegroup (gemmi.SpaceGroup): Space group.
            dmin (float): Highest resolution in Ångstroms.
            anomalous (bool): If true, treat Friedel mates as non-redudant.

        Attributes:
            asu_size (int): Number of unique reflections in the rasu.
            Hmax (np.array(np.int32)): Maximum h, k, and l in the rasu under the given dmin.
            Hasu (np.ndarray(np.int32)): Unique Miller indices in the rasu.
            centric (torch.nn.Parameter): a boolean array indicating if each unique reflection
              is centric.
            multiplicity (torch.nn.Parameter): a float array with the multiplicity of each unique
              reflection. This variable is previously named as epsilon.
            reflection_id (torch.nn.Parameter): a 3D numpy array that contains the 1D integer ID
                of each unique reflection. ID value ranges from 0 to asu_size, and is -1 if the
                reflection is not in the rasu due to dmin cutoff, systematic absences, etc. This
                variable is previously named as miller_id.

        """
        super().__init__(**kwargs)
        self.cell = cell
        self.spacegroup = spacegroup
        self.anomalous = anomalous
        self.dmin = dmin

        self.Hasu = generate_reciprocal_asu(cell, spacegroup, dmin, anomalous)
        self.asu_size = len(self.Hasu)

        go = spacegroup.operations()
        self.Hmax = np.array(cell.get_hkl_limits(dmin))
        hmax, kmax, lmax = self.Hmax
        reflection_id = -np.ones((2 * hmax + 1, 2 * kmax + 1, 2 * lmax + 1), dtype=int)
        for op in go:
            Hop = apply_to_hkl(self.Hasu, op)
            h, k, l = Hop.T
            reflection_id[h, k, l] = np.arange(self.asu_size)
            if not anomalous:
                h, k, l = -Hop.T
                reflection_id[h, k, l] = np.arange(self.asu_size)
        self.reflection_id = torch.nn.Parameter(
            torch.tensor(reflection_id, dtype=torch.int, device=device),
            requires_grad=False,
        )

        self.centric = torch.nn.Parameter(
            torch.tensor(
                go.centric_flag_array(self.Hasu), dtype=torch.bool, device=device
            ),
            requires_grad=False,
        )
        self.multiplicity = torch.nn.Parameter(
            torch.tensor(
                go.epsilon_factor_array(self.Hasu), dtype=torch.float32, device=device
            ),
            requires_grad=False,
        )

    def _sanitize_H(self, H: torch.Tensor) -> torch.Tensor:
        """
        Helper function to remove Miller indices in H that are out of bounds for miller_id
            or are not in the rasu.
        """
        H = H[(torch.abs(H) <= torch.tensor(self.Hmax)).all(axis=1)]
        H = H[self.reflection_id[H[:, 0], H[:, 1], H[:, 2]] >= 0]
        return H

    def gather(self, source: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            source (torch.Tensor): A tensor of shape (self.asu_size, ...)
            H (torch.Tensor): A nested tensor of shape (n_images, n_refln_per_image, 3)
                where n_refln_per_image varies.

        Returns:
            gathered (torch.Tensor): A nested tensor of shape (n_images, ...)
        """
        n_images = len(H)
        gathered = []
        for i in range(n_images):
            H[i] = self._sanitize_H(H[i])
            _reflection_id = self.reflection_id[H[i][:, 0], H[i][:, 1], H[i][:, 2]]
            gathered.append(source[_reflection_id])
        print(gathered)
        return torch.nested.nested_tensor(gathered)
