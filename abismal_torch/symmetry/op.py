from typing import Optional, Self

import gemmi
import numpy as np
import reciprocalspaceship as rs
import torch


class Op(torch.nn.Module):
    """
    A torch module that wraps a gemmi.Op object and
    applies it to a tensor of hkl indices.
    """
    def __init__(self, op: str, device: Optional[str] = None) -> None:
        """
        Initialize from a string of triplet (e.g. "-y,x-y,z+1/3").
        """
        super().__init__()
        dtype = torch.float32

        if not isinstance(op, str):
            raise ValueError(f"Expected str of triplet, got {type(op)}")
        self.__gemmi_op__ = gemmi.Op(op)

        self.rot = torch.nn.Parameter(
            torch.tensor(self.__gemmi_op__.rot, dtype=dtype, device=device),
            requires_grad=False,
        )
        self.den = torch.nn.Parameter(
            torch.tensor(self.__gemmi_op__.DEN, dtype=dtype, device=device),
            requires_grad=False,
        )
        self.identity = self.__gemmi_op__ == "x,y,z"

    def __str__(self) -> str:
        return f"Op({self.__gemmi_op__.triplet()})"

    def forward(self, hkl) -> torch.Tensor:
        if self.identity:
            return hkl
        dtype = hkl.dtype
        hkl = hkl.type(torch.float32)
        hkl = torch.floor_divide(hkl @ self.rot, self.den)
        hkl = hkl.type(dtype)
        return hkl

    def to_gemmi(self) -> gemmi.Op:
        """
        Return the gemmi.Op object.
        """
        return self.__gemmi_op__

    @classmethod
    def from_gemmi(cls, op) -> Self:
        """
        Alternative constructor from a gemmi.Op object.
        """
        return cls(op.triplet())
