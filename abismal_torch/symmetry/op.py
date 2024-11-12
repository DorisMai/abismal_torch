from typing import Optional

import gemmi
import torch


class Op(torch.nn.Module):
    """Torch wrapping of a gemmi.Op object. It can be applied to a tensor
    of Miller indices of size 3.

    Args:
        op (str): The triplet of the operator, e.g. "x,-y,z+1/3".

    Attributes:
        _rot (torch.Tensor): The rotation matrix of the operator. Registered as a buffer.
        _den (torch.Tensor): The denominator of the operator. Registered as a buffer.
        _identity (bool): Whether the operator is the identity. Registered as a buffer.
    """

    def __init__(self, op: str) -> None:
        super().__init__()
        _dtype = torch.float32

        if not isinstance(op, str):
            raise ValueError(f"Expected str of triplet, got {type(op)}")
        self.__gemmi_op__ = gemmi.Op(op)

        self.register_buffer(
            "_rot",
            torch.tensor(self.__gemmi_op__.rot, dtype=_dtype),
        )
        self.register_buffer(
            "_den",
            torch.tensor(self.__gemmi_op__.DEN, dtype=_dtype),
        )
        self.register_buffer(
            "_identity",
            torch.tensor(self.__gemmi_op__ == "x,y,z", dtype=torch.bool),
        )

    def __str__(self) -> str:
        return f"Op({self.__gemmi_op__.triplet()})"

    def forward(self, hkl: torch.Tensor) -> torch.Tensor:
        if self._identity:
            return hkl
        dtype = hkl.dtype
        hkl = hkl.type(torch.float32)
        hkl = torch.floor_divide(hkl @ self._rot, self._den)
        hkl = hkl.type(dtype)
        return hkl

    def to_gemmi(self) -> gemmi.Op:
        """Returns the gemmi.Op object."""
        return self.__gemmi_op__

    @classmethod
    def from_gemmi(cls, op: gemmi.Op):
        """An alternative constructor from a gemmi.Op object.

        Args:
            op ("gemmi.Op"): The gemmi.Op object.

        Returns:
            Op: The torch-wrapped operator.
        """
        return cls(op.triplet())
