import gemmi
import pytest
import torch

from abismal_torch.symmetry import Op


@pytest.mark.parametrize("op_triplet", ["x,y,z", "-y,x-y,z+1/3"])
def test_op_construction(op_triplet):
    gemmi_op = gemmi.Op(op_triplet)
    assert Op(op_triplet).to_gemmi() == gemmi_op
    op1 = Op(op_triplet)
    op2 = Op.from_gemmi(gemmi_op)
    assert op1.to_gemmi() == op2.to_gemmi()
    assert op1.identity == op2.identity
    if op_triplet == "x,y,z":
        assert Op(op_triplet).identity


@pytest.mark.parametrize("op_triplet", ["x,y,z", "-y,x-y,z+1/3"])
@pytest.mark.parametrize("hkl", [[3, 0, 1]])
def test_op_forward(op_triplet, hkl):
    equivalent_hkl = gemmi.Op(op_triplet).apply_to_hkl(hkl)
    op = Op(op_triplet)
    assert torch.equal(op(torch.tensor(hkl)), torch.tensor(equivalent_hkl))
