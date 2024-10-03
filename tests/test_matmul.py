import pytest
import torch
from abismal.simple_matmul import matmul_and_addone

def test_matmul_func():
    tensor = torch.ones(4, 4)
    tensor[:,1] = 0
    y1 = tensor @ tensor.T
    y2 = tensor.matmul(tensor.T)
    y3 = torch.rand_like(y1)
    torch.matmul(tensor, tensor.T, out=y3)
    #assert y1 y2 y3 are equal
    assert torch.equal(y1, y2), "y1 and y2 are not equal"
    assert torch.equal(y2, y3), "y2 and y3 are not equal"

def test_matmul_and_addone():
    tensor = torch.ones(4, 4)
    tensor[:,1] = 0
    y = tensor @ tensor.T + 1
    y_hat = matmul_and_addone(tensor)
    #assert y and y_hat are equal
    assert torch.equal(y, y_hat), "y and y_hat are not equal"
