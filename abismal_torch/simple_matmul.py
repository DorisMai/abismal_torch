def matmul_and_addone(tensor):
    """blah blah blah

    :meta public:
    """
    return tensor @ tensor.T + 1


def matmul_and_addone_hmmm(tensor):
    """hello there you shouldn't see me either

    :meta private:
    """
    return tensor @ tensor.T + 1


def __matmul_and_addone_private(tensor):
    """you should NOT see me"""
    return tensor @ tensor.T + 1


def matmul_and_addone_public(tensor):
    """you should see me!!!"""
    return tensor @ tensor.T + 1
