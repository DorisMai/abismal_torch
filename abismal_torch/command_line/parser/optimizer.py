title = "Optimizer"
description = "Arguments affecting the optimization algorithm"


args_and_kwargs = (
    (
        ("--learning-rate",),
        {
            "help": "Learning rate for Adam with default 1e-3.",
            "default": 1e-3,
            "type": float,
        },
    ),
    (
        ("--beta-1",),
        {
            "help": "First moment momentum parameter for Adam with default 0.9.",
            "default": 0.9,
            "type": float,
        },
    ),
    (
        ("--beta-2",),
        {
            "help": "Second moment momentum parameter for Adam with default 0.999.",
            "default": 0.999,
            "type": float,
        },
    ),
    (
        ("--adam-epsilon",),
        {
            "help": "A small constant for numerical stability with default 1e-9.",
            "default": 1e-9,
            "type": float,
        },
    ),
    (
        ("--amsgrad",),
        {
            "help": "Optionally use the amsgrad variant of Adam.",
            "action": "store_true",
        },
    ),
)
