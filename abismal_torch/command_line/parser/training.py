title = "Training options"
description = "Options that deal with the specifics of model training"


args_and_kwargs = (
    (
        ("--epsilon",),
        {
            "help": "A small constant for numerical stability.",
            "default": 1e-12,
            "type": float,
        },
    ),
    (
        ("--mc-samples",),
        {
            "help": "The number of monte carlo samples used to estimate gradients with default 256.",
            "default": 32,
            "type": int,
        },
    ),
    (
        ("--steps-per-epoch",),
        {
            "help": "The number of gradient steps in an epoch dictating how often output can be saved. 1000 is the default.",
            "default": 1_000,
            "type": int,
        },
    ),
    (
        ("--epochs",),
        {
            "help": "The number of training epochs to run with default 30.",
            "default": 30,
            "type": int,
        },
    ),
    (
        ("--validation-steps",),
        {
            "help": "The number of validation steps run at the close of each epoch with default 100.",
            "default": 100,
            "type": int,
        },
    ),
    (
        ("--test-fraction",),
        {
            "help": "The fraction of images reserved for validation with default 0.01.",
            "default": 0.01,
            "type": float,
        },
    ),
    (
        ("--batch-size",),
        {
            "help": "The size (number of images) in each training batch",
            "default": 100,
            "type": int,
        },
    ),
    # (
    #     ("--disable-index-disambiguation",),
    #     {
    #         "help": "Disable index disambiguation if applicable to the space group.",
    #         "action": "store_true",
    #     },
    # ),
    (
        ("--seed",),
        {
            "help": "The seed for the random number generator.",
            "default": 1234,
            "type": int,
        },
    ),
    (
        ("--accelerator",),
        {
            "help": "The accelerator to use for training.",
            "default": "cpu",
            "type": str,
        },
    ),
)
