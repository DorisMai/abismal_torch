title = "Architecture"
description = "Arguments affecting the model architecture and dimensions"

args_and_kwargs = (
    (
        ("--mlp-width",),
        {
            "help": "The number of channels (previously called d-model) in the model with default 32.",
            "default": 32,
            "type": int,
        },
    ),
    (
        ("--mlp-depth",),
        {
            "help": "The number of feedfoward units (previously called layers) with default 20.",
            "default": 20,
            "type": int,
        },
    ),
    (
        ("--use-glu",),
        {
            "help": "Use Gated Linear Unit (GLU) activation functions in the scale model.",
            "action": "store_true",
        },
    ),
    (
        ("--activation",),
        {
            "help": "The name of the activation function used in the scale model."
            "The default is 'SwiGLU' if GLU is used and 'ReLU' otherwise.",
            "default": None,
            "type": str,
        },
    ),
    (
        ("--share-weights",),
        {
            "help": "Share weights between the scaling and image MLPs.",
            "action": "store_true",
        },
    ),
)
