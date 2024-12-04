title = "Architecture"
description = "Arguments affecting the model architecture and dimensions"

args_and_kwargs = (
    (
        ("--d-model",),
        {
            "help": "The number of channels (i.e. mlp_width) in the model with default 32.",
            "default": 32,
            "type": int,
        },
    ),
    (
        ("--layers",),
        {
            "help": "The number of feedfoward units (i.e. mlp_depth) with default 20.",
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
