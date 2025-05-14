title = "IO"
description = "Arguments controlling file inputs and outputs."


args_and_kwargs = (
    (
        (
            "-o",
            "--out-dir",
        ),
        {
            "help": "The directory in which to output results. The current working directory by default",
            "default": ".",
            "type": str,
        },
    ),
    (
        ("--num-cpus",),
        {
            "help": "Number of CPUs to use for parsing CrystFEL .stream files with default 1.\
             This number is also used as num_workers for the DataLoader.",
            "default": 1,
            "type": int,
        },
    ),
    (
        ("inputs",),
        {
            "nargs": "+",
            "help": "Either .stream files from CrystFEL or .refl and .expt files from dials",
        },
    ),
    (
        ("--wavelength",),
        {
            "type": float,
            "default": None,
            "help": "Override the wavelengths inferred from the inputs.",
        },
    ),
    (
        ("--ckpt-path",),
        {
            "help": "Path to a checkpoint file to resume training from.",
            "default": None,
            "type": str,
        },
    ),
    (
        ("--save-every-nepochs",),
        {
            "help": "Save a checkpoint every n epochs.",
            "default": 1,
            "type": int,
        },
    ),
    (
        ("--log-run-name",),
        {
            "help": "Name of the run for logging.",
            "default": None,
            "type": str,
        },
    ),
    (
        ("--pin-memory",),
        {
            "help": "Pin memory for the DataLoader.",
            "action": "store_true",
        },
    ),
    (
        ("--persistent-workers",),
        {
            "help": "Persistent workers for the DataLoader.",
            "action": "store_true",
        },
    ),
    # (
    #     ("--separate",),
    #     {
    #         "help": "Merge the contents of each input file into a separate output file. ",
    #         "action": "store_true",
    #     },
    # ),
)
