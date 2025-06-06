title = "Priors"
description = "Arguments governing the prior distributions"

from .custom_types import list_of_floats, list_of_ints, list_of_ops

args_and_kwargs = (
    (
        ("--kl-weight",),
        {
            "help": "The strength of the structure factor prior distribution with default 1.0.",
            "default": 1.0,
            "type": float,
        },
    ),
    (
        ("--scale-kl-weight",),
        {
            "help": "The strength of the scale prior distribution with default 1.0.",
            "default": 1.0,
            "type": float,
        },
    ),
    (
        ("--parents",),
        {
            "help": "Set parent asu for the Multi-Wilson prior. This is used with --separate flag."
            "Supply parent asu ids as a comma separated list of integers. If an asu has no"
            " parent, supply its own asu id. for example, --parents 0,0, indicates that "
            "the first asu has no parent and the second asu is dependent on the first.",
            "default": None,
            "type": list_of_ints,
        },
    ),
    (
        (
            "-r",
            "--prior-correlation",
        ),
        {
            "help": "The prior correlation (r-value) for each ASU and its parent. Supply "
            "comma-separated floating point Values. "
            "Values supplied for ASUs without a parent will be ignored. "
            "Example: -r 0.0,0.99",
            "default": None,
            "type": list_of_floats,
        },
    ),
    (
        ("--normalization-sigma",),
        {
            "help": "The normalization sigma value. This represents the average intensity stratified by a measure like resolution. Defaults to 1.0.",
            "default": 1.0,
            "type": float,
        },
    ),
    (
        ("--reindexing-ops",),
        {
            "help": "Supply semicolon-separated reindexing ops which map from the child asu"
            "convention into the parent convention. "
            'For example, --reindexing-ops "x,y,z;-x,-y,-z". ',
            "default": None,
            "type": list_of_ops,
        },
    ),
    (
        ("--rasu-ids",),
        {
            "help": "The reciprocal asymmetric unit (RASU) ids corresponding to the input files. "
            "Supply a comma-separated list of integers starting from 0. If not provided, each input\
            file will be assigned a unique RASU id. Merging output files are saved per RASU."
            'For example, --rasu-ids "0,1,0". ',
            "default": None,
            "type": list_of_ints,
        },
    ),
)
