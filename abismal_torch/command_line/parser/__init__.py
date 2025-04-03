"""
ABISMAL - merge serial diffraction data using neural networks and variational inference.
"""

import abismal_torch.command_line.parser.architecture as architecture
import abismal_torch.command_line.parser.io as io
import abismal_torch.command_line.parser.likelihood as likelihood
import abismal_torch.command_line.parser.optimizer as optimizer
import abismal_torch.command_line.parser.priors as priors
import abismal_torch.command_line.parser.surrogate_posterior as surrogate_posterior
import abismal_torch.command_line.parser.training as training

groups = [
    architecture,
    io,
    likelihood,
    optimizer,
    priors,
    surrogate_posterior,
    training,
]

from argparse import ArgumentParser

parser = ArgumentParser(description=__doc__)
for group in groups:
    g = parser.add_argument_group(group.title, group.description)
    for args, kwargs in group.args_and_kwargs:
        g.add_argument(*args, **kwargs)

all = [
    parser,
]
