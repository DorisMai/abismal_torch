# Contributing to `mypackage`

We invite anyone to contribute to this project and help make it better. Ways to contribute include filing [Issues on GitHub](https://github.com/DorisMai/mypackage/issues), adding examples and tutorials, and filing [Pull Requests](https://github.com/DorisMai/mypackage/pulls) (PR).

## Overview for submitting a PR
1. Fork the repository on GitHub.
2. Clone your forked repository locally.
3. Setup a [development environment](#setting-up-a-development-environment).
4. Make your changes and commit them.
5. Push to your forked repository and submit a PR.

## Checklist before submitting a PR

Before submitting a pull request, please ensure that:
- Changes are based on the latest main branch.
- A new [test](#testing) is added for any new feature.
- Your code follows the project's [guidelines](#specific-guidelines) and passes all tests.
- Commit messages are descriptive.
- PR clearly describes why and what changes.

## Specific guidelines
### Setting up a development environment

1. [Install a package manager](https://hatch.pypa.io/latest/install/).
This project uses [`poetry`](https://python-poetry.org/docs/basic-usage/), which can be installed using [`pipx`](https://pipx.pypa.io/stable/installation/):
  ```
  pipx install poetry
  ```
  You can optionally [enable tab completion](https://python-poetry.org/docs/#enable-tab-completion-for-bash-fish-or-zsh) with `poetry` in your shell.

2. Go to the project directory and set up a new virtual environment.
- You can do this with `conda` (e.g. to specify a particular Python version):
  ```
  conda create -n myenv python=3.10
  conda activate myenv
  ```
  and deactivate with `conda deactivate`.
  The activated environment that's not `(base)` is automatically recognized by `poetry`.
- or, you can set up a new virtual environment with `poetry`:
  ```
  poetry shell
  ```
  and exit the virtual environment with `exit`.

  Note that by default `poetry` uses the Python that was used during its installation. If you want to use a different Python version (e.g. the one installed with `conda`), you can specify it with:
  ```
  poetry env use /path/to/python
  ```

3. Install the project dependencies:
  ```
  poetry install
  ```
  To update the dependencies, run:
  ```
  poetry update
  ```

### Testing

If you are adding a feature or fixing a bug that was undetected by previous tests, please add the corresponding test.

This project uses [pytest](https://docs.pytest.org/en/8.0.x/) for running tests:
```
poetry run pytest [path/to/test_file.py::TestClassName]
```
If no path is specified, all tests (named as `test_*.py` or `*_test.py`) in the `tests` directory will be run.

Please make sure you are in the project root directory and have run `poetry install` before running the tests.

#### Continuous integration (CI)

Upon any PR, the tests will automatically run on GitHub Actions. The workflow is defined in the `.github/workflows/build.yml` file which runs the tests and checks code coverage. Only PRs that pass the tests will be merged.

### Style and Formatting

This projects sets up pre-commit hooks to ensure that the code is formatted correctly. The hooks are defined in the `.pre-commit-config.yaml` file and run automatically before each commit.

### Documentation
This project uses [MkDocs](https://www.mkdocs.org/) to build the documentation from Markdown files, which can be found in the `docs` directory.

You can build the documentation using `mkdocs` locally with:
```
mkdocs build
```
and preview the documentation website using:
```
mkdocs serve
```
Note that the build command generates a `site` directory which contains the HTML files but it is not tracked by Git.
