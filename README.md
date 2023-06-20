# hackasaurus-rex

A longer description of your project goes here...

## Installation

In order to set up the necessary environment:

1. install in a container
   ```
   enroot create -n pyxis_torch /hkfs/work/workspace/scratch/ih5525-E2/nvidia+pytorch+23.05-py3.sqsh
   ```
   Allocate a node with interactive script: `./scripts/enroot_salloc -N 1 -G 1`
   ```
   pip install ultralitics torchmetrics transformers
   ```
   Exit the node

2. To run code, use `scripts/launch_job.sbatch` This will launch a new sbatch job with the specified python job.

## Running a Testing job

1. Make sure that the script in `scripts/launch_job.sbatch` is the target python script

2. Ensure that the config selected in `scripts/launch_job.sbatch` is what you want to run (if you want to run on
   a different dataset, change the target parameter). If you want to train on the entire dataset you are giving it, make sure to
   set `split_data` in the config to `False`

3. Once the config is updated, check to make sure that the data is included in the `TOMOUNT` list of paths in `scripts/launch_job.sbatch`.
   The `TOMOUNT` variable is composed of paths which WHICH ARE SEPERATED BY COMMAS!

4. launch the job with `sbatch scripts/launch_job.sbatch`

## Project Organization

```
├── AUTHORS.md              <- List of developers and maintainers.
├── CHANGELOG.md            <- Changelog to keep track of new features and fixes.
├── CONTRIBUTING.md         <- Guidelines for contributing to this project.
├── Dockerfile              <- Build a docker container with `docker build .`.
├── LICENSE.txt             <- License as chosen on the command-line.
├── README.md               <- The top-level README for developers.
├── configs                 <- Directory for configurations of model & application.
├── data
│   ├── external            <- Data from third party sources.
│   ├── interim             <- Intermediate data that has been transformed.
│   ├── processed           <- The final, canonical data sets for modeling.
│   └── raw                 <- The original, immutable data dump.
├── docs                    <- Directory for Sphinx documentation in rst or md.
├── environment.yml         <- The conda environment file for reproducibility.
├── models                  <- Trained and serialized models, model predictions,
│                              or model summaries.
├── notebooks               <- Jupyter notebooks. Naming convention is a number (for
│                              ordering), the creator's initials and a description,
│                              e.g. `1.0-fw-initial-data-exploration`.
├── pyproject.toml          <- Build configuration. Don't change! Use `pip install -e .`
│                              to install for development or to build `tox -e build`.
├── references              <- Data dictionaries, manuals, and all other materials.
├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures             <- Generated plots and figures for reports.
├── scripts                 <- Analysis and production scripts which import the
│                              actual PYTHON_PKG, e.g. train_model.
├── setup.cfg               <- Declarative configuration of your project.
├── setup.py                <- [DEPRECATED] Use `python setup.py develop` to install for
│                              development or `python setup.py bdist_wheel` to build.
├── src
│   └── hackasaurus_rex     <- Actual Python package where the main functionality goes.
├── tests                   <- Unit tests which can be run with `pytest`.
├── .coveragerc             <- Configuration for coverage reports of unit tests.
├── .isort.cfg              <- Configuration for git hook that sorts imports.
└── .pre-commit-config.yaml <- Configuration of pre-commit git hooks.
```

<!-- pyscaffold-notes -->

## Note

This project has been set up using [PyScaffold] 4.4.1 and the [dsproject extension] 0.7.2.

[conda]: https://docs.conda.io/
[pre-commit]: https://pre-commit.com/
[Jupyter]: https://jupyter.org/
[nbstripout]: https://github.com/kynan/nbstripout
[Google style]: http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
[PyScaffold]: https://pyscaffold.org/
[dsproject extension]: https://github.com/pyscaffold/pyscaffoldext-dsproject
