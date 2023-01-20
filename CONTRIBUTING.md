# Contributing to pymovements
Thank you for taking your time to contribute to pymovements! We encourage you to report any bugs or contribute
to new features, optimisations or documentation.

Here we give you an overview of the workflow and best practices for contributing
to pymovements.

*Questions:* If you have any developer-related questions, please [open an issue](
https://github.com/aeye-lab/pymovements/issues/new/choose) or write us at
[daniel.krakowczyk@uni-potsdam.de](mailto:daniel.krakowczyk@uni-potsdam.de).


## Table of Contents
- [Reporting Bugs](#reporting-bugs)
- [First-time Contributors](#first-time-contributors)
- [Getting Started](#getting-started)
  - [Development Installation](#development-installation)
  - [Creating a Branch](#creating-a-branch)
  - [Code Style](#code-style)
  - [Testing](#testing)
  - [Documentation](#documentation)
  - [Pull Requests](#pull-requests)
  - [Continuous Integration](#continuous-integration)
- [Core Developer Guidelines](#core-developer-guidelines)
- [License](#license)
- [Questions](#questions)


## Code of Conduct

Everyone participating in the pymovements project, and in particular in our issue tracker and pull
requests, is expected to treat other people with respect and more generally to follow the guidelines
articulated in the [Python Community Code of Conduct](https://www.python.org/psf/codeofconduct/).


## Reporting Bugs

If you discover a bug, as a first step please check the existing
[Issues](https://github.com/aeye-lab/pymovements/issues) to see if this bug has already been
reported. In case the bug has not been reported yet, please do the following:

- [Open an issue](https://github.com/aeye-lab/pymovements/issues/new?labels=bug&template=ISSUE.md).
- Add a descriptive title to the issue and write a short summary of the problem.
- We provide you with a default template to guide you through a typical reporting process.
- Adding more context, including error messages and references to the problematic parts of the code,
would be very helpful to us.

Once a bug is reported, our development team will try to address the issue as quickly as possible.


## First-time Contributors

If you're looking for things to help with, try browsing our [issue tracker](
https://github.com/aeye-lab/pymovements/issues) first.

In particular, look for:

- [good first issues](https://github.com/aeye-lab/pymovements/labels/good-first-issue)
- [documentation issues](https://github.com/aeye-lab/pymovements/labels/documentation)

You do not need to ask for permission to work on any of these issues. The current status of the
issue will let you know if someone else is or was already working on it.

To get help fixing a specific issue, it's often best to comment on the issue itself. You're much
more likely to get targeted help if you provide details about what you've tried and where you've looked.

To start out with developing, [install the dependencies](#development-installation) and
[create a branch](#creating-a-branch) for your contribution.


Open a [pull request](#pull-requests) when you feel confident to publish your progress. Don't
hesitate if it's a work in progress, we can give you early feedback on your work.
If you can, try to add [unit tests](#testing) early on to verify correctness.


## Getting Started

This is a general guide to contributing changes to pymovements. Before you start developing, make
sure to read our [documentation](https://pymovements.readthedocs.io/) first.


### Development Installation

Make sure to install the latest pymovements version from the main branch.

```bash
git clone https://github.com/aeye-lab/pymovements.git
cd pymovements
pip install -e .
```


### Creating a Branch

Before you start making changes to the code, create a local branch from the latest version of the
`main` branch.

```bash
git checkout main
git pull origin main
git checkout -b feature/your-feature-branch
```

We do not allow for pushing directly to the `main` branch and merge changes exclusively by
[pull requests](#pull-requests).

We will squash your commits into a single commit on merge to maintain a clean git history.
We use a linear git-history, where each commit contains a full feature/bug fix, such that each
commit represents an executable version. This way you also don't have to worry much about your
intermediate commits and can focus on getting your work done first.


### Code Style

We write our code to follow [PEP-8](https://www.python.org/dev/peps/pep-0008) with a maximum
line-width of 100 characters. We additionally use type annotations as in [PEP-484](
https://peps.python.org/pep-0484). For docstrings we use the [numpydoc](
https://numpydoc.readthedocs.io/en/latest/format.html) formatting standard.

We use [`flake8`](https://pypi.org/project/flake8/) for quick style checks and
[`pylint`](https://pypi.org/project/pylint/) for thorough style checks and [`mypy`](
https://pypi.org/project/mypy/) for checking type annotations.


### Testing

Tests are written using [Pytest](https://docs.pytest.org) and executed
in a separate environment using [Tox](https://tox.readthedocs.io/en/latest/).

A full style check and all tests can be run by simply calling `tox` in the repository root.
```bash
tox
```
Running `tox` the first time in the repository will take a few minutes, as all necessary python
environments will have to be set up with their dependencies. Runtime should be short on the
subsequent runs.

If you add a new feature, please also include appropriate tests to verify its intended
functionality. We try to keep our code coverage close to 100%.

It is possible to limit the scope of testing to specific environments and files, for example, to 
only test transformations using the Python 3.7 environment use:
```bash
tox -e py37 tests/test_transformations.py
```


### Documentation

Make sure to add docstrings to every class, method and function that you add to the codebase.
Docstrings should include a description of all parameters, returns and exceptions. Use the existing
documentation as an example.

The API-documentation is generated from the numpydoc-style docstring of the respective
modules/classes/functions by [Sphinx](https://www.sphinx-doc.org).
You can build the documentation locally using the respective tox environment:
```bash
tox -e docs
```
It will appear in the `build/docs` directory.

To rebuild the full documentation use
```bash
tox -e docs -- -aE
```


### Pull Requests

Once you are ready to publish your changes:

- Create a [pull request (PR)](https://github.com/aeye-lab/pymovements/compare)
- Provide a summary of the changes you are introducing according to the [template](
https://github.com/aeye-lab/pymovements/blob/main/.github/PULL_REQUEST_TEMPLATE.md).
- In case you are resolving an issue, don't forget to link it.

The [pull request template](
https://github.com/aeye-lab/pymovements/blob/main/.github/PULL_REQUEST_TEMPLATE.md) is meant as a
helper and should guide you through the process of creating a pull request. 
It's also OK to submit work in progress, in which case you'll likely be asked to make some further
changes.

If your change will be a significant amount of work to write, we highly recommend starting by
opening an issue laying out what you want to do. That lets a conversation happen early in case other
contributors disagree with what you'd like to do or have ideas that will help you do it.

The best pull requests are focused, clearly describe what they're for and why they're correct, and
contain tests for whatever changes they make to the code's behavior. As a bonus these are easiest
for someone to review, which helps your pull request get merged quickly. Standard advice about good
pull requests for open-source projects applies. We have our own [template](
https://github.com/aeye-lab/pymovements/blob/main/.github/PULL_REQUEST_TEMPLATE.md) to guide you
through the process.

Do not squash your commits after you have submitted a pull request, as this
erases context during review. We will squash commits when the pull request is ready to be merged.


### Continuous Integration

Linting, tests and the documentation are all additionally checked using a Github Actions
workflow which executes the appropriate tox environments.


## Core Developer Guidelines

Core developers should follow these rules when processing pull requests:

- Always wait for tests to pass before merging PRs.
- Use "[Squash and merge](https://github.com/blog/2141-squash-your-commits)" to merge PRs.
- Delete branches for merged PRs.
- Edit the final commit message before merging to conform to the following style (we wish to have a
clean `git log` output):

```
Category: Short subject describing changes (50 characters or less)

- detailed description, wrapped at 72 characters
- bullet points or sentences are okay
- all changes should be documented and explained
- valid categories are, for example:
    - `Docs` for documentation
    - `Tests` for tests
    - `Core` for core changes
    - `Events` for changes in event detection
    - `Transforms` for changes in transformations
    - `Package` for package-related changes, e.g. in setup.cfg
```

Make sure:

  - that when merging a multi-commit PR the commit message doesn't
    contain the local history from the committer and the review history from
    the PR. Edit the message to only describe the end state of the PR.
  - that there is a *single- newline between subject and description, 
  - that there is a *single- newline at the end of the commit message.
    This way there is a single empty line between commits in `git log`
    output.
  - that the maximum subject line length is under 50 characters
  - that the maximum line length of the commit message is under 72 characters
  - to capitalize the subject and each paragraph.
  - that the subject of the commit message has no trailing dot.
  - to use the imperative mood in the subject line (e.g. "Fix typo in README").
  - if the PR fixes an issue, something like "Fixes #xxx." occurs in the body of the message (not in
the subject).
  - to use Markdown for formatting.


## License

Please note that by contributing to the project you agree that it will be licensed under the
[License](https://github.com/aeye-lab/pymovements/blob/main/LICENSE) of this project.

If you did not write the code yourself, ensure the existing license is compatible and include the
license information in the contributed files, or obtain permission from the original author to
relicense the contributed code.


## Questions

If you have any developer-related questions, please [open an issue](
https://github.com/aeye-lab/pymovements/issues/new/choose) or write us at
[daniel.krakowczyk@uni-potsdam.de](mailto:daniel.krakowczyk@uni-potsdam.de).