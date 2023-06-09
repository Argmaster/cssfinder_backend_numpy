# CSSFinder NumPy Backend

Implementation of CSSFinder backend using NumPy library.

## Installing

To install CSSFinder NumPy Backend from PyPI, use `pip` in terminal:

```
pip install cssfinder_backend_numpy
```

If you want to use development version, traverse `Development` and `Packaging`
sections below.

## Development

This project uses `Python` programming language and requires at least python
`3.8` for development and distribution. Development dependencies
[`poetry`](https://pypi.org/project/poetry/) for managing dependencies and
distribution building. It is necessary to perform any operations in development
environment.

To install poetry globally (preferred way) use `pip` in terminal:

```
pip install poetry
```

Then use

```
poetry shell
```

to spawn new shell with virtual environment activated. Virtual environment will
be indicated by terminal prompt prefix `(cssfinder_backend_numpy-py3.10)`,
version indicated in prefix depends on used version of Python interpreter. It
is not necessary to use Python 3.10, however at least 3.8 is required.

Within shell with active virtual environment use:

```
poetry install --sync
```

To install all dependencies. Every time you perform a `git pull` or change a
branch, you should call this command to make sure you have the correct versions
of dependencies.

Last line should contain something like:

```
Installing the current project: cssfinder_backend_numpy (0.1.1)
```

If no error messages are shown, You are good to go.

## Packaging

A Python Wheel is a built package format for Python that can be easily
installed and distributed, containing all the files necessary to install a
module and can be installed with pip with all dependencies automatically
installed too.

To create wheel of cssfinder_backend_numpy use `poe` task in terminal:

```
poe build
```

![poe_build](https://user-images.githubusercontent.com/56170852/223251363-61fc4d00-68ad-429c-9fbb-8ab7f4712451.png)

This will create `dist/` directory with `cssfinder_backend_numpy-0.5.0` or
alike inside.

Wheel file can be installed with

```
pip install ./dist/cssfinder_backend_numpy-0.5.0
```

What you expect is

```
Successfully installed cssfinder_backend_numpy-0.5.0
```

or rather something like

```
Successfully installed click-8.1.3 contourpy-1.0.7 cssfinder_backend_numpy-0.5.0 cycler-0.11.0 dnspython-2.3.0 email-validator-1.3.1 fonttools-4.39.0 idna-3.4 jsonref-1.1.0 kiwisolver-1.4.4 llvmlite-0.39.1 markdown-it-py-2.2.0 matplotlib-3.7.1 mdurl-0.1.2 numba-0.56.4 numpy-1.23.5 packaging-23.0 pandas-1.5.3 pendulum-2.1.2 pillow-9.4.0 pydantic-1.10.5 pygments-2.14.0 pyparsing-3.0.9 python-dateutil-2.8.2 pytz-2022.7.1 pytzdata-2020.1 rich-13.3.2 scipy-1.10.1 six-1.16.0 typing-extensions-4.5.0
```

But `cssfinder_backend_numpy-0.5.0` should be included in this list.

## Code quality

To ensure that all code follow same style guidelines and code quality rules,
multiple static analysis tools are used. For simplicity, all of them are
configured as `pre-commit` ([learn about pre-commit](https://pre-commit.com/))
hooks. Most of them however are listed as development dependencies.

- `autocopyright`: This hook automatically adds copyright headers to files. It
  is used to ensure that all files in the repository have a consistent
  copyright notice.

- `autoflake`: This hook automatically removes unused imports from Python code.
  It is used to help keep code clean and maintainable by removing unnecessary
  code.

- `docformatter`: This hook automatically formats docstrings in Python code. It
  is used to ensure that docstrings are consistent and easy to read.

- `prettier`: This hook automatically formats code in a variety of languages,
  including JavaScript, HTML, CSS, and Markdown. It is used to ensure that code
  is consistently formatted and easy to read.

- `isort`: This hook automatically sorts Python imports. It is used to ensure
  that imports are organized in a consistent and readable way.

- `black`: This hook automatically formats Python code. It is used to ensure
  that code is consistently formatted and easy to read.

- `check-merge-conflict`: This hook checks for merge conflicts. It is used to
  ensure that code changes do not conflict with other changes in the
  repository.

- `check-case-conflict`: This hook checks for case conflicts in file names. It
  is used to ensure that file names are consistent and do not cause issues on
  case-sensitive file systems.

- `trailing-whitespace`: This hook checks for trailing whitespace in files. It
  is used to ensure that files do not contain unnecessary whitespace.

- `end-of-file-fixer`: This hook adds a newline to the end of files if one is
  missing. It is used to ensure that files end with a newline character.

- `debug-statements`: This hook checks for the presence of debugging statements
  (e.g., print statements) in code. It is used to ensure that code changes do
  not contain unnecessary debugging code.

- `check-added-large-files`: This hook checks for large files that have been
  added to the repository. It is used to ensure that large files are not
  accidentally committed to the repository.

- `check-toml`: This hook checks for syntax errors in TOML files. It is used to
  ensure that TOML files are well-formed.

- `mixed-line-ending`: This hook checks for mixed line endings (e.g., a mix of
  Windows and Unix line endings) in text files. It is used to ensure that text
  files have consistent line endings.

To run all checks, you must install hooks first with poe

```
poe install-hooks
```

After you have once used this command, you wont have to use it in this
environment. Then you can use

```
poe run-hooks
```

To run checks and automatic fixing. Not all issues can be automatically fixed,
some of them will require your intervention.

Successful hooks run should leave no Failed tasks:

![run_hooks_output](https://user-images.githubusercontent.com/56170852/223247968-8333e9ee-c0f0-4cce-afe1-a8e7917d9b0a.png)

Example of failed task:

![failed_task](https://user-images.githubusercontent.com/56170852/223249222-113a1269-fb3c-4d2c-b2ba-3d26e8ac090a.png)

Those hooks will be run also while you try to commit anything. If any tasks
fails, no commit will be created, instead you will be expected to fix errors
and add stage fixes. Then you may retry running `git commit`.
