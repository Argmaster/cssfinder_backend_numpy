# Changelog

NOTE: cssfinder_backend_numpy follows the [semver](https://semver.org/)
versioning standard.

### 0.4.0 - 27 Apr 2023

- Bump cssfinder from 0.4.0 to 0.5.0 (#9)
- Bump ruff from 0.0.257 to 0.0.262 (#17)
- Bump ipykernel from 6.21.3 to 6.22.0 (#5)
- Bump poetry from 1.4.0 to 1.4.2 (#14)
- Bump pre-commit from 3.2.0 to 3.2.2 (#15)
- Ignore typing issue with numba imports
- Fix apply_symmetries for jitted backends
- Fix indexing in random_d_fs
- Simplify loop in optimize_d_fs
- Bump pre-commit from 3.1.1 to 3.2.0 (#6)

### 0.3.0 - 21 Mar 2023

- Loosen version requirement on CSSFinder.

### 0.2.0 - 20 Mar 2023

- Fix Cython backend on Windows.

### 0.1.1 - 20 Mar 2023

- Fix CI for uploading wheels to PyPI.

### 0.1.0 - 20 Mar 2023

- Added NumPy based backend.
- Added NumPy based JITed backend.
- Added NumPy based cythonized backend.
- Added NumPy based debug backend.
