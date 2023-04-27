# Copyright 2023 Krzysztof Wiśniewski <argmaster.world@gmail.com>
#
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the “Software”), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# ------------------------------------------------------------------------------------ #
#                                                                                      #
#   THIS FILE WAS AUTOMATICALLY GENERATED FROM TEMPLATE. DO NOT MODIFY.                #
#                                                                                      #
#   To modify this file, modify `scripts/templates/numpy.pyjinja2` and                 #
#   use `poe gen-numpy-impl` to generate python files.                                 #
#                                                                                      #
# ------------------------------------------------------------------------------------ #
#
"""Module contains implementation of backend operations in numpy.

Spec
----

- Floating precision:   np.float32
- Complex precision:    np.complex64

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


from typing import Any

#   ██████     ███████    ██████     ██    ██     ██████
#   ██   ██    ██         ██   ██    ██    ██    ██
#   ██   ██    █████      ██████     ██    ██    ██   ███
#   ██   ██    ██         ██   ██    ██    ██    ██    ██
#   ██████     ███████    ██████      ██████      ██████


def assert_dtype(ob: Any, dtype: Any) -> None:
    if ob.dtype != dtype:
        raise AssertionError(ob.dtype)


#    █████  ██████  ███    ███ ███    ███  ██████  ███    ██
#   ██     ██    ██ ████  ████ ████  ████ ██    ██ ████   ██
#   ██     ██    ██ ██ ████ ██ ██ ████ ██ ██    ██ ██ ██  ██
#   ██     ██    ██ ██  ██  ██ ██  ██  ██ ██    ██ ██  ██ ██
#    █████  ██████  ██      ██ ██      ██  ██████  ██   ████


_REAL = np.cos(0.01 * np.pi)
_IMAG = 1j * np.sin(0.01 * np.pi)
_VALUE = (_REAL + _IMAG - 1).astype(np.complex64)


def product(
    matrix1: npt.NDArray[np.complex64], matrix2: npt.NDArray[np.complex64]
) -> np.float32:
    """Calculate scalar product of two matrices."""
    assert_dtype(matrix1, np.complex64)
    assert_dtype(matrix2, np.complex64)

    retval = np.trace(np.dot(matrix1, matrix2)).real
    assert_dtype(retval, np.float32)

    return retval  # type: ignore[no-any-return]


def get_random_haar_1d(depth: int) -> npt.NDArray[np.complex64]:
    """Generate a random vector with Haar measure."""
    real = np.random.uniform(0, 1, depth)  # noqa: NPY002
    imag = np.random.uniform(0, 1, depth)  # noqa: NPY002

    retval = np.exp(2 * np.pi * 1j * real) * np.sqrt(-np.log(imag))

    retval = (retval).astype(np.complex64)

    assert_dtype(retval, np.complex64)

    return retval  # type: ignore[no-any-return]


def get_random_haar_2d(depth: int, quantity: int) -> npt.NDArray[np.complex64]:
    """Generate multiple random vectors with Haar measure in form of matrix."""
    real = np.random.uniform(0, 1, (quantity, depth))  # noqa: NPY002
    imag = np.random.uniform(0, 1, (quantity, depth))  # noqa: NPY002

    retval = np.exp(2 * np.pi * 1j * real) * np.sqrt(-np.log(imag))

    retval = (retval).astype(np.complex64)

    assert_dtype(retval, np.complex64)

    return retval  # type: ignore[no-any-return]


def normalize(mtx: npt.NDArray[np.complex64]) -> npt.NDArray[np.complex64]:
    """Normalize a vector."""
    assert_dtype(mtx, np.complex64)

    mtx2 = np.dot(mtx, np.conj(mtx))
    assert_dtype(mtx2, np.complex64)

    val = np.sqrt(np.real(mtx2))
    assert_dtype(val, np.float32)

    retval = mtx / val
    assert_dtype(retval, np.complex64)

    return retval  # type: ignore[no-any-return]


def project(mtx1: npt.NDArray[np.complex64]) -> npt.NDArray[np.complex64]:
    """Build a projection from a vector."""
    assert_dtype(mtx1, np.complex64)

    retval = np.outer(mtx1, np.conj(mtx1))
    assert_dtype(retval, np.complex64)

    return retval  # type: ignore[no-any-return]


def kronecker(
    mtx: npt.NDArray[np.complex64], mtx1: npt.NDArray[np.complex64]
) -> npt.NDArray[np.complex64]:
    """Kronecker Product."""
    assert_dtype(mtx, np.complex64)
    assert_dtype(mtx1, np.complex64)

    ddd1 = len(mtx)
    ddd2 = len(mtx1)

    output_shape = (ddd1 * ddd2, ddd1 * ddd2)

    dot_0_1 = np.tensordot(mtx, mtx1, 0)
    assert_dtype(dot_0_1, np.complex64)

    out_mtx = np.swapaxes(dot_0_1, 1, 2)
    assert_dtype(out_mtx, np.complex64)

    retval = out_mtx.reshape(output_shape).astype(np.complex64, copy=False)
    assert_dtype(retval, np.complex64)

    return retval  # type: ignore[no-any-return]


def rotate(
    rho2: npt.NDArray[np.complex64], unitary: npt.NDArray[np.complex64]
) -> npt.NDArray[np.complex64]:
    """Sandwich an operator with a unitary."""
    assert_dtype(rho2, np.complex64)
    assert_dtype(unitary, np.complex64)

    rho2a = np.dot(rho2, np.conj(unitary).T)  # matmul replaced with dot
    assert_dtype(rho2a, np.complex64)

    rho2a = np.dot(unitary, rho2a)  # matmul replaced with dot
    assert_dtype(rho2a, np.complex64)

    return rho2a  # type: ignore[no-any-return]


def apply_symmetries(
    rho: npt.NDArray[np.complex64],
    symmetries: list[list[npt.NDArray[np.complex64]]],
) -> npt.NDArray[np.complex64]:
    """Apply symmetries to density matrix.

    Parameters
    ----------
    rho : npt.NDArray[np.complex64]
        Density matrix to which we want to apply symmetries.
    symmetries : list[list[npt.NDArray[np.complex64]]]
        List of matrices representing the symmetries.

    Returns
    -------
    npt.NDArray[np.complex64]
        The result of applying the symmetries to the given density matrix.

    Notes
    -----
    The first input `rho` is modified by this function. If you don't want to modify the
    original array, make a copy before passing it to this function.

    This function calculates the trace of output density matrix and normalizes it before
    returning.

    """
    assert_dtype(rho, np.complex64)

    for row_ in symmetries:
        for sym_ in row_:
            assert_dtype(sym_, np.complex64)

    output = rho
    for row in symmetries:
        for sym in row:
            output += rotate(output, sym)

    output /= np.trace(output)
    return output


#   ██████     ███████    ███████            ███    ███     ██████     ██████     ███████   # noqa: E501
#   ██   ██    ██         ██                 ████  ████    ██    ██    ██   ██    ██        # noqa: E501
#   ██   ██    █████      ███████            ██ ████ ██    ██    ██    ██   ██    █████     # noqa: E501
#   ██   ██    ██              ██            ██  ██  ██    ██    ██    ██   ██    ██        # noqa: E501
#   ██████     ██         ███████            ██      ██     ██████     ██████     ███████   # noqa: E501


def optimize_d_fs(
    new_state: npt.NDArray[np.complex64],
    visibility_state: npt.NDArray[np.complex64],
    depth: int,
    quantity: int,
    updates_count: int,
) -> npt.NDArray[np.complex64]:
    """Optimize implementation for FSnQd mode."""
    assert_dtype(new_state, np.complex64)
    assert_dtype(visibility_state, np.complex64)

    product_2_3 = product(new_state, visibility_state)

    # To make sure rotated_2 is not unbound
    unitary = random_unitary_d_fs(depth, quantity, 0)
    assert_dtype(unitary, np.complex64)

    rotated_2 = rotate(new_state, unitary)

    for idx in range(updates_count):
        idx_mod = idx % int(quantity)
        unitary = random_unitary_d_fs(depth, quantity, idx_mod)
        assert_dtype(unitary, np.complex64)

        rotated_2 = rotate(new_state, unitary)
        assert_dtype(rotated_2, np.complex64)

        product_rot2_3 = product(rotated_2, visibility_state)

        if product_2_3 > product_rot2_3:
            unitary = unitary.conj().T
            rotated_2 = rotate(new_state, unitary)
            assert_dtype(rotated_2, np.complex64)

        while product_rot2_3 > product_2_3:
            product_2_3 = product_rot2_3
            rotated_2 = rotate(rotated_2, unitary)
            assert_dtype(rotated_2, np.complex64)

            product_rot2_3 = product(rotated_2, visibility_state)

    return rotated_2.astype(np.complex64, copy=False)  # type: ignore[no-any-return]


def random_unitary_d_fs(
    depth: int, quantity: int, idx: int
) -> npt.NDArray[np.complex64]:
    """N quDits."""
    value = _random_unitary_d_fs(depth)
    assert_dtype(value, np.complex64)

    mtx = expand_d_fs(value, depth, quantity, idx)
    assert_dtype(mtx, np.complex64)

    return mtx  # type: ignore[no-any-return]


def _random_unitary_d_fs(depth: int) -> npt.NDArray[np.complex64]:
    random_mtx = random_d_fs(depth, 1)
    assert_dtype(random_mtx, np.complex64)

    identity_mtx = np.identity(depth).astype(np.complex64)
    assert_dtype(identity_mtx, np.complex64)

    rand_mul = np.multiply(_VALUE, random_mtx)
    assert_dtype(rand_mul, np.complex64)

    value = np.add(rand_mul, identity_mtx)
    assert_dtype(value, np.complex64)
    return value  # type: ignore[no-any-return]


def random_d_fs(depth: int, quantity: int) -> npt.NDArray[np.complex64]:
    """Random n quDit state."""
    rand_vectors = get_random_haar_2d(depth, quantity)
    vector = normalize(rand_vectors[0])
    assert_dtype(vector, np.complex64)

    for i in range(1, quantity):
        idx_vector = normalize(rand_vectors[i])
        assert_dtype(idx_vector, np.complex64)

        vector = np.outer(vector, idx_vector).flatten()
        assert_dtype(vector, np.complex64)

    vector = project(vector)
    assert_dtype(vector, np.complex64)

    return vector  # type: ignore[no-any-return]


def expand_d_fs(
    value: npt.NDArray[np.complex64],
    depth: int,
    quantity: int,
    idx: int,
) -> npt.NDArray[np.complex64]:
    """Expand an operator to n quDits."""
    assert_dtype(value, np.complex64)

    depth_1 = int(depth**idx)
    identity_1 = np.identity(depth_1, dtype=np.complex64)
    assert_dtype(identity_1, np.complex64)

    depth_2 = int(depth ** (quantity - idx - 1))
    identity_2 = np.identity(depth_2, dtype=np.complex64)
    assert_dtype(identity_2, np.complex64)

    kronecker_1 = kronecker(identity_1, value)
    assert_dtype(kronecker_1, np.complex64)

    kronecker_2 = kronecker(kronecker_1, identity_2)
    assert_dtype(kronecker_2, np.complex64)

    return kronecker_2  # type: ignore[no-any-return]


#   ██████     ███████            ███    ███     ██████     ██████     ███████
#   ██   ██    ██                 ████  ████    ██    ██    ██   ██    ██
#   ██████     ███████            ██ ████ ██    ██    ██    ██   ██    █████
#   ██   ██         ██            ██  ██  ██    ██    ██    ██   ██    ██
#   ██████     ███████            ██      ██     ██████     ██████     ███████


def random_bs(depth: int, quantity: int) -> npt.NDArray[np.complex64]:
    """Draw random biseparable state."""
    random_vector_1 = normalize(get_random_haar_1d(depth))
    random_vector_2 = normalize(get_random_haar_1d(quantity))

    vector = np.outer(random_vector_1, random_vector_2).flatten()
    assert_dtype(vector, np.complex64)

    vector = project(vector)
    assert_dtype(vector, np.complex64)

    return vector  # type: ignore[no-any-return]


def random_unitary_bs(depth: int, quantity: int) -> npt.NDArray[np.complex64]:
    """Draw random unitary for biseparable state."""
    random_vector = normalize(get_random_haar_1d(depth))
    assert_dtype(random_vector, np.complex64)

    random_matrix = project(random_vector)
    assert_dtype(random_matrix, np.complex64)

    identity_depth = np.identity(depth).astype(np.complex64)
    assert_dtype(identity_depth, np.complex64)

    identity_quantity = np.identity(quantity).astype(np.complex64)
    assert_dtype(identity_quantity, np.complex64)

    unitary_biseparable = _VALUE * random_matrix + identity_depth
    assert_dtype(unitary_biseparable, np.complex64)

    retval = kronecker(unitary_biseparable, identity_quantity)
    assert_dtype(retval, np.complex64)

    return retval  # type: ignore[no-any-return]


def random_unitary_bs_reverse(depth: int, quantity: int) -> npt.NDArray[np.complex64]:
    """Draw random unitary for biseparable state."""
    random_vector = normalize(get_random_haar_1d(depth))
    assert_dtype(random_vector, np.complex64)

    random_matrix = project(random_vector)
    assert_dtype(random_matrix, np.complex64)

    identity_depth = np.identity(depth).astype(np.complex64)
    assert_dtype(identity_depth, np.complex64)

    identity_quantity = np.identity(quantity).astype(np.complex64)
    assert_dtype(identity_quantity, np.complex64)

    unitary_biseparable = _VALUE * random_matrix + identity_depth
    assert_dtype(unitary_biseparable, np.complex64)

    retval = kronecker(identity_quantity, unitary_biseparable)
    assert_dtype(retval, np.complex64)

    return retval  # type: ignore[no-any-return]


def optimize_bs(
    new_state: npt.NDArray[np.complex64],
    visibility_state: npt.NDArray[np.complex64],
    depth: int,
    quantity: int,
    updates_count: int,
) -> npt.NDArray[np.complex64]:
    """Run the minimization algorithm to optimize the biseparable state.

    Parameters
    ----------
    new_state : npt.NDArray[np.complex64]
        Randomly drawn state to be optimized.
    visibility_state : npt.NDArray[np.complex64]
        Visibility matrix.
    depth : int
        Depth of analyzed system.
    quantity : int
        Quantity of quDits in system.
    updates_count : int
        Number of optimizer iterations to execute.

    Returns
    -------
    npt.NDArray[np.complex64]
        Optimized state.

    """
    assert_dtype(new_state, np.complex64)
    assert_dtype(visibility_state, np.complex64)

    pp1 = product(new_state, visibility_state)

    return_state = new_state.copy()

    for index in range(updates_count):
        if index % 2:
            unitary = random_unitary_bs(depth, quantity)
        else:
            unitary = random_unitary_bs_reverse(depth, quantity)
        assert_dtype(unitary, np.complex64)

        return_state = rotate(new_state, unitary)
        assert_dtype(return_state, np.complex64)

        if pp1 > product(return_state, visibility_state):
            unitary = unitary.conj().T
            return_state = rotate(new_state, unitary)
            assert_dtype(return_state, np.complex64)

        pp2 = product(return_state, visibility_state)

        while pp2 > pp1:
            pp1 = pp2
            return_state = rotate(return_state, unitary)
            pp2 = product(return_state, visibility_state)
            assert_dtype(return_state, np.complex64)

        assert_dtype(return_state, np.complex64)

    assert_dtype(return_state, np.complex64)

    return return_state
