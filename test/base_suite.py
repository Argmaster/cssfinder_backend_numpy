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


"""Module contains base class for creating test suites of implementations of Gilbert
algorithm.
"""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np
import pytest

if TYPE_CHECKING:
    import numpy.typing as npt

    from cssfinder_backend_numpy.impl import Implementation

PRIMARY = TypeVar("PRIMARY", np.complex128, np.complex64)
SECONDARY_co = TypeVar("SECONDARY_co", np.float64, np.float32, covariant=True)


class NumPyBaseSuite(Generic[PRIMARY, SECONDARY_co]):
    """Implementation of Gilbert algorithm using python numpy library."""

    impl: Implementation[PRIMARY, SECONDARY_co]
    primary_t: type[PRIMARY]
    secondary_t: type[SECONDARY_co]

    @pytest.fixture()
    def matrix1(self) -> npt.NDArray[PRIMARY]:
        """First example matrix."""
        return np.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]], dtype=self.primary_t)

    @pytest.fixture()
    def matrix2(self) -> npt.NDArray[PRIMARY]:
        """Second example matrix."""
        return np.array([[5 + 5j, 6 + 6j], [7 + 7j, 8 + 8j]], dtype=self.primary_t)

    def test_product_same_matrices(self, matrix1: npt.NDArray[PRIMARY]) -> None:
        """The product of the same matrix."""
        assert self.impl.product(matrix1, matrix1) == pytest.approx(0.0)

    def test_product_different_matrices(
        self, matrix1: npt.NDArray[PRIMARY], matrix2: npt.NDArray[PRIMARY]
    ) -> None:
        """The product of two different matrices."""
        assert self.impl.product(matrix1, matrix2) == pytest.approx(0.0)

    def test_product_identity_matrices(self) -> None:
        """The product of two identity matrices."""
        identity = np.eye(2, dtype=self.primary_t)
        assert self.impl.product(identity, identity) == pytest.approx(2.0)

    def test_product_zero_matrix(self, matrix1: npt.NDArray[PRIMARY]) -> None:
        """The product of a matrix and a zero matrix."""
        zero_matrix = np.zeros((2, 2), dtype=self.primary_t)
        assert self.impl.product(matrix1, zero_matrix) == pytest.approx(0.0)

    def test_product_conjugate_transpose(self, matrix1: npt.NDArray[PRIMARY]) -> None:
        """The product of a matrix and its conjugate transpose."""
        assert self.impl.product(matrix1, matrix1.conj().T) == pytest.approx(60.0)

    @pytest.fixture()
    def vector(self) -> npt.NDArray[PRIMARY]:
        """Generate example vector."""
        return np.array([1 + 1j, 2 + 2j, 3 + 3j], dtype=self.primary_t)

    def test_normalize_single_element_vector(self) -> None:
        """Normalize a single-element vector."""
        vector = np.array([1 + 1j], dtype=self.primary_t)
        normalized_vector = self.impl.normalize(vector)
        assert np.linalg.norm(normalized_vector) == pytest.approx(1.0)

    def test_normalize_complex_vector(self, vector: npt.NDArray[PRIMARY]) -> None:
        """Normalize a complex vector."""
        normalized_vector = self.impl.normalize(vector)
        assert np.linalg.norm(normalized_vector) == pytest.approx(1.0)

    def test_normalize_real_vector(self) -> None:
        """Normalize a real vector."""
        real_vector = np.array([1, 2, 3], dtype=self.primary_t)
        normalized_vector = self.impl.normalize(real_vector)
        assert np.linalg.norm(normalized_vector) == pytest.approx(1.0)

    def test_normalize_imaginary_vector(self) -> None:
        """Normalize an imaginary vector."""
        imaginary_vector = np.array([1j, 2j, 3j], dtype=self.primary_t)
        normalized_vector = self.impl.normalize(imaginary_vector)
        assert np.linalg.norm(normalized_vector) == pytest.approx(1.0)

    def test_normalize_zero_vector(self) -> None:
        """Normalizing a zero vector, which should warn about 0 division."""
        zero_vector = np.zeros(3, dtype=self.primary_t)
        with pytest.warns(RuntimeWarning):
            print(self.impl.normalize(zero_vector))

    def test_project_single_element_vector(self) -> None:
        """Projecting a single-element vector."""
        vector = np.array([1 + 1j], dtype=self.primary_t)
        projection = self.impl.project(vector)
        assert projection.shape == (1, 1)
        assert projection[0, 0] == pytest.approx(2.0)

    def test_project_complex_vector(self, vector: npt.NDArray[PRIMARY]) -> None:
        """Projecting a complex vector."""
        projection = self.impl.project(vector)
        assert projection.shape == (3, 3)

    def test_project_real_vector(self) -> None:
        """Projecting a real vector."""
        real_vector = np.array([1, 2, 3], dtype=self.primary_t)
        projection = self.impl.project(real_vector)
        assert projection.shape == (3, 3)

    def test_project_imaginary_vector(self) -> None:
        """Projecting an imaginary vector."""
        imaginary_vector = np.array([1j, 2j, 3j], dtype=self.primary_t)
        projection = self.impl.project(imaginary_vector)
        assert projection.shape == (3, 3)

    def test_project_zero_vector(self) -> None:
        """Projecting a zero vector, which should result in a zero matrix."""
        zero_vector = np.zeros(3, dtype=self.primary_t)
        projection = self.impl.project(zero_vector)
        assert projection.shape == (3, 3)
        assert np.allclose(projection, np.zeros((3, 3)))

    @pytest.fixture()
    def operator(self) -> npt.NDArray[PRIMARY]:
        """Create example operator."""
        return np.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]], dtype=self.primary_t)

    @pytest.fixture()
    def unitary(self) -> npt.NDArray[PRIMARY]:
        """Create example unitary."""
        return np.array([[1, 0], [0, 1]], dtype=self.primary_t)

    def test_rotate_identity_operator(
        self, operator: npt.NDArray[PRIMARY], unitary: npt.NDArray[PRIMARY]
    ) -> None:
        """Rotating an operator with an identity unitary matrix (should return the same
        operator).
        """
        rotated = self.impl.rotate(operator, unitary)
        assert np.allclose(operator, rotated)

    def test_rotate_zero_operator(self, unitary: npt.NDArray[PRIMARY]) -> None:
        """Rotating a zero operator (should return a zero operator)."""
        zero_operator = np.zeros((2, 2), dtype=self.primary_t)
        rotated = self.impl.rotate(zero_operator, unitary)
        assert np.allclose(zero_operator, rotated)

    def test_rotate_with_hadamard(self, operator: npt.NDArray[PRIMARY]) -> None:
        """Rotating an operator with a Hadamard unitary matrix."""
        hadamard = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=self.primary_t)
        rotated = self.impl.rotate(operator, hadamard)
        if self.primary_t == np.complex128:
            expected = np.array(
                [
                    [5.00000000e00 + 5.00000000e00j, -1.00000000e00 - 1.00000000e00j],
                    [-2.00000000e00 - 2.00000000e00j, 0 - 0j],
                ],
                dtype=self.primary_t,
            )
        else:
            pytest.skip()

        assert np.allclose(rotated, expected)

    def test_rotate_with_pauli_x(self, operator: npt.NDArray[PRIMARY]) -> None:
        """Rotating an operator with a Pauli-X unitary matrix."""
        pauli_x = np.array([[0, 1], [1, 0]], dtype=self.primary_t)
        rotated = self.impl.rotate(operator, pauli_x)
        expected = np.array([[4 + 4j, 3 + 3j], [2 + 2j, 1 + 1j]], dtype=self.primary_t)
        assert np.allclose(rotated, expected)

    def test_rotate_with_pauli_y(self, operator: npt.NDArray[PRIMARY]) -> None:
        """Rotating an operator with a Pauli-Y unitary matrix."""
        pauli_y = np.array([[0, -1j], [1j, 0]], dtype=self.primary_t)
        rotated = self.impl.rotate(operator, pauli_y)
        expected = np.array(
            [[4.0 + 4.0j, -3.0 - 3.0j], [-2.0 - 2.0j, 1.0 + 1.0j]], dtype=self.primary_t
        )
        print(rotated)
        assert np.allclose(rotated, expected)

    def test_kronecker_identity_matrices(self) -> None:
        """Kronecker product of two 2x2 identity matrices (should return the same 2x2
        identity matrix).
        """
        identity2 = np.eye(2, dtype=self.primary_t)
        kronecker_product = self.impl.kronecker(identity2, identity2)
        expected = np.array(
            [
                [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
            ],
            dtype=self.primary_t,
        )
        assert np.allclose(kronecker_product, expected)

    def test_kronecker_single_element_matrices(self) -> None:
        """Kronecker product of two single-element matrices."""
        matrix1 = np.array([[1 + 1j]], dtype=self.primary_t)
        matrix2 = np.array([[2 + 2j]], dtype=self.primary_t)
        kronecker_product = self.impl.kronecker(matrix1, matrix2)
        expected = np.array([[0.0 + 4.0j]], dtype=self.primary_t)
        assert np.allclose(kronecker_product, expected)

    def test_kronecker_same_matrices(self, matrix1: npt.NDArray[PRIMARY]) -> None:
        """Kronecker product of the same 2x2 matrices."""
        kronecker_product = self.impl.kronecker(matrix1, matrix1)
        expected = np.array(
            [
                [
                    [0.0 + 2.0j, 0.0 + 4.0j, 0.0 + 4.0j, 0.0 + 8.0j],
                    [0.0 + 6.0j, 0.0 + 8.0j, 0.0 + 12.0j, 0.0 + 16.0j],
                    [0.0 + 6.0j, 0.0 + 12.0j, 0.0 + 8.0j, 0.0 + 16.0j],
                    [0.0 + 18.0j, 0.0 + 24.0j, 0.0 + 24.0j, 0.0 + 32.0j],
                ]
            ],
            dtype=self.primary_t,
        )
        assert np.allclose(kronecker_product, expected)

    def test_kronecker_different_matrices(
        self, matrix1: npt.NDArray[PRIMARY], matrix2: npt.NDArray[PRIMARY]
    ) -> None:
        """Kronecker product of two different 2x2 matrices."""
        kronecker_product = self.impl.kronecker(matrix1, matrix2)
        expected = np.array(
            [
                [
                    [0.0 + 10.0j, 0.0 + 12.0j, 0.0 + 20.0j, 0.0 + 24.0j],
                    [0.0 + 14.0j, 0.0 + 16.0j, 0.0 + 28.0j, 0.0 + 32.0j],
                    [0.0 + 30.0j, 0.0 + 36.0j, 0.0 + 40.0j, 0.0 + 48.0j],
                    [0.0 + 42.0j, 0.0 + 48.0j, 0.0 + 56.0j, 0.0 + 64.0j],
                ]
            ],
            dtype=self.primary_t,
        )
        assert np.allclose(kronecker_product, expected)

    def test_kronecker_zero_matrix(self, matrix1: npt.NDArray[PRIMARY]) -> None:
        """Kronecker product of a 2x2 matrix and a 2x2 zero matrix (should return a 4x4
        zero matrix).
        """
        zero_matrix = np.zeros((2, 2), dtype=self.primary_t)
        kronecker_product = self.impl.kronecker(matrix1, zero_matrix)
        expected = np.zeros((4, 4), dtype=self.primary_t)
        assert np.allclose(kronecker_product, expected)


with suppress(AttributeError):
    del NumPyBaseSuite.__new__
