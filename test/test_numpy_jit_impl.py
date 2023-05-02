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


"""Test suite of raw numpy implementation of gilbert algorithm."""

from __future__ import annotations

from test.base_suite import NumPyBaseSuite
from typing import TypeVar

import numpy as np
import pytest

from cssfinder_backend_numpy.numpy_jit import _complex64, _complex128

PRIMARY = TypeVar("PRIMARY", np.complex128, np.complex64)
SECONDARY_co = TypeVar("SECONDARY_co", np.float64, np.float32, covariant=True)


class NumPyBaseSuiteJit(NumPyBaseSuite[PRIMARY, SECONDARY_co]):
    """Implementation of Gilbert algorithm using python numpy with JIT."""

    def test_normalize_zero_vector(self) -> None:
        """Normalizing a zero vector, which should raise a ZeroDivisionError."""
        zero_vector = np.zeros(3, dtype=self.primary_t)
        with pytest.raises(ZeroDivisionError):
            print(self.impl.normalize(zero_vector))


class TestNumPyJitImplementationF32(NumPyBaseSuiteJit):
    """Test suite of jit-ed numpy single precision implementation."""

    impl = _complex64  # type: ignore[assignment]
    primary_t = np.complex64
    secondary_t = np.float32


class TestNumPyJitImplementationF64(NumPyBaseSuiteJit):
    """Test suite of jit-ed numpy double precision implementation."""

    impl = _complex128  # type: ignore[assignment]
    primary_t = np.complex128
    secondary_t = np.float64
