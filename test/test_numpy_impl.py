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

import numpy as np

from cssfinder_backend_numpy.numpy import _complex64, _complex128


class TestNumPyImplementationF32(NumPyBaseSuite):
    """Test suite of raw numpy single precision implementation."""

    impl = _complex64  # type: ignore[assignment]
    primary_t = np.complex64
    secondary_t = np.float32


class TestNumPyImplementationF64(NumPyBaseSuite):
    """Test suite of raw numpy double precision implementation."""

    impl = _complex128  # type: ignore[assignment]
    primary_t = np.complex128
    secondary_t = np.float64
