# Copyright (c) 2022-2023 The pymovements Project Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Test pymovements base classes.
"""
import polars as pl
import pytest

from pymovements.base import DataFrame


def test_dataframe_is_polars_dataframe():
    assert issubclass(DataFrame, pl.DataFrame)


def test_dataframe_does_not_override_method():
    expected_msg_substrings = (
        'InvalidDataFrame', 'must not override', 'DataFrame', 'method', 'apply',
    )

    with pytest.raises(Exception) as excinfo:
        class InvalidDataFrame(DataFrame):  # pylint: disable=unused-variable
            def apply(self):  # pylint: disable=arguments-differ
                pass

    msg, = excinfo.value.args
    for msg_substring in expected_msg_substrings:
        assert msg_substring in msg


def test_dataframe_does_not_override_attribute():
    expected_msg_substrings = (
        'InvalidDataFrame', 'must not override', 'DataFrame', 'attribute', '_accessors',
    )

    with pytest.raises(Exception) as excinfo:
        class InvalidDataFrame(DataFrame):  # pylint: disable=unused-variable
            _accessors = None

    msg, = excinfo.value.args
    for msg_substring in expected_msg_substrings:
        assert msg_substring in msg
