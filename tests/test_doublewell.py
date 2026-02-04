"""Tests for the doublewell module."""

import pytest
import numpy as np
from ch_timedisc.doublewell.doublewell import DoubleWell


class TestDoubleWell:
    """Test cases for the DoubleWell class."""

    def test_call_equals_c_minus_e(self):
        """Test that __call__ equals c - e."""
        dw = DoubleWell(scaling=1.0)

        # Test with various values
        test_values = np.linspace(0, 1, 11)

        for pf in test_values:
            call_result = dw(pf)
            c_minus_e = dw.c(pf) - dw.e(pf)

            assert np.isclose(
                call_result, c_minus_e
            ), f"Mismatch at pf={pf}: __call__={call_result}, c-e={c_minus_e}"

    def test_call_equals_c_minus_e_with_scaling(self):
        """Test that __call__ equals c - e with different scaling factors."""
        test_scalings = [0.5, 1.0, 2.0, 10.0]
        test_values = np.linspace(0, 1, 11)

        for scaling in test_scalings:
            dw = DoubleWell(scaling=scaling)

            for pf in test_values:
                call_result = dw(pf)
                c_minus_e = dw.c(pf) - dw.e(pf)

                assert np.isclose(
                    call_result, c_minus_e
                ), f"Mismatch at scaling={scaling}, pf={pf}: __call__={call_result}, c-e={c_minus_e}"

    def test_cprime_minus_eprime_equals_prime(self):
        """Test that cprime - eprime equals prime."""
        dw = DoubleWell(scaling=1.0)

        # Test with various values
        test_values = np.linspace(0, 1, 11)

        for pf in test_values:
            prime_result = dw.prime(pf)
            cprime_minus_eprime = dw.cprime(pf) - dw.eprime(pf)

            assert np.isclose(
                prime_result, cprime_minus_eprime
            ), f"Mismatch at pf={pf}: prime={prime_result}, cprime-eprime={cprime_minus_eprime}"

    def test_cprime_minus_eprime_equals_prime_with_scaling(self):
        """Test that cprime - eprime equals prime with different scaling factors."""
        test_scalings = [0.5, 1.0, 2.0, 10.0]
        test_values = np.linspace(0, 1, 11)

        for scaling in test_scalings:
            dw = DoubleWell(scaling=scaling)

            for pf in test_values:
                prime_result = dw.prime(pf)
                cprime_minus_eprime = dw.cprime(pf) - dw.eprime(pf)

                assert np.isclose(
                    prime_result, cprime_minus_eprime
                ), f"Mismatch at scaling={scaling}, pf={pf}: prime={prime_result}, cprime-eprime={cprime_minus_eprime}"
