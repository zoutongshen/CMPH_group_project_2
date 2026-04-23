"""
Post-processing utilities for the 2D XY model Monte Carlo simulation.

All routines here take plain numpy time series as input and return scalars
or arrays. They are independent of xy_simulation.py so that the analysis
can be rerun on stored data without repeating the Monte Carlo.

The definitions follow the lecture 8 notes:

  - autocorrelation chi(t) of a time series m(t')
        chi(t) = <m(t') m(t'+t)> - <m(t')><m(t'+t)>
      with matched-subset means so that chi(0) = var(m).

  - correlation time
        tau = sum_{t=0}^{t*} chi(t) / chi(0),
      truncated at the first t* where chi(t) turns negative.

  - standard deviation of the mean for a correlated time series
        sigma = sqrt( 2 tau / N_samples * var(m) ).

  - per-spin susceptibility and specific heat from fluctuations
        chi_M = beta * n_sites * var(m_per_spin)
        C     = n_sites / T^2 * var(e_per_spin).

  - blocking estimate of the error on any derived quantity by computing it
      on successive blocks and taking the standard error of the block values.

Author: CMPH 2026 Project 2
Date: April 2026
"""

from typing import Callable, Optional, Tuple

import numpy as np


def autocorrelation(series: np.ndarray,
                    max_lag: Optional[int] = None) -> np.ndarray:
    """
    Time-displaced autocorrelation of a scalar time series.

    For each lag t the head (series[: N - t]) and tail (series[t :]) are
    averaged independently, so that the returned value equals
    <m(t') m(t'+t)> - <m(t')><m(t'+t)> evaluated on the same index subset.
    This matches eq. 2 of the lecture 8 notes and makes chi(0) equal to the
    sample variance.

    Args:
        series: 1D array of equilibrated measurements, spaced by one sweep
        max_lag: Largest lag to evaluate. Defaults to N // 4, which is
                 comfortably longer than the correlation time for this system
                 except very close to T_c.

    Returns:
        Array chi of shape (max_lag + 1,) with chi[t] the autocorrelation
        at lag t sweeps.
    """
    n_samples = len(series)
    if max_lag is None:
        max_lag = n_samples // 4
    max_lag = min(max_lag, n_samples - 1)

    chi = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        head = series[: n_samples - lag]
        tail = series[lag:]
        chi[lag] = (head * tail).mean() - head.mean() * tail.mean()
    return chi


def correlation_time(series: np.ndarray,
                     max_lag: Optional[int] = None) -> float:
    """
    Integrated correlation time tau of a time series.

    Sums chi(t) / chi(0) from t = 0 up to (and not including) the first lag
    at which chi(t) turns negative, as recommended in the lecture notes to
    avoid the noisy large-lag tail.

    Args:
        series: 1D array of equilibrated measurements
        max_lag: Passed through to autocorrelation()

    Returns:
        tau in units of sweeps. Returns 0 for a constant series.
    """
    chi = autocorrelation(series, max_lag=max_lag)
    if chi[0] <= 0.0:
        return 0.0
    normalised = chi / chi[0]
    negative = np.where(normalised < 0.0)[0]
    cutoff = negative[0] if negative.size else len(normalised)
    return float(normalised[:cutoff].sum())


def mean_with_error(series: np.ndarray,
                    tau: float) -> Tuple[float, float]:
    """
    Mean of a correlated time series and the standard deviation of the mean.

    Uses sigma = sqrt( 2 tau / N_samples * var(m) ), which corrects the naive
    standard error by accounting for the fact that only N_samples / (2 tau)
    of the draws are effectively independent. tau is floored at 0.5 so that
    a series whose autocorrelation has already decayed below chi(0) at lag 1
    (tau from correlation_time() then equals 1, or 0 for a constant series)
    still reports the uncorrelated standard error sqrt(var / N_samples).

    Args:
        series: 1D array of equilibrated measurements
        tau: Correlation time in sweeps (same unit as the spacing of series)

    Returns:
        (mean, sigma) where sigma is the standard error of the mean.
    """
    n_samples = len(series)
    mean = float(series.mean())
    variance = float(series.var())
    effective_tau = max(tau, 0.5)
    sigma = float(np.sqrt(2.0 * effective_tau / n_samples * variance))
    return mean, sigma


def susceptibility(magnetization_per_spin: np.ndarray,
                   beta: float, n_sites: int) -> float:
    """
    Magnetic susceptibility per spin from a per-spin magnetization series.

    Implements chi_M = (beta / N^2) * (<M^2> - <M>^2) with M = n_sites * m.
    Expanding gives chi_M = beta * n_sites * var(m_per_spin).

    Args:
        magnetization_per_spin: Time series of |M| / N^2
        beta: Inverse temperature 1 / (k_B T)
        n_sites: Number of lattice sites N^2

    Returns:
        chi_M at the corresponding temperature.
    """
    return beta * n_sites * float(np.var(magnetization_per_spin))


def specific_heat(energy_per_spin: np.ndarray,
                  temperature: float, n_sites: int) -> float:
    """
    Specific heat per spin from a per-spin energy series.

    Implements C = 1 / (N^2 T^2) * (<E^2> - <E>^2) with E = n_sites * e.
    Expanding gives C = n_sites * var(e_per_spin) / T^2.

    Args:
        energy_per_spin: Time series of E / N^2
        temperature: Temperature T in the same units as the simulation
        n_sites: Number of lattice sites N^2

    Returns:
        C at the corresponding temperature.
    """
    return n_sites * float(np.var(energy_per_spin)) / (temperature ** 2)


def block_error(series: np.ndarray,
                estimator: Callable[[np.ndarray], float],
                block_length: int) -> Tuple[float, float]:
    """
    Blocking estimate of the mean and error of a derived quantity.

    Splits the series into non-overlapping blocks of length block_length,
    evaluates estimator() on each block, and returns the mean and standard
    error of the block values. As discussed in the lecture notes, the block
    length should be much larger than tau (a reasonable choice is 16 tau)
    so that blocks are approximately independent.

    Args:
        series: Equilibrated time series
        estimator: Callable that maps a 1D array to a scalar (e.g. a lambda
                   wrapping susceptibility() for a fixed (beta, n_sites))
        block_length: Length of each block, in sweeps

    Returns:
        (mean_over_blocks, standard_error_of_mean).
    """
    if block_length < 1:
        raise ValueError("block_length must be a positive integer")
    n_blocks = len(series) // block_length
    if n_blocks < 2:
        raise ValueError(
            f"Need at least 2 blocks to estimate an error "
            f"(got n_blocks={n_blocks} for block_length={block_length} "
            f"and series length {len(series)})."
        )
    block_values = np.array([
        estimator(series[i * block_length:(i + 1) * block_length])
        for i in range(n_blocks)
    ])
    mean = float(block_values.mean())
    standard_error = float(block_values.std(ddof=1) / np.sqrt(n_blocks))
    return mean, standard_error


def scan_statistics(magnetization_series: np.ndarray,
                    energy_series: np.ndarray,
                    temperatures: np.ndarray,
                    n_sites: int,
                    *, block_factor: int = 16) -> dict:
    """
    Compute all lecture 8 quantities for a temperature scan in one pass.

    For each temperature, measures tau from the |M|/N^2 series, computes
    <|m|> and <e> with the correlated standard error, and uses blocking
    with block length block_factor * tau to estimate chi_M, C and their
    errors. Temperatures at which there are fewer than two blocks are
    returned with NaN error bars.

    Args:
        magnetization_series: Shape (n_temperatures, n_samples) of |M| / N^2
        energy_series:        Shape (n_temperatures, n_samples) of E / N^2
        temperatures:         Shape (n_temperatures,) of temperature values
        n_sites:              Number of lattice sites N^2
        block_factor:         Block length as multiples of tau (default 16)

    Returns:
        Dict with arrays of shape (n_temperatures,):
            'tau',
            'magnetization_mean', 'magnetization_error',
            'energy_mean', 'energy_error',
            'susceptibility', 'susceptibility_error',
            'specific_heat', 'specific_heat_error'.
    """
    n_temperatures = len(temperatures)
    tau = np.zeros(n_temperatures)
    magnetization_mean = np.zeros(n_temperatures)
    magnetization_error = np.zeros(n_temperatures)
    energy_mean = np.zeros(n_temperatures)
    energy_error = np.zeros(n_temperatures)
    susceptibility_values = np.zeros(n_temperatures)
    susceptibility_errors = np.full(n_temperatures, np.nan)
    specific_heat_values = np.zeros(n_temperatures)
    specific_heat_errors = np.full(n_temperatures, np.nan)

    for i, temperature in enumerate(temperatures):
        mag_series = magnetization_series[i]
        e_series = energy_series[i]
        beta = 1.0 / temperature

        tau_i = correlation_time(mag_series)
        tau[i] = tau_i

        magnetization_mean[i], magnetization_error[i] = mean_with_error(mag_series, tau_i)
        energy_mean[i], energy_error[i] = mean_with_error(e_series, tau_i)

        susceptibility_values[i] = susceptibility(mag_series, beta, n_sites)
        specific_heat_values[i] = specific_heat(e_series, temperature, n_sites)

        block_length = max(1, int(round(block_factor * tau_i)))
        n_blocks = len(mag_series) // block_length
        if n_blocks >= 2:
            _, susceptibility_errors[i] = block_error(
                mag_series,
                lambda block, b=beta: susceptibility(block, b, n_sites),
                block_length,
            )
            _, specific_heat_errors[i] = block_error(
                e_series,
                lambda block, t=temperature: specific_heat(block, t, n_sites),
                block_length,
            )

    return {
        'tau': tau,
        'magnetization_mean': magnetization_mean,
        'magnetization_error': magnetization_error,
        'energy_mean': energy_mean,
        'energy_error': energy_error,
        'susceptibility': susceptibility_values,
        'susceptibility_error': susceptibility_errors,
        'specific_heat': specific_heat_values,
        'specific_heat_error': specific_heat_errors,
    }
