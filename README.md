# Project 2: Monte Carlo Simulation of the 2D XY Model

## Overview

Metropolis Monte Carlo simulation of the two-dimensional XY model on an
N × N square lattice with periodic boundary conditions. Each site carries a
planar spin `s_i = (cos theta_i, sin theta_i)`. The Hamiltonian, with
`J1 = k_B = 1`, is

```
H = -J1 sum_<i,j>  cos(theta_i - theta_j)
    -J2 sum_<<i,j>> cos(theta_i - theta_j)
```

where the second sum runs over the four next-nearest-neighbour diagonals
(the `extension` branch). The CLI flag `--j2-ratio` sets `J2 / J1`; at
`--j2-ratio 0` (the default) the model reduces to the nearest-neighbour
2D XY model from `main`. The code tracks the total energy and the
magnetization vector incrementally, so each accepted spin move costs
O(1) work.

## Files

- `xy_simulation.py` — simulation code (`XYModel` class, `temperature_scan`
  helper, CLI)
- `analysis.py` — post-processing routines: autocorrelation, correlation
  time τ, correlated-mean error σ = √(2τ/t_max · var), susceptibility χ_M,
  specific heat C, and the blocking-method error estimator
- `vortices.py` — plaquette winding-number based vortex detection
- `plot_results.py` — plotting utilities for time series, two-start
  comparisons, temperature scans, correlation time, vortex counts, and
  spin configurations
- `data/` — directory for simulation output (`.npz` files)
- `figures/` — directory for generated plots

## Environment

The shared virtual environment at
`CMPH_code/.venv` already contains `numpy`, `matplotlib`, and `tqdm`.
Activate it before running:

```bash
source ../.venv/bin/activate
```

## How to run

### Single run at one temperature

```bash
python xy_simulation.py --size 20 --temperature 1.0 \
    --n-equil-sweeps 1000 --n-prod-sweeps 5000
```

Saves `data/run_N20_T1.00.npz` with the sweep index, energy per spin,
magnetization per spin, and the final lattice configuration.

### Two starts at the same temperature (equilibration check)

```bash
python xy_simulation.py --two-starts --size 20 --temperature 1.0
```

Runs one ordered-start and one random-start simulation at the same
temperature and writes both time series to a single `.npz` file.

### Temperature scan

```bash
python xy_simulation.py --scan --size 50 \
    --t-min 0.5 --t-max 2.5 --t-step 0.2 \
    --n-equil-sweeps 2000 --n-prod-sweeps 30000
```

Runs an independent, equilibrated simulation at every temperature. For each
temperature it saves the **full per-sweep time series** of `|M|/N^2` and
`E/N^2`, plus the final lattice. The analysis pipeline then consumes that
file:

- correlation time τ(T) from the autocorrelation of `|M|/N^2`
- `<|m|>`, `<e>` with the correlated standard error σ = √(2τ/t_max · var)
- χ_M and C from the fluctuation formulas, with error bars from the
  blocking method (block length ≈ 16τ)

Long production runs are needed near T_c ≈ 0.881 because τ peaks there
(critical slowing down), and blocking requires many blocks each much longer
than τ.

### Plots

```bash
# time series from a single run
python plot_results.py --mode time-series \
    --input data/run_N20_T1.00.npz \
    --output figures/time_series_N20_T1.00.png

# two-start comparison
python plot_results.py --mode two-starts \
    --input data/two_starts_N20_T1.00.npz \
    --output figures/two_starts_N20_T1.00.png

# 2x2 panel of <|m|>, <e>, chi_M, C with correlated errors
python plot_results.py --mode scan \
    --input data/scan_N50.npz \
    --output figures/scan_N50.png

# correlation time tau(T); peaks near T_c
python plot_results.py --mode tau \
    --input data/scan_N50.npz \
    --output figures/tau_N50.png

# vortex / anti-vortex count of the final configuration vs T
python plot_results.py --mode vortex-count \
    --input data/scan_N50.npz \
    --output figures/vortex_count_N50.png

# final spin configurations at every T in the scan
python plot_results.py --mode scan-configurations \
    --input data/scan_N50.npz \
    --output figures/scan_configurations_N50.png

# single configuration (from any run file that stores final_angles)
python plot_results.py --mode configuration \
    --input data/run_N20_T1.00.npz \
    --output figures/configuration_N20_T1.00.png \
    --title "T = 1.0"

# same configuration with vortices / anti-vortices circled
python plot_results.py --mode configuration --overlay-vortices \
    --input data/run_N20_T1.00.npz \
    --output figures/configuration_vortices_N20_T1.00.png \
    --title "T = 1.0"

# arrows instead of a colour grid
python plot_results.py --mode vectors \
    --input data/run_N20_T1.00.npz \
    --output figures/vectors_N20_T1.00.png \
    --title "T = 1.0"
```

## Parameters

| Flag | Default | Meaning |
|---|---|---|
| `--size` | 20 | Lattice side length N (try 10, 20, 50) |
| `--temperature` | 1.0 | Temperature T in units of J/k_B |
| `--n-equil-sweeps` | 1000 | Equilibration sweeps before production |
| `--n-prod-sweeps` | 5000 | Production sweeps with recording |
| `--record-interval` | 1 | Record observables every this many sweeps |
| `--delta` | π | Maximum angle change per trial move |
| `--j2-ratio` | 0.0 | Ratio of NNN diagonal coupling to J1 (see Extension below) |
| `--seed` | 42 | Random seed |
| `--t-min` / `--t-max` / `--t-step` | 0.5 / 2.5 / 0.2 | Temperature scan range |

One Monte Carlo sweep is defined as `N^2` trial moves, so on average every
spin gets one attempted update per sweep — independent of lattice size.

## Extension: second-neighbour coupling

Adding a diagonal next-nearest-neighbour coupling `J2` (with `J2 / J1`
controlled by `--j2-ratio`) introduces 4 extra bonds per site along
(±1, ±1). More ferromagnetic coupling means a more strongly ordered
phase, so the transition temperature shifts upward and vortex cores cost
more energy.

### Running the extension

```bash
# J2 = 0.5 * J1 scan, same range and size as the main-branch run
python xy_simulation.py --scan --size 20 \
    --t-min 0.5 --t-max 2.5 --t-step 0.2 \
    --n-equil-sweeps 2000 --n-prod-sweeps 20000 \
    --j2-ratio 0.5

# writes data/scan_N20_j2_0.50.npz (the _j2_* suffix disambiguates from J2=0)
```

### Overlay comparison plot

```bash
python plot_results.py --mode compare-scans \
    --inputs data/scan_N20.npz data/scan_N20_j2_0.50.npz \
    --output figures/compare_scans_N20.png
```

Produces a 2x3 panel of `<|m|>`, `<e>`, τ, χ_M, C, and the vortex count
with both scans overlaid. Labels come from each file's stored
`j2_ratio`.

### Expected effects

- χ_M and C peaks move to higher T (rough shift with J2 = 0.5: from
  T ≈ 1.1 to T ≈ 1.9 on N = 20).
- τ peak tracks the new T_c — critical slowing down at the shifted
  transition.
- At a fixed T, configurations are smoother and carry fewer vortices.
- Energy per spin is more negative at every T because more bonds
  contribute.

### Regression guarantee

At `--j2-ratio 0` the code path through the NNN terms is gated by
`if j2_ratio != 0.0`, so the trajectory is bit-identical to the
nearest-neighbour simulation on `main` for the same seed.

## Notes on the algorithm

- Trial step: pick a site uniformly at random, propose
  `theta' = theta + u * delta` with `u ~ Uniform(-1, 1)`. The trial proposal
  is symmetric so detailed balance only requires the Metropolis
  accept/reject step.
- Accept always if `dE <= 0`, otherwise accept with probability
  `exp(-beta * dE)`. The energy change `dE` only depends on the chosen spin
  and its four nearest neighbours.
- Total energy and the two components `M_x`, `M_y` of the magnetization are
  updated from the increment of every accepted move, which avoids an O(N^2)
  recomputation each sweep.
