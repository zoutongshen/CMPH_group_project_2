"""
Monte Carlo simulation of the 2D XY model.

Implements the Metropolis algorithm on an N x N square lattice with periodic
boundary conditions. Each site carries a spin s_i = (cos theta_i, sin theta_i)
with theta_i in [-pi, pi]. The Hamiltonian is

    H = -J sum_<i,j> s_i . s_j = -J sum_<i,j> cos(theta_i - theta_j)

with the sum running over nearest-neighbour pairs on the lattice. We set
J = k_B = 1 throughout.

Author: CMPH 2026 Project 2
Date: April 2026
"""

import argparse
import numpy as np
from typing import Optional, Tuple
from tqdm import tqdm


def local_energy(angles: np.ndarray, row: int, col: int) -> float:
    """
    Interaction energy of a single site with its four neighbours.

    Uses periodic boundary conditions via the modulo operator.

    Args:
        angles: Array of shape (size, size) with spin angles
        row: Row index of the spin
        col: Column index of the spin

    Returns:
        -sum_{nn} cos(theta_i - theta_nn) for the specified site.
    """
    size = angles.shape[0]
    theta = angles[row, col]
    right = angles[row, (col + 1) % size]
    left = angles[row, (col - 1) % size]
    up = angles[(row - 1) % size, col]
    down = angles[(row + 1) % size, col]
    return -(np.cos(theta - right) + np.cos(theta - left)
             + np.cos(theta - up) + np.cos(theta - down))


def total_energy(angles: np.ndarray) -> float:
    """
    Total energy of the lattice under the XY Hamiltonian.

    Sums over nearest-neighbour pairs without double counting by looking only
    at the right and down neighbours of each site.

    Args:
        angles: Array of shape (size, size) with spin angles

    Returns:
        Total energy, i.e. -sum_<i,j> cos(theta_i - theta_j).
    """
    size = angles.shape[0]
    energy = 0.0
    for row in range(size):
        for col in range(size):
            theta = angles[row, col]
            right = angles[row, (col + 1) % size]
            down = angles[(row + 1) % size, col]
            energy -= np.cos(theta - right) + np.cos(theta - down)
    return energy


def total_magnetization(angles: np.ndarray) -> Tuple[float, float]:
    """
    Vector magnetization of the lattice.

    For spins s_i = (cos theta_i, sin theta_i), returns the (x, y) components
    of M = sum_i s_i.

    Args:
        angles: Array of shape (size, size) with spin angles

    Returns:
        Tuple (Mx, My).
    """
    return float(np.sum(np.cos(angles))), float(np.sum(np.sin(angles)))


def metropolis_sweep(angles: np.ndarray, rng: np.random.Generator,
                     *, beta: float, delta: float) -> Tuple[float, float, float, int]:
    """
    Perform one Monte Carlo sweep over the lattice.

    A sweep consists of N^2 single-spin trial moves. Each attempt picks a site
    uniformly at random, proposes a new angle theta' = theta + u * delta with
    u drawn uniformly from (-1, 1), and accepts with the Metropolis criterion
    min(1, exp(-beta * dE)).

    Random numbers for the whole sweep are drawn in one numpy call each; this
    is noticeably cheaper than drawing a scalar per trial move.

    Args:
        angles: Lattice of shape (size, size); modified in place
        rng: Random generator used to draw the trial moves
        beta: Inverse temperature 1 / (k_B T)
        delta: Maximum absolute angle change per trial move

    Returns:
        Tuple (energy_change, mag_x_change, mag_y_change, n_accepted) summed
        over this sweep.
    """
    size = angles.shape[0]
    n_sites = size * size
    proposals = delta * rng.uniform(-1.0, 1.0, size=n_sites)
    sites = rng.integers(0, size, size=(n_sites, 2))
    accept_draws = rng.random(size=n_sites)

    energy_change = 0.0
    mag_x_change = 0.0
    mag_y_change = 0.0
    n_accepted = 0

    for step in range(n_sites):
        row = sites[step, 0]
        col = sites[step, 1]
        theta_old = angles[row, col]
        theta_new = theta_old + proposals[step]

        e_old = local_energy(angles, row, col)
        angles[row, col] = theta_new
        e_new = local_energy(angles, row, col)
        delta_e = e_new - e_old

        if delta_e <= 0.0 or accept_draws[step] < np.exp(-beta * delta_e):
            energy_change += delta_e
            mag_x_change += np.cos(theta_new) - np.cos(theta_old)
            mag_y_change += np.sin(theta_new) - np.sin(theta_old)
            n_accepted += 1
        else:
            angles[row, col] = theta_old

    return energy_change, mag_x_change, mag_y_change, n_accepted


def initial_lattice(size: int, rng: np.random.Generator, *,
                    ordered: bool = False) -> np.ndarray:
    """
    Generate an initial spin configuration.

    Args:
        size: Lattice side length N (total of N^2 spins)
        rng: Random generator used when ordered=False
        ordered: If True, all spins start aligned at theta = 0 (T = 0 limit).
                 Otherwise angles are drawn uniformly from [-pi, pi]
                 (T = infinity limit).

    Returns:
        Array of shape (size, size) with spin angles.
    """
    if ordered:
        return np.zeros((size, size))
    return rng.uniform(-np.pi, np.pi, size=(size, size))


class XYModel:
    """
    State and dynamics of a 2D XY model simulated by the Metropolis algorithm.

    The lattice is stored as an (N, N) array of angles. Total energy and the
    two components of the magnetization vector are kept updated incrementally
    from the per-sweep deltas returned by the sweep kernel.
    """

    def __init__(self, size: int, temperature: float, *,
                 ordered_start: bool = False, delta: float = np.pi,
                 seed: Optional[int] = None):
        """
        Initialise the simulation.

        Args:
            size: Lattice side length N
            temperature: Temperature T in units where J = k_B = 1
            ordered_start: If True, start from all-aligned spins (T=0 limit);
                           otherwise start from uniformly random spins.
            delta: Maximum absolute angle change per Metropolis trial move
            seed: Optional random seed for the internal generator
        """
        self.size = size
        self.n_sites = size * size
        self.temperature = temperature
        self.beta = 1.0 / temperature
        self.delta = delta
        self.rng = np.random.default_rng(seed)

        self.angles = initial_lattice(size, self.rng, ordered=ordered_start)
        self.energy = total_energy(self.angles)
        mag_x, mag_y = total_magnetization(self.angles)
        self.mag_x = mag_x
        self.mag_y = mag_y

    def sweep(self) -> int:
        """
        Advance the simulation by one sweep (N^2 single-spin trial moves).

        Delegates to the module-level metropolis_sweep kernel and updates the
        cached energy and magnetization from the increments it returns.

        Returns:
            Number of accepted moves in this sweep.
        """
        delta_e, delta_mx, delta_my, n_accepted = metropolis_sweep(
            self.angles, self.rng, beta=self.beta, delta=self.delta)

        self.energy += delta_e
        self.mag_x += delta_mx
        self.mag_y += delta_my
        return n_accepted

    def magnetization_per_spin(self) -> float:
        """Return |M| / N^2, the magnitude of the mean spin vector."""
        return np.sqrt(self.mag_x**2 + self.mag_y**2) / self.n_sites

    def energy_per_spin(self) -> float:
        """Return the total energy divided by N^2."""
        return self.energy / self.n_sites

    def equilibrate(self, n_sweeps: int, *, show_progress: bool = False) -> None:
        """
        Run n_sweeps sweeps and discard the trajectory.

        Used to relax the system toward thermal equilibrium before production
        measurements.

        Args:
            n_sweeps: Number of sweeps
            show_progress: If True, display a tqdm progress bar
        """
        iterator = tqdm(range(n_sweeps), desc="equilibrating") if show_progress else range(n_sweeps)
        for _ in iterator:
            self.sweep()

    def simulate(self, n_sweeps: int, *, record_interval: int = 1,
                 show_progress: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run n_sweeps sweeps and record observables along the way.

        Args:
            n_sweeps: Number of sweeps
            record_interval: Record observables every this many sweeps
            show_progress: If True, display a tqdm progress bar

        Returns:
            Tuple (sweep_indices, energies_per_spin, magnetizations_per_spin).
            Each array has length ceil(n_sweeps / record_interval).
        """
        n_records = (n_sweeps + record_interval - 1) // record_interval
        sweep_indices = np.zeros(n_records, dtype=np.int64)
        energies_per_spin = np.zeros(n_records)
        magnetizations_per_spin = np.zeros(n_records)

        iterator = tqdm(range(n_sweeps), desc=f"T={self.temperature:.2f}") if show_progress else range(n_sweeps)
        record_idx = 0
        for sweep_idx in iterator:
            self.sweep()
            if sweep_idx % record_interval == 0:
                sweep_indices[record_idx] = sweep_idx
                energies_per_spin[record_idx] = self.energy_per_spin()
                magnetizations_per_spin[record_idx] = self.magnetization_per_spin()
                record_idx += 1

        return sweep_indices, energies_per_spin, magnetizations_per_spin


def temperature_scan(size: int, temperatures: np.ndarray, *,
                     n_equil_sweeps: int, n_prod_sweeps: int,
                     record_interval: int = 1, seed: Optional[int] = None,
                     show_progress: bool = True) -> dict:
    """
    Run independent simulations at a list of temperatures.

    For each temperature a fresh ordered-start simulation is equilibrated and
    then sampled. The mean and standard deviation of the per-spin observables
    over the production run are recorded. The final lattice configuration is
    kept for later visualisation.

    Args:
        size: Lattice side length N
        temperatures: 1D array of temperatures
        n_equil_sweeps: Equilibration sweeps per temperature
        n_prod_sweeps: Production sweeps per temperature
        record_interval: Sampling interval during production
        seed: Base random seed; each temperature uses seed + index
        show_progress: Whether to show a tqdm bar per simulation

    Returns:
        Dict with keys 'temperatures', 'energy_mean', 'energy_std',
        'magnetization_mean', 'magnetization_std', 'final_angles'
        (stacked array of shape (len(temperatures), size, size)).
    """
    energy_mean = np.zeros_like(temperatures, dtype=float)
    energy_std = np.zeros_like(temperatures, dtype=float)
    mag_mean = np.zeros_like(temperatures, dtype=float)
    mag_std = np.zeros_like(temperatures, dtype=float)
    final_angles = np.zeros((len(temperatures), size, size))

    for i, temperature in enumerate(temperatures):
        model = XYModel(size, temperature, ordered_start=True,
                        seed=(seed + i) if seed is not None else None)
        model.equilibrate(n_equil_sweeps, show_progress=show_progress)
        _, energies, magnetizations = model.simulate(
            n_prod_sweeps, record_interval=record_interval,
            show_progress=show_progress)
        energy_mean[i] = energies.mean()
        energy_std[i] = energies.std()
        mag_mean[i] = magnetizations.mean()
        mag_std[i] = magnetizations.std()
        final_angles[i] = model.angles

    return {
        'temperatures': temperatures,
        'energy_mean': energy_mean,
        'energy_std': energy_std,
        'magnetization_mean': mag_mean,
        'magnetization_std': mag_std,
        'final_angles': final_angles,
    }


def run_single(args: argparse.Namespace) -> None:
    """Run one simulation at a single temperature and save the trajectory."""
    print(f"Running N={args.size} at T={args.temperature}")
    model = XYModel(args.size, args.temperature,
                    ordered_start=False, delta=args.delta, seed=args.seed)
    model.equilibrate(args.n_equil_sweeps, show_progress=True)
    sweeps, energies, magnetizations = model.simulate(
        args.n_prod_sweeps, record_interval=args.record_interval,
        show_progress=True)

    filename = f'data/run_N{args.size}_T{args.temperature:.2f}.npz'
    np.savez(filename,
             sweeps=sweeps,
             energies=energies,
             magnetizations=magnetizations,
             final_angles=model.angles)
    print(f"Saved {filename}")
    print(f"Final energy/spin = {model.energy_per_spin():.4f}, "
          f"|M|/spin = {model.magnetization_per_spin():.4f}")


def run_two_starts(args: argparse.Namespace) -> None:
    """Run an ordered-start and a random-start simulation at the same T."""
    print(f"Running two starts at T={args.temperature}, N={args.size}")
    results = {}
    for label, ordered in [('ordered', True), ('random', False)]:
        model = XYModel(args.size, args.temperature,
                        ordered_start=ordered, delta=args.delta,
                        seed=args.seed + (0 if ordered else 1))
        sweeps, energies, magnetizations = model.simulate(
            args.n_equil_sweeps + args.n_prod_sweeps,
            record_interval=args.record_interval,
            show_progress=True)
        results[f'sweeps_{label}'] = sweeps
        results[f'energies_{label}'] = energies
        results[f'magnetizations_{label}'] = magnetizations
        results[f'final_angles_{label}'] = model.angles.copy()

    filename = f'data/two_starts_N{args.size}_T{args.temperature:.2f}.npz'
    np.savez(filename, **results)
    print(f"Saved {filename}")


def run_scan(args: argparse.Namespace) -> None:
    """Run an equilibrated simulation at every temperature in the requested range."""
    temperatures = np.arange(args.t_min, args.t_max + 1e-9, args.t_step)
    print(f"Running temperature scan, N={args.size}, {len(temperatures)} temperatures")
    results = temperature_scan(
        args.size, temperatures,
        n_equil_sweeps=args.n_equil_sweeps,
        n_prod_sweeps=args.n_prod_sweeps,
        record_interval=args.record_interval,
        seed=args.seed)
    filename = f'data/scan_N{args.size}.npz'
    np.savez(filename, **results)
    print(f"Saved {filename}")


def main() -> None:
    """
    Command-line entry point.

    Three modes:
      - default: single (size, temperature) run; saves sweep/energy/|M| time
        series and the final lattice.
      - --two-starts: one ordered-start and one random-start run at the same
        temperature so equilibration curves can be compared.
      - --scan: temperature sweep from --t-min to --t-max in steps of --t-step;
        saves mean and std of the observables at each temperature.
    """
    parser = argparse.ArgumentParser(description="2D XY model Monte Carlo simulation")
    parser.add_argument("--size", type=int, default=20,
                        help="Lattice side length N (default: 20)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature in units of J/k_B (default: 1.0)")
    parser.add_argument("--n-equil-sweeps", type=int, default=1000,
                        help="Equilibration sweeps (default: 1000)")
    parser.add_argument("--n-prod-sweeps", type=int, default=5000,
                        help="Production sweeps (default: 5000)")
    parser.add_argument("--record-interval", type=int, default=1,
                        help="Record observables every this many sweeps (default: 1)")
    parser.add_argument("--delta", type=float, default=np.pi,
                        help="Metropolis angle step size (default: pi)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--two-starts", action="store_true",
                        help="Run both ordered and random initial conditions at the given T")
    parser.add_argument("--scan", action="store_true",
                        help="Sweep temperature from --t-min to --t-max in steps of --t-step")
    parser.add_argument("--t-min", type=float, default=0.5,
                        help="Minimum temperature for --scan (default: 0.5)")
    parser.add_argument("--t-max", type=float, default=2.5,
                        help="Maximum temperature for --scan (default: 2.5)")
    parser.add_argument("--t-step", type=float, default=0.2,
                        help="Temperature step for --scan (default: 0.2)")
    args = parser.parse_args()

    if args.scan:
        run_scan(args)
    elif args.two_starts:
        run_two_starts(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
