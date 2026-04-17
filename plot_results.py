"""
Plotting utilities for the 2D XY model Monte Carlo simulation.

Loads .npz files saved by xy_simulation.py and produces:
  - magnetization and energy vs sweep index for a single run
  - two-starts equilibration comparison
  - temperature scan of mean magnetization and energy
  - final spin configuration as an HSV colour grid or as arrows
  - a grid of final configurations across a temperature scan

Author: CMPH 2026 Project 2
Date: April 2026
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt


def save_figure(fig: plt.Figure, figure_path: str) -> None:
    """Tighten layout, write PNG, close the figure, and log the path."""
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    print(f"Saved {figure_path}")
    plt.close(fig)


def wrapped_angles(angles: np.ndarray) -> np.ndarray:
    """Return angles wrapped into the interval [-pi, pi) for colour mapping."""
    return np.mod(angles + np.pi, 2 * np.pi) - np.pi


def plot_time_series(filename: str, figure_path: str) -> None:
    """
    Plot |M|/spin and energy/spin as a function of sweep index.

    Args:
        filename: Path to an .npz file saved by xy_simulation.py (default mode)
        figure_path: Output PNG path
    """
    data = np.load(filename)
    sweeps = data['sweeps']
    energies = data['energies']
    magnetizations = data['magnetizations']

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(sweeps, magnetizations, color='tab:blue')
    axes[0].set_ylabel('|M| / N$^2$')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(sweeps, energies, color='tab:red')
    axes[1].set_xlabel('Monte Carlo sweep')
    axes[1].set_ylabel('E / N$^2$')
    axes[1].grid(True, alpha=0.3)

    save_figure(fig, figure_path)


def plot_two_starts(filename: str, figure_path: str) -> None:
    """
    Plot the magnetization and energy time series from ordered and random starts.

    Args:
        filename: Path to an .npz file saved by xy_simulation.py --two-starts
        figure_path: Output PNG path
    """
    data = np.load(filename)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(data['sweeps_ordered'], data['magnetizations_ordered'],
                 label='ordered start', color='tab:blue')
    axes[0].plot(data['sweeps_random'], data['magnetizations_random'],
                 label='random start', color='tab:orange')
    axes[0].set_ylabel('|M| / N$^2$')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(data['sweeps_ordered'], data['energies_ordered'],
                 label='ordered start', color='tab:blue')
    axes[1].plot(data['sweeps_random'], data['energies_random'],
                 label='random start', color='tab:orange')
    axes[1].set_xlabel('Monte Carlo sweep')
    axes[1].set_ylabel('E / N$^2$')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    save_figure(fig, figure_path)


def plot_temperature_scan(filename: str, figure_path: str) -> None:
    """
    Plot mean |M|/spin and mean E/spin as a function of temperature.

    Args:
        filename: Path to an .npz file saved by xy_simulation.py --scan
        figure_path: Output PNG path
    """
    data = np.load(filename)
    temperatures = data['temperatures']

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].errorbar(temperatures, data['magnetization_mean'],
                     yerr=data['magnetization_std'], fmt='o-',
                     color='tab:blue', capsize=3)
    axes[0].set_xlabel('Temperature T')
    axes[0].set_ylabel(r'$\langle |M| \rangle / N^2$')
    axes[0].grid(True, alpha=0.3)

    axes[1].errorbar(temperatures, data['energy_mean'],
                     yerr=data['energy_std'], fmt='s-',
                     color='tab:red', capsize=3)
    axes[1].set_xlabel('Temperature T')
    axes[1].set_ylabel(r'$\langle E \rangle / N^2$')
    axes[1].grid(True, alpha=0.3)

    save_figure(fig, figure_path)


def plot_spin_configuration(angles: np.ndarray, figure_path: str,
                            title: str = "") -> None:
    """
    Plot a spin configuration as an HSV colour grid of spin angles.

    Each lattice site is shown as a pixel whose colour encodes the spin angle
    in [-pi, pi). This visualises vortices and aligned domains at a glance.

    Args:
        angles: Array of shape (size, size) with spin angles
        figure_path: Output PNG path
        title: Optional figure title
    """
    fig, ax = plt.subplots(figsize=(6, 5.5))
    image = ax.imshow(wrapped_angles(angles), cmap='hsv',
                      vmin=-np.pi, vmax=np.pi, origin='lower')
    cbar = fig.colorbar(image, ax=ax, label='spin angle')
    cbar.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    cbar.set_ticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if title:
        ax.set_title(title)
    save_figure(fig, figure_path)


def plot_spin_vectors(angles: np.ndarray, figure_path: str,
                      title: str = "") -> None:
    """
    Plot a spin configuration as arrows on the lattice.

    Args:
        angles: Array of shape (size, size) with spin angles
        figure_path: Output PNG path
        title: Optional figure title
    """
    size = angles.shape[0]
    ys, xs = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.quiver(xs, ys, cos_angles, sin_angles, wrapped_angles(angles),
              cmap='hsv', pivot='middle', scale=size * 1.3)
    ax.set_aspect('equal')
    ax.set_xlim(-1, size)
    ax.set_ylim(-1, size)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if title:
        ax.set_title(title)
    save_figure(fig, figure_path)


def plot_scan_configurations(filename: str, figure_path: str) -> None:
    """
    Plot final spin configurations from a temperature scan as a grid of panels.

    Args:
        filename: Path to an .npz file saved by xy_simulation.py --scan
        figure_path: Output PNG path
    """
    data = np.load(filename)
    temperatures = data['temperatures']
    final_angles = data['final_angles']

    n_panels = len(temperatures)
    n_cols = min(4, n_panels)
    n_rows = (n_panels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.5 * n_cols, 3.5 * n_rows),
                             squeeze=False)

    for panel_idx, (temperature, angles) in enumerate(zip(temperatures, final_angles)):
        ax = axes[panel_idx // n_cols, panel_idx % n_cols]
        ax.imshow(wrapped_angles(angles), cmap='hsv',
                  vmin=-np.pi, vmax=np.pi, origin='lower')
        ax.set_title(f'T = {temperature:.2f}')
        ax.set_xticks([])
        ax.set_yticks([])

    for panel_idx in range(n_panels, n_rows * n_cols):
        axes[panel_idx // n_cols, panel_idx % n_cols].axis('off')

    save_figure(fig, figure_path)


def main() -> None:
    """
    Command-line entry point for plotting.

    Dispatches based on --mode to the appropriate plotting routine.
    """
    parser = argparse.ArgumentParser(description="Plot XY model MC results")
    parser.add_argument("--mode",
                        choices=["time-series", "two-starts", "scan",
                                 "scan-configurations", "configuration",
                                 "vectors"],
                        required=True,
                        help="Which type of plot to produce")
    parser.add_argument("--input", type=str, required=True,
                        help="Input .npz file produced by xy_simulation.py")
    parser.add_argument("--output", type=str, required=True,
                        help="Output PNG path")
    parser.add_argument("--title", type=str, default="",
                        help="Optional title for single-panel plots")
    args = parser.parse_args()

    if args.mode == "time-series":
        plot_time_series(args.input, args.output)
    elif args.mode == "two-starts":
        plot_two_starts(args.input, args.output)
    elif args.mode == "scan":
        plot_temperature_scan(args.input, args.output)
    elif args.mode == "scan-configurations":
        plot_scan_configurations(args.input, args.output)
    elif args.mode == "configuration":
        angles = np.load(args.input)['final_angles']
        plot_spin_configuration(angles, args.output, title=args.title)
    elif args.mode == "vectors":
        angles = np.load(args.input)['final_angles']
        plot_spin_vectors(angles, args.output, title=args.title)


if __name__ == "__main__":
    main()
