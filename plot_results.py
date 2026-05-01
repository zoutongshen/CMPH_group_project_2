"""
Plotting utilities for the 2D XY model Monte Carlo simulation.

Loads .npz files saved by xy_simulation.py and produces:
  - magnetization and energy vs sweep index for a single run
  - two-starts equilibration comparison
  - 2x2 temperature-scan panel of <|m|>, <e>, chi_M, and C with correlated
    and blocking-method error bars (via analysis.scan_statistics)
  - correlation time tau as a function of temperature
  - vortex and anti-vortex counts of the final configurations vs temperature
  - final spin configuration as an HSV colour grid or as arrows, optionally
    with vortices / anti-vortices circled
  - a grid of final configurations across a temperature scan
  - side-by-side comparison of several scans (compare-scans mode), for
    instance J2 = 0 vs J2 > 0 from the extension branch

Plots that read a scan file pick up j2_ratio from the .npz and show it in
the title when it is non-zero; pre-extension scan files without the field
are treated as j2_ratio = 0.

Author: CMPH 2026 Project 2
Date: April 2026
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

import analysis
import vortices


def save_figure(fig: plt.Figure, figure_path: str) -> None:
    """Tighten layout, write PNG, close the figure, and log the path."""
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    print(f"Saved {figure_path}")
    plt.close(fig)


def wrapped_angles(angles: np.ndarray) -> np.ndarray:
    """Return angles wrapped into the interval [-pi, pi) for colour mapping."""
    return np.mod(angles + np.pi, 2 * np.pi) - np.pi


def _j2_ratio_of(data: np.lib.npyio.NpzFile) -> float:
    """Read j2_ratio from an .npz file; default 0.0 for pre-extension data."""
    if 'j2_ratio' in data.files:
        return float(data['j2_ratio'])
    return 0.0


def _j2_label(j2_ratio: float) -> str:
    """Short figure-title fragment for j2_ratio; empty when zero."""
    if j2_ratio == 0.0:
        return ''
    return f'  [J2/J1 = {j2_ratio:g}]'


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


def _scan_statistics_from_file(filename: str) -> tuple:
    """
    Load a temperature-scan .npz and return (stats, temperatures, n_sites).

    Args:
        filename: Path to an .npz file saved by xy_simulation.py --scan

    Returns:
        Tuple (stats_dict, temperatures, n_sites) where stats_dict is the
        return value of analysis.scan_statistics().
    """
    data = np.load(filename)
    temperatures = data['temperatures']
    magnetization_series = data['magnetization_series']
    energy_series = data['energy_series']
    n_sites = int(data['final_angles'].shape[1] * data['final_angles'].shape[2])
    stats = analysis.scan_statistics(
        magnetization_series, energy_series, temperatures, n_sites)
    stats['j2_ratio'] = _j2_ratio_of(data)
    return stats, temperatures, n_sites


def plot_scan_observables(filename: str, figure_path: str) -> None:
    """
    Plot the four lecture-8 observables as a function of temperature.

    Produces a 2x2 panel of <|m|>, <e>, chi_M, C with correlated standard
    errors on the first two and blocking-method errors on chi_M and C.

    Args:
        filename: Path to an .npz file saved by xy_simulation.py --scan
        figure_path: Output PNG path
    """
    stats, temperatures, _ = _scan_statistics_from_file(filename)

    # One spec per panel: which key to read, axis label, marker, colour.
    # Reading from a list keeps the four very-similar plot blocks DRY.
    panel_specs = [
        ((0, 0), 'magnetization_mean', 'magnetization_error',
         r'$\langle |m| \rangle$', 'o-', 'tab:blue'),
        ((0, 1), 'energy_mean', 'energy_error',
         r'$\langle e \rangle$', 's-', 'tab:red'),
        ((1, 0), 'susceptibility', 'susceptibility_error',
         r'$\chi_M$', 'D-', 'tab:green'),
        ((1, 1), 'specific_heat', 'specific_heat_error',
         'C', '^-', 'tab:purple'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for (row, col), mean_key, error_key, ylabel, fmt, color in panel_specs:
        ax = axes[row, col]
        ax.errorbar(temperatures, stats[mean_key], yerr=stats[error_key],
                    fmt=fmt, color=color, capsize=3)
        ax.set_xlabel('Temperature T')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    suffix = _j2_label(stats['j2_ratio'])
    if suffix:
        fig.suptitle(suffix.strip())
    save_figure(fig, figure_path)


def plot_correlation_time(filename: str, figure_path: str,
                          *, critical_temperature: float = 0.881) -> None:
    """
    Plot the correlation time tau as a function of temperature.

    tau is expected to peak near the critical temperature (critical slowing
    down). The dashed vertical line marks T_c for reference.

    Args:
        filename: Path to an .npz file saved by xy_simulation.py --scan
        figure_path: Output PNG path
        critical_temperature: Reference T_c drawn as a dashed line
    """
    stats, temperatures, _ = _scan_statistics_from_file(filename)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(temperatures, stats['tau'], 'o-', color='tab:orange')
    ax.axvline(critical_temperature, linestyle='--', color='grey',
               label=f'$T_c = {critical_temperature}$')
    ax.set_xlabel('Temperature T')
    ax.set_ylabel(r'Correlation time $\tau$ (sweeps)')
    suffix = _j2_label(stats['j2_ratio'])
    if suffix:
        ax.set_title(suffix.strip())
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_figure(fig, figure_path)


def plot_autocorrelation_curves(filename: str, figure_path: str,
                                *, target_temperatures: tuple = (0.5, 0.9, 2.1)
                                ) -> None:
    """
    Plot the normalised autocorrelation function of |M|/N^2 at several T.

    For each target temperature the closest temperature in the scan is used.
    Each curve is truncated at the first lag where chi(t) turns negative,
    matching the cutoff used by analysis.correlation_time and avoiding the
    noisy large-lag tail.

    Args:
        filename: Path to an .npz file saved by xy_simulation.py --scan
        figure_path: Output PNG path
        target_temperatures: Temperatures to display; the nearest available
                             scan temperature is selected for each.
    """
    data = np.load(filename)
    temperatures = data['temperatures']
    magnetization_series = data['magnetization_series']

    fig, ax = plt.subplots(figsize=(8, 5))
    for target in target_temperatures:
        index = int(np.argmin(np.abs(temperatures - target)))
        actual_temperature = float(temperatures[index])
        series = magnetization_series[index]

        chi = analysis.autocorrelation(series)
        if chi[0] <= 0.0:
            continue
        normalised = chi / chi[0]
        negative = np.where(normalised < 0.0)[0]
        cutoff = int(negative[0]) if negative.size else len(normalised)
        lags = np.arange(cutoff)
        ax.plot(lags, normalised[:cutoff],
                label=f'T = {actual_temperature:.2f}')

    ax.axhline(0.0, color='grey', linewidth=0.8, linestyle=':')
    ax.set_xlabel('Lag t (sweeps)')
    ax.set_ylabel(r'$\chi(t) / \chi(0)$')
    ax.grid(True, alpha=0.3)
    ax.legend()
    suffix = _j2_label(_j2_ratio_of(data))
    if suffix:
        ax.set_title(suffix.strip())
    save_figure(fig, figure_path)


def plot_vortex_counts(filename: str, figure_path: str) -> None:
    """
    Plot the vortex and anti-vortex counts of the final configurations.

    Reads the per-temperature final configurations from a scan file and
    counts the topological charges. On a periodic lattice the two counts
    must match exactly.

    Args:
        filename: Path to an .npz file saved by xy_simulation.py --scan
        figure_path: Output PNG path
    """
    data = np.load(filename)
    temperatures = data['temperatures']
    final_angles = data['final_angles']

    n_vortices = np.zeros(len(temperatures), dtype=int)
    n_antivortices = np.zeros(len(temperatures), dtype=int)
    for i, angles in enumerate(final_angles):
        n_vortices[i], n_antivortices[i] = vortices.count_vortices(angles)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(temperatures, n_vortices, 'o-', color='tab:red',
            label='vortices (+1)')
    ax.plot(temperatures, n_antivortices, 's--', color='tab:blue',
            label='anti-vortices (-1)')
    ax.set_xlabel('Temperature T')
    ax.set_ylabel('Count per final configuration')
    suffix = _j2_label(_j2_ratio_of(data))
    if suffix:
        ax.set_title(suffix.strip())
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_figure(fig, figure_path)


def _overlay_vortices(ax: plt.Axes, angles: np.ndarray) -> None:
    """Mark +1 vortices in red and -1 anti-vortices in blue on an axes."""
    (vortex_rows, vortex_cols), (antivortex_rows, antivortex_cols) = \
        vortices.vortex_locations(angles)
    ax.scatter(vortex_cols, vortex_rows, s=60, facecolors='none',
               edgecolors='red', linewidths=1.5, label='+1')
    ax.scatter(antivortex_cols, antivortex_rows, s=60, facecolors='none',
               edgecolors='blue', linewidths=1.5, label=r'$-1$')


def plot_spin_configuration(angles: np.ndarray, figure_path: str,
                            title: str = "",
                            overlay_vortices: bool = False) -> None:
    """
    Plot a spin configuration as an HSV colour grid of spin angles.

    Each lattice site is shown as a pixel whose colour encodes the spin angle
    in [-pi, pi). This visualises vortices and aligned domains at a glance.

    Args:
        angles: Array of shape (size, size) with spin angles
        figure_path: Output PNG path
        title: Optional figure title
        overlay_vortices: If True, circle the locations of vortices and
                          anti-vortices identified by the plaquette winding.
    """
    fig, ax = plt.subplots(figsize=(6, 5.5))
    image = ax.imshow(wrapped_angles(angles), cmap='hsv',
                      vmin=-np.pi, vmax=np.pi, origin='lower')
    cbar = fig.colorbar(image, ax=ax, label='spin angle')
    cbar.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    cbar.set_ticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if overlay_vortices:
        _overlay_vortices(ax, angles)
        ax.legend(loc='upper right')
    if title:
        ax.set_title(title)
    save_figure(fig, figure_path)


def plot_spin_vectors(angles: np.ndarray, figure_path: str,
                      title: str = "",
                      overlay_vortices: bool = False) -> None:
    """
    Plot a spin configuration as arrows on the lattice.

    Args:
        angles: Array of shape (size, size) with spin angles
        figure_path: Output PNG path
        title: Optional figure title
        overlay_vortices: If True, circle vortices and anti-vortices.
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
    if overlay_vortices:
        _overlay_vortices(ax, angles)
        ax.legend(loc='upper right')
    if title:
        ax.set_title(title)
    save_figure(fig, figure_path)


def plot_compare_scans(filenames: list, figure_path: str) -> None:
    """
    Overlay the six scan summary quantities from several scan files.

    For each input file, computes tau, <|m|>, <e>, chi_M, C via
    analysis.scan_statistics and counts vortices in the final configurations,
    then plots them all on one 2x3 figure labelled by j2_ratio. This makes
    the J1 vs J1+J2 comparison a single artifact.

    Args:
        filenames: List of .npz paths saved by xy_simulation.py --scan
        figure_path: Output PNG path
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for filename in filenames:
        stats, temperatures, _ = _scan_statistics_from_file(filename)
        data = np.load(filename)
        final_angles = data['final_angles']
        n_vortices = np.array([vortices.count_vortices(cfg)[0]
                               for cfg in final_angles])
        label = (f'J2/J1 = {stats["j2_ratio"]:g}'
                 if stats['j2_ratio'] != 0.0 else 'J2 = 0')

        axes[0, 0].errorbar(temperatures, stats['magnetization_mean'],
                            yerr=stats['magnetization_error'],
                            fmt='o-', capsize=3, label=label)
        axes[0, 1].errorbar(temperatures, stats['energy_mean'],
                            yerr=stats['energy_error'],
                            fmt='s-', capsize=3, label=label)
        axes[0, 2].plot(temperatures, stats['tau'], 'D-', label=label)
        axes[1, 0].errorbar(temperatures, stats['susceptibility'],
                            yerr=stats['susceptibility_error'],
                            fmt='o-', capsize=3, label=label)
        axes[1, 1].errorbar(temperatures, stats['specific_heat'],
                            yerr=stats['specific_heat_error'],
                            fmt='s-', capsize=3, label=label)
        axes[1, 2].plot(temperatures, n_vortices, 'D-', label=label)

    for ax, ylabel in zip(
        axes.ravel(),
        [r'$\langle |m| \rangle$', r'$\langle e \rangle$',
         r'$\tau$ (sweeps)', r'$\chi_M$', 'C',
         'vortices per final configuration'],
    ):
        ax.set_xlabel('Temperature T')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()

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

    suffix = _j2_label(_j2_ratio_of(data))
    if suffix:
        fig.suptitle(suffix.strip())
    save_figure(fig, figure_path)


def main() -> None:
    """
    Command-line entry point for plotting.

    Dispatches based on --mode to the appropriate plotting routine.
    """
    parser = argparse.ArgumentParser(description="Plot XY model MC results")
    parser.add_argument("--mode",
                        choices=["time-series", "two-starts", "scan",
                                 "tau", "autocorrelation", "vortex-count",
                                 "compare-scans", "scan-configurations",
                                 "configuration", "vectors"],
                        required=True,
                        help="Which type of plot to produce")
    parser.add_argument("--input", type=str,
                        help="Input .npz file (single-file modes)")
    parser.add_argument("--inputs", type=str, nargs='+',
                        help="Input .npz files for compare-scans")
    parser.add_argument("--output", type=str, required=True,
                        help="Output PNG path")
    parser.add_argument("--title", type=str, default="",
                        help="Optional title for single-panel plots")
    parser.add_argument("--overlay-vortices", action="store_true",
                        help="Mark vortices / anti-vortices on configuration "
                             "and vectors plots")
    args = parser.parse_args()

    # compare-scans is the only multi-input mode; handle it separately.
    if args.mode == "compare-scans":
        if not args.inputs:
            parser.error("--inputs is required for --mode compare-scans")
        plot_compare_scans(args.inputs, args.output)
        return

    if not args.input:
        parser.error("--input is required for this mode")

    # Plain (input, output) plotters dispatched from a name -> function map.
    single_input_plotters = {
        "time-series": plot_time_series,
        "two-starts": plot_two_starts,
        "scan": plot_scan_observables,
        "tau": plot_correlation_time,
        "autocorrelation": plot_autocorrelation_curves,
        "vortex-count": plot_vortex_counts,
        "scan-configurations": plot_scan_configurations,
    }
    if args.mode in single_input_plotters:
        single_input_plotters[args.mode](args.input, args.output)
        return

    # configuration / vectors take an extracted angles array plus optional
    # title and vortex overlay flag; handle them after the simple dispatch.
    angles = np.load(args.input)['final_angles']
    if args.mode == "configuration":
        plot_spin_configuration(angles, args.output, title=args.title,
                                overlay_vortices=args.overlay_vortices)
    elif args.mode == "vectors":
        plot_spin_vectors(angles, args.output, title=args.title,
                          overlay_vortices=args.overlay_vortices)


if __name__ == "__main__":
    main()
