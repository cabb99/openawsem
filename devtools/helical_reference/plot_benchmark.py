#!/usr/bin/env python
"""
Turn ``benchmark.py`` output into a Markdown table and a log-log chart.

Usage::

    python devtools/helical_reference/plot_benchmark.py helical_benchmark.json \
        --png results/helical_benchmark.png --md results/helical_benchmark.md
"""
import argparse
import json
from pathlib import Path

# Categorical slots 1-7 of the reference data-viz palette (fixed order); the inexact
# baseline is drawn as a muted reference line rather than as a series colour.
SERIES = [
    ('numba-kdtree', 'python', 'numba + k-d tree (no OpenMM)'),
    ('numpy-kdtree', 'python', 'NumPy + k-d tree (no OpenMM)'),
    ('torch', 'python', 'PyTorch autograd (no OpenMM)'),
    ('openmm-python[numba]', 'Reference', 'OpenMM PythonForce + numba'),
    ('openmm-torchforce', 'CPU', 'OpenMM TorchForce'),
    ('openmm-cvforce', 'Reference', 'OpenMM CustomCVForce'),
    ('openmm-compound', 'Reference', 'OpenMM CustomCompoundBondForce'),
]
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6']
BASELINE = ('openmm-frozen_density', 'Reference', 'Frozen density, like helical_term (inexact)')
INK, INK_2, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df', '#fcfcfb'


def load(path):
    data = json.loads(Path(path).read_text())
    return data['meta'], [r for r in data['rows'] if r['status'] == 'ok']


def series(rows, method, platform, key):
    pts = sorted((r['n_residues'], r[key]) for r in rows if r['method'] == method and r['platform'] == platform)
    return [p[0] for p in pts], [p[1] for p in pts]


def markdown(meta, rows, path):
    sizes = sorted({r['n_residues'] for r in rows})
    combos = sorted({(r['method'], r['platform']) for r in rows})
    lines = [f"Machine: {meta.get('processor') or meta.get('machine')}; OpenMM {meta.get('openmm', '?')}, "
             f"PyTorch {meta.get('torch', '?')} ({meta.get('torch_threads', '?')} threads).", '',
             '**Energy + forces, ms per evaluation** (median; `-` = not run, over the time budget)', '',
             '| Method | Platform | ' + ' | '.join(f'N={n}' for n in sizes) + ' |',
             '|---|---|' + '---:|' * len(sizes)]
    for method, plat in combos:
        vals = {r['n_residues']: r['eval_ms'] for r in rows if r['method'] == method and r['platform'] == plat}
        lines.append(f'| {method} | {plat} | ' + ' | '.join(f'{vals[n]:.3g}' if n in vals else '-' for n in sizes) + ' |')
    lines += ['', '**Setup (force construction + Context + first evaluation), s**', '',
              '| Method | Platform | ' + ' | '.join(f'N={n}' for n in sizes) + ' |',
              '|---|---|' + '---:|' * len(sizes)]
    for method, plat in combos:
        vals = {r['n_residues']: r['setup_s'] for r in rows if r['method'] == method and r['platform'] == plat}
        lines.append(f'| {method} | {plat} | ' + ' | '.join(f'{vals[n]:.3g}' if n in vals else '-' for n in sizes) + ' |')
    Path(path).write_text('\n'.join(lines) + '\n')


def chart(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK_2,
                         'xtick.color': INK_2, 'ytick.color': INK_2, 'text.color': INK})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE)
    panels = [('eval_ms', 'Energy + forces per evaluation (ms)'), ('setup_s', 'Setup incl. Context creation (s)')]
    for ax, (key, title) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for (method, plat, label), color in zip(SERIES, COLORS):
            x, y = series(rows, method, plat, key)
            if x:
                ax.plot(x, y, color=color, lw=2, marker='o', ms=5, mec=SURFACE, mew=1.5, label=label)
        x, y = series(rows, *BASELINE[:2], key)
        if x:
            ax.plot(x, y, color=MUTED, lw=1.5, ls='--', label=BASELINE[2])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Residues')
        ax.set_title(title, loc='left', fontsize=11, color=INK)
        ax.grid(True, which='major', color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('json', type=Path)
    parser.add_argument('--png', type=Path, default=None)
    parser.add_argument('--md', type=Path, default=None)
    args = parser.parse_args()
    meta, rows = load(args.json)
    if args.md:
        markdown(meta, rows, args.md)
    if args.png:
        chart(rows, args.png)


if __name__ == '__main__':
    main()
