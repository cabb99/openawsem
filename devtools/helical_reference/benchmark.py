#!/usr/bin/env python
"""
Benchmark every implementation of the AWSEM density-dependent helical term.

Systems are built from the stored myoglobin reference structure (153 residues):
the first N residues for N <= 153, and for larger N whole copies placed 60 A apart on
a cubic grid (one chain each) and cut to N residues. For each implementation and
OpenMM platform the script records

* ``setup``: building the force(s), creating the Context and the first evaluation (s);
* ``eval``: median wall time of one energy + force evaluation (ms);
* ``err``: largest force difference to the vectorized NumPy backend (kcal/mol/A).

A method is dropped for larger sizes once ``setup`` exceeds ``--max-setup`` or
``eval`` exceeds ``--max-eval``.

Usage::

    python devtools/helical_reference/benchmark.py --sizes 20 50 100 200 400 800 1600 3200 \
        --out helical_benchmark.json

    # e.g. on a GPU
    python devtools/helical_reference/benchmark.py --platforms CUDA --properties Precision=mixed
"""
import argparse
import json
import platform as pyplatform
import statistics
import time
from pathlib import Path

import numpy as np

from awsemmd_reference import CoarseGrainedProtein
from openawsem.functionTerms.helicalTerms import (HelixParameters, NumpyHelix, OpenMMEvaluator, build_helix_forces,
                                                  reference_energy_forces)

DATA = Path(__file__).resolve().parents[2] / 'tests' / 'data' / 'helical_reference'


def myoglobin():
    d = np.load(DATA / '1mbn_1r69_example.npz')
    x = d['positions'][0]
    seq = str(d['sequences'][0])
    return seq, x[0::3], x[1::3], x[2::3]


def make_system(n_residues: int) -> CoarseGrainedProtein:
    """First n residues of tiled myoglobin copies (one chain per copy)."""
    seq, ca, o, b = myoglobin()
    m = len(seq)
    copies = -(-n_residues // m)
    side = int(np.ceil(copies ** (1 / 3)))
    seqs, cas, os_, bs = [], [], [], []
    left = n_residues
    for c in range(copies):
        shift = 60.0 * np.array([c % side, (c // side) % side, c // side ** 2])
        k = min(m, left)
        seqs.append(seq[:k])
        cas.append(ca[:k] + shift)
        os_.append(o[:k] + shift)
        bs.append(b[:k] + shift)
        left -= k
    return CoarseGrainedProtein.from_residues(seqs, np.concatenate(cas), np.concatenate(os_), np.concatenate(bs))


def _timed(fn, repeats):
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return statistics.median(times)


def python_methods():
    """Name -> factory(topology, params) returning an object with compute(x)."""
    methods = {
        'numpy-dense': lambda t, p: NumpyHelix(t, p, neighbors='dense'),
        'numpy-kdtree': lambda t, p: NumpyHelix(t, p, neighbors='kdtree'),
    }
    try:
        import numba  # noqa: F401
        from openawsem.functionTerms.helicalTerms import NumbaHelix
        methods['numba-dense'] = lambda t, p: NumbaHelix(t, p, neighbors='dense')
        methods['numba-kdtree'] = lambda t, p: NumbaHelix(t, p, neighbors='kdtree')
    except ImportError:
        pass
    try:
        import torch  # noqa: F401
        from openawsem.functionTerms.helicalTerms import _TorchModel
        methods['torch'] = lambda t, p: _TorchModel(t, p)
    except ImportError:
        pass

    class _Ref:
        def __init__(self, t, p):
            self.t, self.p = t, p

        def compute(self, x):
            return reference_energy_forces(x, self.t, self.p, delta_helix_xi=0.0)
    methods['reference-loops'] = _Ref
    return methods


def openmm_methods():
    """Name -> (builder method, options)."""
    methods = {
        'openmm-frozen_density': ('frozen_density', {}),
        'openmm-compound': ('compound', {'max_residues': None}),
        'openmm-cvforce': ('cvforce', {}),
        'openmm-python[numpy]': ('python', {'backend': 'numpy'}),
        'openmm-python[numba]': ('python', {'backend': 'numba'}),
        'openmm-python[torch]': ('python', {'backend': 'torch'}),
        'openmm-torchforce': ('torchforce', {}),
    }
    return methods


# rq-60f9eee2
def run_benchmark(sizes, repeats=5, platforms=('Reference', 'CPU'), max_setup=300.0, max_eval=5.0,
                  methods=None, params=None, verbose=True, properties=None):
    """Benchmark rows: dicts with method, platform, n_residues, setup_s, eval_ms, max_force_err."""
    params = HelixParameters.awsemmd_1r69_example() if params is None else params
    rows = []
    dropped = set()
    py_methods = python_methods()
    import openmm
    available = {openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())}
    mm_methods = openmm_methods()
    for n in sizes:
        protein = make_system(n)
        x = protein.positions
        topo = protein.topology
        _, f_ref = NumpyHelix(topo, params).compute(x)
        jobs = [(name, None) for name in py_methods] + \
               [(name, plat) for name in mm_methods for plat in platforms if plat in available]
        for name, plat in jobs:
            key = (name, plat)
            if methods is not None and name not in methods:
                continue
            if key in dropped:
                continue
            try:
                t0 = time.perf_counter()
                if plat is None:
                    model = py_methods[name](topo, params)
                else:
                    method, options = mm_methods[name]
                    model = OpenMMEvaluator(protein.n_particles, build_helix_forces(method, topo, params, **options),
                                            plat, properties=properties)
                _, f = model.compute(x)
                setup = time.perf_counter() - t0
                for _ in range(3):  # warm-up (TorchScript's profiling executor re-optimizes early calls)
                    model.compute(x)
                evaluation = _timed(lambda: model.compute(x), repeats)
                err = float(np.abs(f - f_ref).max())
                status = 'ok'
            except Exception as exc:  # report and drop, keep benchmarking the others
                setup, evaluation, err, status = float('nan'), float('nan'), float('nan'), f'error: {exc}'[:200]
            row = dict(method=name, platform=plat or 'python', n_residues=n, setup_s=setup, eval_ms=evaluation * 1e3,
                       max_force_err=err, status=status)
            rows.append(row)
            if verbose:
                print(f"{name:28s} {row['platform']:9s} N={n:5d} setup={setup:9.3f}s eval={evaluation * 1e3:10.3f}ms "
                      f"err={err:.1e} {status if status != 'ok' else ''}", flush=True)
            if status != 'ok' or setup > max_setup or evaluation > max_eval:
                dropped.add(key)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sizes', type=int, nargs='+', default=[20, 50, 100, 200, 400, 800, 1600, 3200])
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--platforms', nargs='+', default=['Reference', 'CPU'])
    parser.add_argument('--max-setup', type=float, default=300.0)
    parser.add_argument('--max-eval', type=float, default=5.0)
    parser.add_argument('--methods', nargs='+', default=None)
    parser.add_argument('--properties', nargs='*', default=[], help='OpenMM platform properties, KEY=VALUE')
    parser.add_argument('--out', type=Path, default=Path('helical_benchmark.json'))
    args = parser.parse_args()
    properties = dict(p.split('=', 1) for p in args.properties)
    rows = run_benchmark(args.sizes, args.repeats, args.platforms, args.max_setup, args.max_eval, args.methods,
                         properties=properties)
    meta = dict(machine=pyplatform.platform(), processor=pyplatform.processor(), python=pyplatform.python_version(),
                platform_properties=properties)
    try:
        import openmm
        meta['openmm'] = openmm.__version__
    except ImportError:
        pass
    try:
        import torch
        meta['torch'] = torch.__version__
        meta['torch_threads'] = torch.get_num_threads()
    except ImportError:
        pass
    args.out.write_text(json.dumps(dict(meta=meta, rows=rows), indent=1) + '\n')


if __name__ == '__main__':
    main()
