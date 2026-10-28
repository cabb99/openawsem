Machine: x86_64; OpenMM 8.5.2, PyTorch 2.12.1 (4 threads).

**Energy + forces, ms per evaluation** (median; `-` = not run, over the time budget)

| Method | Platform | N=20 | N=50 | N=100 | N=200 | N=400 | N=800 | N=1600 | N=3200 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| numba-dense | python | 0.00716 | 0.0175 | 0.0451 | 0.22 | 0.423 | 1.54 | 5.52 | 22.2 |
| numba-kdtree | python | 0.0404 | 0.0483 | 0.0846 | 0.329 | 0.408 | 0.941 | 2.01 | 4.12 |
| numpy-dense | python | 0.123 | 0.184 | 0.412 | 1.79 | 3.43 | 12.3 | 52.9 | 227 |
| numpy-kdtree | python | 0.153 | 0.264 | 0.288 | 0.85 | 0.876 | 1.76 | 3.42 | 6.85 |
| openmm-compound | CPU | 8.28 | 114 | - | - | - | - | - | - |
| openmm-compound | Reference | 7.63 | 112 | - | - | - | - | - | - |
| openmm-cvforce | CPU | 7.08 | 22.9 | 44.4 | 92.8 | 197 | 410 | 1.06e+03 | 2.53e+03 |
| openmm-cvforce | Reference | 0.15 | 0.717 | 1.6 | 4.24 | 17.9 | 59 | 197 | 701 |
| openmm-frozen_density | CPU | 0.356 | 0.239 | 0.338 | 0.813 | 0.681 | 1.1 | 2.88 | 3.71 |
| openmm-frozen_density | Reference | 0.0433 | 0.0718 | 0.123 | 0.195 | 0.397 | 0.745 | 1.49 | 2.91 |
| openmm-python[numba] | CPU | 0.23 | 0.346 | 0.376 | 0.47 | 0.918 | 1.46 | 2.64 | 6.64 |
| openmm-python[numba] | Reference | 0.0414 | 0.063 | 0.102 | 0.195 | 0.519 | 1.1 | 2.24 | 4.46 |
| openmm-python[numpy] | CPU | 0.399 | 0.52 | 0.711 | 1.44 | 1.32 | 2.29 | 4.5 | 9.75 |
| openmm-python[numpy] | Reference | 0.202 | 0.291 | 0.453 | 0.588 | 1.03 | 2.03 | 3.98 | 7.4 |
| openmm-python[torch] | CPU | 1.56 | 1.53 | 1.59 | 2.52 | 5.02 | 15.1 | 67.9 | 264 |
| openmm-python[torch] | Reference | 1.06 | 1.94 | 1.6 | 3.28 | 5.08 | 17.1 | 96.9 | 281 |
| openmm-torchforce | CPU | 1.11 | 1.06 | 2 | 2.45 | 4.48 | 14.8 | 77.7 | 282 |
| openmm-torchforce | Reference | 0.619 | 0.78 | 1.25 | 2.2 | 4.37 | 21.2 | 75.4 | 276 |
| reference-loops | python | 1.4 | 9.75 | 23.6 | 82.7 | 337 | 1.33e+03 | 5.21e+03 | - |
| torch | python | 0.815 | 1.19 | 1.54 | 3.54 | 5.11 | 14.9 | 70.1 | 268 |

**Setup (force construction + Context + first evaluation), s**

| Method | Platform | N=20 | N=50 | N=100 | N=200 | N=400 | N=800 | N=1600 | N=3200 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| numba-dense | python | 0.18 | 0.000225 | 0.000355 | 0.00125 | 0.00238 | 0.00911 | 0.0382 | 0.137 |
| numba-kdtree | python | 0.000328 | 0.0002 | 0.000273 | 0.000761 | 0.000876 | 0.0016 | 0.00312 | 0.0062 |
| numpy-dense | python | 0.000294 | 0.0104 | 0.0103 | 0.0197 | 0.0146 | 0.0398 | 0.0878 | 0.347 |
| numpy-kdtree | python | 0.000582 | 0.000431 | 0.00063 | 0.00139 | 0.00146 | 0.00253 | 0.00457 | 0.00914 |
| openmm-compound | CPU | 12.2 | 166 | - | - | - | - | - | - |
| openmm-compound | Reference | 11.9 | 156 | - | - | - | - | - | - |
| openmm-cvforce | CPU | 0.519 | 1.61 | 3.06 | 6.35 | 13.4 | 29.5 | 64.2 | 149 |
| openmm-cvforce | Reference | 0.535 | 1.57 | 3.05 | 6.14 | 12.6 | 26.5 | 57.4 | 132 |
| openmm-frozen_density | CPU | 0.0326 | 0.034 | 0.0332 | 0.0339 | 0.0341 | 0.0365 | 0.0526 | 0.0641 |
| openmm-frozen_density | Reference | 0.0332 | 0.0389 | 0.0414 | 0.0405 | 0.0342 | 0.0385 | 0.0578 | 0.0539 |
| openmm-python[numba] | CPU | 0.000923 | 0.00104 | 0.0015 | 0.00227 | 0.00234 | 0.00408 | 0.00827 | 0.0125 |
| openmm-python[numba] | Reference | 0.000924 | 0.00124 | 0.00138 | 0.00255 | 0.00225 | 0.00369 | 0.00659 | 0.0125 |
| openmm-python[numpy] | CPU | 0.00141 | 0.0019 | 0.00238 | 0.00241 | 0.00333 | 0.00489 | 0.00968 | 0.0164 |
| openmm-python[numpy] | Reference | 0.00799 | 0.0211 | 0.029 | 0.056 | 0.143 | 0.24 | 0.585 | 1.52 |
| openmm-python[torch] | CPU | 0.00266 | 0.00376 | 0.00282 | 0.00579 | 0.00945 | 0.0269 | 0.129 | 0.418 |
| openmm-python[torch] | Reference | 0.00306 | 0.00524 | 0.00377 | 0.00643 | 0.0102 | 0.0312 | 0.123 | 0.449 |
| openmm-torchforce | CPU | 0.0293 | 0.0322 | 0.029 | 0.0316 | 0.035 | 0.0555 | 0.157 | 0.594 |
| openmm-torchforce | Reference | 0.0808 | 0.0225 | 0.0203 | 0.0231 | 0.0279 | 0.0536 | 0.23 | 0.72 |
| reference-loops | python | 0.00221 | 0.0101 | 0.0252 | 0.167 | 0.32 | 1.39 | 5 | - |
| torch | python | 0.0235 | 0.00227 | 0.00251 | 0.0062 | 0.00925 | 0.0405 | 0.1 | 0.473 |
