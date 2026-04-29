# QCE26 Numerics — QAOA Mixer Comparison for the TSP

Companion code for the IEEE Quantum Week 2026 submission.
Preprint: <https://arxiv.org/abs/2604.24297>

This repository reproduces the numerical experiments comparing three QAOA
mixer strategies on the Traveling Salesman Problem (TSP):

| Strategy    | Source prefix |
|-------------|---------------|
| Adjacency   | `adjacency`   |
| Bubble sort | `bubbles`     |
| Bit insert  | `binsert`     |

## Repository contents

| File pattern           | Purpose                                                         |
|------------------------|-----------------------------------------------------------------|
| `<strategy>.c`         | 4-city sanity check; verifies that the optimisation step works  |
| `<strategy>_9ex1.c`    | 9-city instance, example 1                                      |
| `<strategy>_9ex2.c`    | 9-city instance, example 2 — **this is the instance depicted in the paper** |
| `<strategy>_9ex3.c`    | 9-city instance, example 3                                      |

The simulation back-end is the [Orkan](https://github.com/Timo59/orkan)
quantum simulator (vendored as a submodule, pinned to v0.2.0). The classical
optimiser is [NLopt](https://github.com/stevengj/nlopt) (also vendored).

## Requirements

- CMake ≥ 3.27
- A C17-capable compiler (gcc 11+, clang 14+)
- OpenMP runtime
- ~2 GB of RAM for the 9-city experiments (24 qubits, 2²⁴ complex amplitudes)

## Build

```bash
git clone --recurse-submodules https://github.com/<owner>/qce26_numerics.git
cd qce26_numerics
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

If you forgot `--recurse-submodules` at clone time:

```bash
git submodule update --init --recursive
```

The build produces 12 executables in `build/`:

```
test_adjacency  adjacency_ex1  adjacency_ex2  adjacency_ex3
test_bubbles    bubbles_ex1    bubbles_ex2    bubbles_ex3
test_binsert    binsert_ex1    binsert_ex2    binsert_ex3
```

The `test_*` targets are the 4-city sanity checks, the `*_ex{1,2,3}` targets
are the 9-city instances.

## Running an experiment

Each executable takes no arguments. It builds the cost Hamiltonian, runs
COBYLA via NLopt on the QAOA objective, and prints the result to stdout.

```bash
./build/adjacency_ex2
```

Expected output:

```
Optimal tour cost:        29.0
Approximation ratio:      0.9655
Optimal parameters:       [0.7854, 0.7854, ..., ...]

History:
0       :  1.0827
1       :  1.0931
...
```

- **Optimal tour cost** — known optimum for the instance (used as denominator).
- **Approximation ratio** — final QAOA expectation value divided by the optimum.
  Lower is better; values close to 1 indicate the QAOA found near-optimal solutions.
- **Optimal parameters** — final mixer/separator angles produced by COBYLA.
- **History** — per-call convergence trace of the approximation ratio.

## Reproducing the figures in the paper

The instance plotted in the paper is example 2 (`*_9ex2`). To regenerate
all three mixer traces shown there:

```bash
./build/adjacency_ex2 > adjacency_ex2.log
./build/bubbles_ex2   > bubbles_ex2.log
./build/binsert_ex2   > binsert_ex2.log
```

The `History:` block in each log gives the convergence curve.

## Continuous integration

A GitHub Actions workflow (`.github/workflows/build.yml`) builds the project
on `ubuntu-latest` for every push and pull request to verify that the
submodule pins and build configuration remain reviewer-reproducible.

## License

AGPL-3.0-or-later. See [`LICENSE`](LICENSE).

This choice matches NLopt's licensing (the optimisation back-end). Orkan is
distributed separately under its own license; consult the upstream repository.
