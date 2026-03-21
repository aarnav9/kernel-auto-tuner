# Metal GEMM Autotuner

Swift + Metal autotuning project for custom GEMM kernels on Apple Silicon.

The repo started from Apple's `PerformingCalculationsOnAGPU` sample and grew into a small benchmarking and search framework for answering a more specific question:

- when do custom Metal kernels, chosen by an autotuner, match or beat `MPSMatrixMultiplication`?

Right now the strongest results are on plain GEMM, especially for edge-heavy shapes on an M2.

### Layout

`Sources/MetalHello/Resources/ComputeKernels.metal`: vector add, naive matmul, scalar tiled GEMM, scalar fused kernels

`Sources/MetalHello/Resources/SimdgroupMatrixKernels.metal`: Apple-specific SIMD-group matrix kernels compiled when supported

`Sources/MetalHello/MetalComputeRunner.swift`: Metal setup, dispatch, correctness checks, benchmark fixtures, MPS baselines

`Sources/MetalHello/SimulatedAnnealing.swift`: simulated annealing search over the validated config space

`Sources/MetalHello/ResultsPackaging.swift`: CSV, markdown, and SVG report generation

`Sources/MetalHello/main.swift`: runs correctness, exhaustive sweeps, confidence reruns, shape hunt, and strict validation

`PerformingCalculationsOnAGPU/`: local Apple reference sample used during development

### Commands

```bash
swift run MetalHello
```

### Functionality

Currently the project supports:

- vector add validation
- plain GEMM validation
- fused `matmul + bias + ReLU` validation
- scalar tiled GEMM kernels with multiple tile and threadgroup shapes
- SIMD-group matrix kernels using:
  - `simdgroup_float8x8`
  - `simdgroup_load`
  - `simdgroup_multiply_accumulate`
  - `simdgroup_store`
- deterministic exhaustive tuning over validated configs
- simulated annealing over the same measured objective table
- confidence reruns for selected winners
- extra shape-hunt sweeps against `MPS`
- strict win validation with more passes and more timing iterations

### Methods

The project uses two kernel families for plain GEMM:

- scalar tiled kernels
- SIMD-group matrix kernels

The scalar tiled family explores:

- `tileM`
- `tileN`
- `tileK`
- `threadsX`
- `threadsY`

The SIMD-group family builds `16x16` output tiles from `8x8` matrix fragments and is the main reason the repo is interesting on Apple Silicon. On the current M2 runs, that family produced the strict wins over `MPS`.

The search logic has two stages:

- exhaustive search
- simulated annealing

Exhaustive search is the ground-truth baseline. It benchmarks every validated config for a target shape and chooses the best one using median latency plus a stability-aware penalty.

Simulated annealing is evaluated against that same measured table, so its rank, regret, and score gap are directly comparable to exhaustive search.

### Benchmark Protocol

The repo does not trust one fast sweep.

Each target uses:

- fixed input tensors reused across configs
- warmup iterations before timing
- repeated benchmark samples
- median latency as the base score
- a stability-aware tuning score to penalize noisy runs

After the main sweep, the repo runs:

- confidence reruns for the selected custom winner and `MPS`
- shape-hunt targets for extra tail workloads
- a stricter validation pass for the strongest candidates

This is important because some apparent wins disappear under stricter reruns.

### Results

The strongest part of the project today is the plain GEMM path, not the fused path.

On the latest strict validation pass on an Apple M2:

- `edge_square`: `sg_m16n16k8_t32x4` ran at `0.238 ms` vs `MPS` at `0.263 ms`
- `hunt_edge129`: `sg_m16n16k8_t8x16` ran at `0.241 ms` vs `MPS` at `0.268 ms`

Those are the safest numbers to cite because they survived the stricter rerun protocol.

The main result files are:

- [`results/latest_results.md`](results/latest_results.md)
- [`results/confirmed_wins.csv`](results/confirmed_wins.csv)
- [`results/shape_hunt.csv`](results/shape_hunt.csv)
- [`results/tuning_sweep.csv`](results/tuning_sweep.csv)
- [`results/fused_tuning_sweep.csv`](results/fused_tuning_sweep.csv)
- [`results/sa_summary.csv`](results/sa_summary.csv)
- [`results/sa_history.csv`](results/sa_history.csv)
- [`results/plain_vs_mps.svg`](results/plain_vs_mps.svg)
- [`results/sa_regret.svg`](results/sa_regret.svg)

### Sample Output

Typical output includes:

- Metal device and capability notes
- whether the SIMD-group matrix path was enabled
- correctness results
- deterministic tuning sweep results
- simulated annealing summaries
- shape-hunt summaries
- strict win validation summaries

### Notes

- `MPSMatrixMultiplication` is the main baseline in this project, not the CPU loop.
- The fused `matmul + bias + ReLU` path is still exploratory and is not the headline result.
- The strict validation pass is the right place to look before making public performance claims.
- If runtime Metal shader compilation fails, install full Xcode and rerun.
