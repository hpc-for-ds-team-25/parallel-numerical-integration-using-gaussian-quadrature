# Benchmarking & Scalability Analysis Guide

This guide defines the experimental methodology for **Section 4 (Performance and Scalability Analysis)** of the project report, aligned with Prof. Sandro Fiore's Lecture 11–12 instructions and the University of Trento PBS cluster setup.

---

## 1. Performance Metrics (Course Definitions)

Let $p$ be the total number of processing units (cores: $p = P_{\text{ranks}}$ for pure MPI, $p = T_{\text{threads}}$ for pure OpenMP, $p = R_{\text{ranks}} \times T_{\text{threads}}$ for Hybrid):

- **Serial execution time:** $T_{\text{serial}}$ — wall-clock time of the **standalone serial binary** (`quad_serial`, compiled with `-O3`, no MPI/OpenMP overhead).
  - *Note:* Always state explicitly whether speedup is computed against the true serial binary (`quad_serial`) or against $p = 1$ of the parallel binary. Using `quad_serial` is the gold standard; also checking $T_{\text{parallel}}(1) / T_{\text{serial}}$ quantifies parallelization overhead at $p = 1$.
- **Parallel execution time:** $T_{\text{parallel}}(p)$ — maximum wall-clock time across all MPI ranks (`MPI_MAX` after an `MPI_Barrier`).
- **Speedup:**
  $$S(p) = \frac{T_{\text{serial}}}{T_{\text{parallel}}(p)}$$
  *(Ideal linear speedup: $S(p) = p$.)*
- **Parallel Efficiency:**
  $$E(p) = \frac{S(p)}{p} = \frac{T_{\text{serial}}}{p \cdot T_{\text{parallel}}(p)}$$
  *(Ideal efficiency: $E(p) = 1.0$, i.e., $100\%$.)*
- **Parallel Overhead:**
  $$T_{\text{overhead}}(p) = T_{\text{parallel}}(p) - \frac{T_{\text{serial}}}{p}$$

### Theoretical Bounds to Discuss in the Report
- **Amdahl's Law (Strong Scaling):** If fraction $f_s$ of serial execution is non-parallelizable (e.g., Phase 1 node computation if serial, setup, final reduction):
  $$S_{\max}(p) \le \frac{1}{f_s + \frac{1 - f_s}{p}} \xrightarrow{p \to \infty} \frac{1}{f_s}$$
- **Gustafson's Law (Weak Scaling):** When problem size grows proportionally with $p$, the parallel portion dominates and scaled speedup grows linearly:
  $$S_{\text{scaled}}(p) = p - f_s (p - 1)$$

---

## 2. Exact Timing Protocol

Follow Lecture 11 & Note 11 strictly so all measurements are fair and comparable:

1. **What to time:**
   - **Primary metric (`t_total`):** Node/weight computation + domain integration + MPI reduction (everything needed to compute the answer).
   - **Breakdown metrics:** `t_nodes` (Phase 1) and `t_eval` (Phase 2 including reduction), plus optionally `t_comm` (pure reduction/dispatch time) to isolate communication overhead.
   - **Exclude:** CLI parsing, file I/O, `printf`, and post-run correctness checking.
2. **MPI timing pattern (never subtract timestamps across different ranks!):**
   ```c
   MPI_Barrier(MPI_COMM_WORLD);
   double t_start = MPI_Wtime();

   /* --- timed computation + reduction --- */

   double t_local = MPI_Wtime() - t_start;
   double t_elapsed = 0.0;
   MPI_Reduce(&t_local, &t_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
   ```
3. **OpenMP timing pattern:** Use `omp_get_wtime()` (high-resolution wall-clock timer).
4. **Serial timing pattern:** Use `clock_gettime(CLOCK_MONOTONIC, &ts)` or `gettimeofday(&tv, NULL)`.
5. **Statistical repetition:** Run every configuration **at least 5 times** (ideally **10 times** for final report numbers). Report:
   - **Mean** $\bar{T}$, **Standard Deviation** $\sigma$, and **Min / Median** (or coefficient of variation $\sigma / \bar{T}$).
   - If using `excl` nodes, residual variance reflects shared cluster network/storage interference.

---

## 3. Problem Size Calibration: Small, Medium, Large

### The Professor's "> 1 Second" Rule (Critical!)
In Lecture 11, Prof. Fiore explicitly warned that **benchmarks completing in `< 1 second` are unacceptable** because OS jitter, scheduling noise, and startup latency overwhelm real scaling behavior.

Calibrate your **Small, Medium, and Large** problem sizes empirically on **1 core of the cluster** before launching the full benchmark sweep:

| Dataset Tier | Target Serial Time $T_{\text{serial}}$ | Target Parallel Time at Max $p$ (e.g., 16–32 cores) | Example 3D Grid ($n = 8$, `heavy_3d`) | Example 1D Composite ($n = 16$, `osc_1d`) |
| --- | --- | --- | --- | --- |
| **Small** | ~15 – 30 s | ~1 – 2 s (exposes overhead at high $p$) | Adjust $M_x = M_y = M_z$ (e.g., 64–80) | Adjust $M$ (e.g., $2 \times 10^7$) |
| **Medium** | ~60 – 120 s | ~3 – 8 s (balanced strong scaling) | Adjust $M_x = M_y = M_z$ (e.g., 100–128) | Adjust $M$ (e.g., $1 \times 10^8$) |
| **Large** | ~3 – 6 min | ~10 – 25 s (clean high-$p$ efficiency) | Adjust $M_x = M_y = M_z$ (e.g., 160–200) | Adjust $M$ (e.g., $4 \times 10^8$) |

> **Action before full benchmarking:** Run `quad_serial` once on the cluster with a few values of $M$ to measure exact evaluations/second on the cluster CPU, then lock in the exact Small / Medium / Large parameters and never change them mid-study!

---

## 4. Experimental Suite Design

### Experiment A — Strong Scalability (Fixed Problem Size, Increasing $p$)
- Fix problem size at **Small**, **Medium**, and **Large**.
- Vary core count: $p \in \{1, 2, 4, 8, 16, 32, \dots\}$ (up to what your cluster queue allows).
- Run for:
  1. Serial baseline ($p = 1$)
  2. Pure MPI — Single-node (`place=pack:excl`) up to node core capacity
  3. Pure MPI — Multi-node (`place=scatter:excl`) across 2, 4, ... nodes
  4. Pure OpenMP — Single-node (`place=pack:excl`)
  5. Hybrid MPI + OpenMP — Multi-node (e.g., 1 rank/node × $T$ threads/rank vs. 2 ranks/node × $T/2$ threads/rank)

### Experiment B — Weak Scalability (Fixed Work per Core, Increasing $p$)
- Fix work per core $W_0$ (e.g., choose $W_0$ so $p = 1$ takes ~10–15 seconds).
- Scale total panels linearly with $p$:
  - *1D:* $M(p) = p \cdot M_0$
  - *3D:* Scale one axis ($M_x(p) = p \cdot M_0$, keeping $M_y, M_z$ fixed) or scale total cells $M_{\text{total}}(p) = p \cdot M_{\text{total}, 0}$.
- Compute weak-scaling efficiency:
  $$E_{\text{weak}}(p) = \frac{T(1)}{T(p)}$$
- Ideal behavior: $T(p)$ remains flat as $p$ grows ($E_{\text{weak}}(p) \approx 1$).

### Experiment C — Single-Node (`pack:excl`) vs. Multi-Node (`scatter:excl`) Placement Comparison
- Hold total cores $p$ fixed (e.g., $p = 4, 8, 16$) and compare:
  - All ranks packed on 1 exclusive node (`place=pack:excl`)
  - Ranks scattered across multiple exclusive nodes (`place=scatter:excl`)
- Analyze whether inter-node network latency affects Gaussian quadrature (for static composite, communication is tiny so `pack` vs. `scatter` should be nearly identical — whereas for dynamic master–worker MPI, inter-node latency becomes clearly visible!).

### Experiment D — Load-Balancing & Strategy Comparison (Irregular / Adaptive Workload)
- Run on `peak_1d` (localized sharp peak) or adaptive quadrature:
  - Compare **MPI Static Block** vs. **MPI Static Cyclic** vs. **MPI Dynamic Master–Worker** (varying chunk size $B$).
  - Compare OpenMP `schedule(static)` vs. `schedule(dynamic, chunk)` vs. `schedule(guided, chunk)` (and coarse-grain vs. fine-grain).
- This directly demonstrates *why* choosing the right parallel strategy matters beyond uniform workloads.

---

## 5. PBS Cluster Execution Templates (UniTN HPC4DS Cluster)

From Lecture 11 (Slides 29–34) and Notes 06 & 09, the course cluster environment uses:
- Queue: `#PBS -q shortCPUQ` (check cluster limits for longer Large runs if a longer queue exists)
- MPI module: `module load MPICH/3.4.2-GCC-10.3.0`
- Network wrapper: `./mpi_wrapper.sh` (sets `UCX_TLS=tcp,self,sm` and `UCX_NET_DEVICES="$IFACE"`)

### Template 1 — Single-Node Pure MPI Benchmark (`pack:excl`)
```bash
#!/bin/bash
#PBS -N quad_mpi_pack
#PBS -q shortCPUQ
#PBS -l select=1:ncpus=8:mpiprocs=8:mem=4gb
#PBS -l place=pack:excl
#PBS -l walltime=00:15:00

cd "$PBS_O_WORKDIR"
module load MPICH/3.4.2-GCC-10.3.0

mpiexec -n 8 ./scripts/mpi_wrapper.sh ./bin/quad_mpi_static \
    --mode 3d --func heavy_3d --order 8 --panels 128 --repeats 5
```

### Template 2 — Multi-Node Pure MPI Benchmark (`scatter:excl`)
```bash
#!/bin/bash
#PBS -N quad_mpi_scatter
#PBS -q shortCPUQ
#PBS -l select=4:ncpus=4:mpiprocs=4:mem=4gb
#PBS -l place=scatter:excl
#PBS -l walltime=00:15:00

cd "$PBS_O_WORKDIR"
module load MPICH/3.4.2-GCC-10.3.0

mpiexec -n 16 ./scripts/mpi_wrapper.sh ./bin/quad_mpi_static \
    --mode 3d --func heavy_3d --order 8 --panels 128 --repeats 5
```

### Template 3 — Single-Node Pure OpenMP Benchmark (`pack:excl`)
```bash
#!/bin/bash
#PBS -N quad_omp
#PBS -q shortCPUQ
#PBS -l select=1:ncpus=8:mem=4gb
#PBS -l place=pack:excl
#PBS -l walltime=00:15:00

cd "$PBS_O_WORKDIR"
export OMP_NUM_THREADS=8
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_SCHEDULE="static"

./bin/quad_omp --mode 3d --func heavy_3d --order 8 --panels 128 --repeats 5
```

### Template 4 — Multi-Node Hybrid MPI + OpenMP Benchmark (`scatter:excl`)
Example: 4 nodes × 1 MPI rank/node × 4 OpenMP threads/rank = 16 total cores:
```bash
#!/bin/bash
#PBS -N quad_hybrid
#PBS -q shortCPUQ
#PBS -l select=4:ncpus=4:mpiprocs=1:ompthreads=4:mem=4gb
#PBS -l place=scatter:excl
#PBS -l walltime=00:15:00

cd "$PBS_O_WORKDIR"
module load MPICH/3.4.2-GCC-10.3.0

export OMP_NUM_THREADS=4
export OMP_PROC_BIND=close
export OMP_PLACES=cores

mpiexec -n 4 ./scripts/mpi_wrapper.sh ./bin/quad_hybrid \
    --mode 3d --func heavy_3d --order 8 --panels 128 --repeats 5
```

> **Verification tip:** Print `rank`, `hostname`, `omp_get_max_threads()`, and `sched_getcpu()` (optional) once at startup (outside the timed region!) to confirm your PBS allocation actually placed ranks and threads where intended!

---

## 6. Plots & Tables to Include in Report Section 4

To produce a thorough, convincing ≥ 10-page report, plan the following figures and tables:

1. **Hardware & Software Environment Table:** Cluster CPU model, cores/node, RAM, interconnect, compiler & flags (`gcc`/`mpicc -O3`), MPICH version, PBS queue & placement settings.
2. **Numerical Validation Table:** Computed integral vs. exact analytical integral and absolute/relative error across Serial, MPI, OpenMP, and Hybrid (confirming accuracy down to ~$10^{-13}$–$10^{-15}$).
3. **Strong Scaling Plots (3 curves per plot: Small, Medium, Large):**
   - Execution Time $T(p)$ vs. $p$ (with error bars $\pm\sigma$).
   - Speedup $S(p)$ vs. $p$ (with dashed ideal line $S = p$).
   - Efficiency $E(p)$ vs. $p$ (with dashed ideal line $E = 1.0$).
4. **Weak Scaling Plot:** Execution Time $T(p)$ and Efficiency $E_{\text{weak}}(p)$ vs. $p$.
5. **Single-Node (`pack:excl`) vs. Multi-Node (`scatter:excl`) Comparison Chart:** Grouped bar chart or overlaid speedup curves showing the cost (or near-absence) of inter-node communication.
6. **Paradigm Comparison Chart (at fixed total cores $p$):** Pure MPI vs. Pure OpenMP (single node) vs. Hybrid MPI+OpenMP (testing different $R \times T$ splits such as $16 \times 1$, $8 \times 2$, $4 \times 4$, $2 \times 8$).
7. **Irregular / Adaptive Workload Comparison:** Static Block vs. Static Cyclic vs. Dynamic Master–Worker (MPI) and `static` vs. `dynamic` vs. `guided` (OpenMP), showing load imbalance percentage $(\max T_i - \bar{T}) / \bar{T}$ per rank/thread.
8. **Critical Discussion:** Explain every deviation from ideal speedup (e.g., turbo-boost frequency drop when all cores are active, NUMA effects, communication vs. computation ratio, Amdahl serial fraction, variance across repeated runs).
