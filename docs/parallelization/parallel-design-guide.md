# Parallel Design & Implementation Guide (MPI → OpenMP → Hybrid)

This guide walks through the parallelization of Gaussian quadrature step-by-step, following the course methodology (Foster's PCAM design, Lecture 16 data-dependency analysis, MPI collectives, OpenMP work-sharing/scheduling, and Hybrid MPI+OpenMP).

Following your roadmap, we structure development into **three progressive stages**:
1. **Stage 1 — Distributed Memory (Pure MPI):** Start here to enable multi-node execution across the cluster.
2. **Stage 2 — Shared Memory (Pure OpenMP):** Add intra-node multithreading and compare scheduling policies & granularity.
3. **Stage 3 — Hybrid (MPI + OpenMP):** Combine inter-node MPI with intra-node OpenMP threads to reduce MPI communication/memory overhead and compare all three paradigms head-to-head.

---

## 1. Parallelism & Data-Dependency Analysis (For Report Section 2)

Before writing parallel code, apply the **3-step data-dependency methodology** (Lecture 16 / Note 16) to the sequential algorithm:

### Phase 1 — Node & Weight Computation ($O(n^2)$)
- Each root index $i \in \{0, \dots, \lfloor (n+1)/2 \rfloor - 1\}$ runs an independent Newton–Raphson loop and writes to distinct slots `nodes[i]`, `nodes[n-1-i]`, `weights[i]`, `weights[n-1-i]`.
- **Dependencies:** None across outer loop iterations `i` (the inner recurrence over $k = 1 \dots n$ has a **loop-carried flow dependence** $P_k \leftarrow P_{k-1}, P_{k-2}$ and must stay sequential per root).
- **Parallelization decision:** For typical composite quadrature orders ($n \le 64$), $O(n^2)$ takes microseconds — **recomputing** the small $(n)$ arrays locally on every MPI rank avoids any `MPI_Bcast` latency and keeps memory minimal. Only for very high single-panel orders ($n \ge 5\,000$) does parallelizing Phase 1 across roots pay off (using `MPI_Allgather` or `#pragma omp parallel for`).

### Phase 2 — Quadrature Evaluation ($O(M \cdot n)$ or $O(M^3 \cdot n^3)$)
Inspecting the composite loop over panels $k \in \{0, \dots, M-1\}$:

| Variable | Access pattern across panel iterations $k$ | Dependency type | Resolution |
| --- | --- | --- | --- |
| `rule.nodes[]`, `rule.weights[]`, `a`, `h` | Read-only | **None** | `shared` (OpenMP) / replicated locally (MPI) |
| `mid`, `x`, `panel_sum` | Written then read within same iteration $k$ | **Output / anti dependence** if shared across threads | **Privatization** (declare inside loop body / `private`) |
| `total` | Read & updated (`total += panel_val`) in every iteration | **Loop-carried flow dependence** | **Reduction** (`MPI_Reduce(..., MPI_SUM)` / `reduction(+:total)`) |

**Key takeaway:** Because every panel $k$ evaluates $f$ independently and only combines into a single scalar sum via an associative operator (`+`), composite Gaussian quadrature is **embarrassingly parallel** with an $O(1)$ reduction payload (1 `double` = 8 bytes per process!).

---

## 2. Foster's PCAM Design Methodology

Applying the same four-step design used for the Trapezoidal Rule (Lecture 5 / Note 05):

1. **Partitioning:**
   - *1D Composite:* Fine-grained task = evaluating one panel $k$ (cost $n \cdot C_f$).
   - *3D Tensor-Product:* Fine-grained task = evaluating one 3D cell $(k_x, k_y, k_z)$ (cost $n^3 \cdot C_f$), or collapsing the 3D grid into a 1D index `idx` $\in [0, M_x M_y M_z - 1]$.
   - *Adaptive:* Task = refining a subinterval $[l, r]$ until local error $\le \text{tol}$.
2. **Communication:**
   - *Static composite:* Zero inter-process communication during evaluation; **one** global reduction (`MPI_Reduce` with `MPI_SUM`) at the end ($O(\log p)$ latency, 8 bytes/message).
   - *Dynamic / Adaptive:* Point-to-point task dispatch and partial-sum collection between manager and workers (or distributed work-stealing).
3. **Agglomeration:**
   - Group contiguous panels into large chunks of size $\approx M / p$ (static) or tunable chunks of size $B$ (dynamic) to amortize scheduling/communication overhead and preserve spatial locality.
4. **Mapping:**
   - *Pure MPI:* 1 MPI rank per CPU core (`pack:excl` on 1 node, `scatter:excl` across nodes).
   - *Pure OpenMP:* 1 process with $T$ threads pinned to $T$ cores on 1 node.
   - *Hybrid MPI+OpenMP:* 1 MPI rank per node (or per NUMA socket) spawning $T$ OpenMP threads across the node's cores.

---

## 3. Stage 1 — Pure MPI Implementation

Start here first, verify correctness against the serial baseline, and benchmark single-node and multi-node scalability before introducing OpenMP.

### Strategy 1A: Static Block Decomposition (Primary MPI Version)

**Why avoid `MPI_Scatter` here?**  
Unlike matrix–vector multiplication where rank 0 holds a large input array in memory, numerical integration defines its domain analytically by `(a, b, M, order)`. Every rank computes its exact panel slice directly from `(rank, size)` — **eliminating scatter communication and root memory bottlenecks!**

#### Handling Any $M$ (Including Non-Divisible $M \bmod p \ne 0$)
Use balanced block partitioning so ranks differ by at most 1 panel:

```c
long long base = M / size;
long long rem  = M % size;

long long local_M = base + (rank < rem ? 1 : 0);
long long start_k = rank * base + (rank < rem ? rank : rem);
long long end_k   = start_k + local_M; /* exclusive */
```

#### Control Flow (Static MPI)
```text
1. MPI_Init(&argc, &argv); get rank, size.
2. Parse CLI arguments (all ranks parse argv, OR rank 0 parses & MPI_Bcast config struct).
3. MPI_Barrier(MPI_COMM_WORLD);
   t_start = MPI_Wtime();
4. Compute Gauss-Legendre nodes & weights locally (order n) -> t_nodes.
5. Compute [start_k, end_k) for this rank.
   (For 3D: flatten total_cells = Mx * My * Mz, partition [start_cell, end_cell),
    and decode kx = idx / (My * Mz), ky = (idx / Mz) % My, kz = idx % Mz.)
6. Evaluate local panels into local_sum (using Kahan summation).
7. MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
8. t_local = MPI_Wtime() - t_start;
   MPI_Reduce(&t_local, &t_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
9. Rank 0 checks error against exact integral & prints timing metrics.
10. Free memory & MPI_Finalize().
```

- **Communication cost:** $O(\log p)$ tree reduction of 1 `double` (plus optional `MPI_Bcast` of config).
- **Pros:** Minimal overhead, near-ideal strong & weak scalability on uniform integrands, zero load-balancing latency.
- **Cons:** If the integrand is adaptive or has position-dependent evaluation cost (e.g., sharp peak at $x = 0.2$), one rank may do 90% of the work while others sit idle.

---

### Strategy 1B: Dynamic Master–Worker / Task Pool (Second MPI Strategy for Comparative Analysis)

Implementing a second MPI strategy gives you rich comparative material for **Section 2 & Section 3** (explicitly rewarded in Slide 7 & Slide 8!).

**Use case:** Adaptive Gauss–Legendre on localized-peak integrands (`peak_1d`) or workloads with uneven per-panel cost.

#### Design Pattern
- **Rank 0 (Master / Coordinator):**
  - Splits $[a, b]$ into $K \gg p$ initial coarse work units (or dynamically maintains a queue of unrefined subintervals).
  - Sends one chunk `(l, r)` to each worker rank $1 \dots p-1$ (`TAG_WORK`).
  - Loops while work remains or workers are busy:
    - Receives `(partial_sum, refined_subintervals_if_any)` from `MPI_ANY_SOURCE`.
    - Adds `partial_sum` to `global_sum` (or pushes split children back to the work queue if adaptive).
    - Dispatches the next work chunk to that worker, or sends a termination signal (`TAG_STOP`) when the queue is empty.
- **Ranks $1 \dots p-1$ (Workers):**
  - Loop receiving `(l, r)` from rank 0; if `status.MPI_TAG == TAG_STOP`, break.
  - Integrate chunk (or run local adaptive refinement up to a bounded budget of steps) and `MPI_Send` result back to rank 0.

**Important MPI trade-offs to discuss in the report:**
1. **Dedicated master overhead:** With $p$ ranks, only $p - 1$ ranks compute (at $p = 2$, only 1 worker runs $\Rightarrow$ no speedup!). Only pays off at larger $p$ when load imbalance exceeds $1/p$.
2. **Chunk granularity ($B$):**
   - Too small $\Rightarrow$ master becomes a communication bottleneck (excessive message latency).
   - Too large $\Rightarrow$ tail latency / load imbalance at the end of the run.
3. **Non-blocking optimization (optional, connects to Lecture 12):** Workers can double-buffer with `MPI_Irecv` / `MPI_Isend` to overlap communication with local integration.
4. **Alternative — Static Cyclic Distribution:** Assigning panel $k$ to rank `k % size` (strided/cyclic) requires **zero communication** like Strategy 1A while spreading localized hotspots across all ranks! Comparing **Static Block vs. Static Cyclic vs. Dynamic Master–Worker** on `peak_1d` makes a standout experiment.

---

## 4. Stage 2 — Pure OpenMP Implementation (Single-Node Shared Memory)

Before building the hybrid version, build a standalone OpenMP implementation (`main_omp.c`) so you can:
1. Benchmark pure OpenMP vs. pure MPI on a **single node** (`pack:excl`).
2. Study OpenMP-specific optimizations covered in Lectures 14–16.

### Fine-Grain Parallelization (`#pragma omp parallel for`)

```c
double total = 0.0;

#pragma omp parallel for default(none) \
    shared(a, h, half_h, M, rule, func) \
    reduction(+:total) \
    schedule(runtime)
for (long long k = 0; k < M; k++) {
    double mid = a + (k + 0.5) * h;
    double panel_sum = 0.0;
    for (int i = 0; i < rule.n; i++) {
        double x = mid + half_h * rule.nodes[i];
        panel_sum += rule.weights[i] * func(x);
    }
    total += half_h * panel_sum;
}
```

- Using `schedule(runtime)` lets you benchmark `OMP_SCHEDULE="static"`, `"static,64"`, `"dynamic,256"`, and `"guided,64"` **without recompiling** (Lecture 15 / Note 15).
- Always specify `default(none)` so the compiler catches any accidental shared variable / race condition (Lecture 14 / Note 14).
- For 3D nested loops (`kx`, `ky`, `kz`), either flatten into a single loop or use `collapse(3)` (or `collapse(2)` on the outer two loops) to expose enough iterations for large thread counts.

### Coarse-Grain Parallelization (Lecture 16 / Note 16)
Instead of relying on OpenMP's built-in loop reduction (which uses plain `+`), open a single `#pragma omp parallel` region, partition `[0, M)` manually by `tid = omp_get_thread_num()` and `nthreads = omp_get_num_threads()`, accumulate each thread's chunk with **thread-local Kahan summation**, and combine once at the end via `reduction(+:total)` or `#pragma omp atomic`:

```c
double total = 0.0;
#pragma omp parallel default(none) shared(a, h, half_h, M, rule, func) reduction(+:total)
{
    int tid = omp_get_thread_num();
    int nt  = omp_get_num_threads();
    long long base = M / nt, rem = M % nt;
    long long start = tid * base + (tid < rem ? tid : rem);
    long long end   = start + base + (tid < rem ? 1 : 0);

    double local_sum = 0.0, comp = 0.0;
    for (long long k = start; k < end; k++) {
        /* evaluate panel k and Kahan-accumulate into local_sum */
    }
    total += local_sum;
}
```
**Why compare fine-grain vs. coarse-grain?**
- Coarse-grain enables per-thread Kahan summation, guarantees zero false sharing, and minimizes runtime scheduling overhead.
- Fine-grain with `schedule(dynamic)` / `schedule(guided)` wins when per-panel work is irregular (adaptive or non-uniform integrand).

### Avoiding False Sharing
Never write thread results into adjacent elements of a shared array like `partial[tid] += val;` inside the inner loop (cache-line bouncing across cores!). Use a local register/stack scalar `local_sum` and reduce once outside the loop.

---

## 5. Stage 3 — Hybrid MPI + OpenMP Implementation

Now combine **MPI across nodes (or sockets)** with **OpenMP threads inside each MPI rank**.

### Architecture: Two-Level Parallelism
```text
Global domain [0, M)
  ├── Level 1 (MPI): partitioned into P rank ranges [start_r, end_r) across nodes/sockets
  │     └── Level 2 (OpenMP): each rank spawns T threads over [start_r, end_r)
  │           ├── Thread-local Kahan accumulation -> OpenMP reduction -> rank local_sum
  └── Final MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD)
```

### Thread Safety: `MPI_Init_thread`
Always initialize hybrid MPI programs with `MPI_Init_thread` instead of `MPI_Init`:

```c
int provided;
MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
if (provided < MPI_THREAD_FUNNELED) {
    fprintf(stderr, "Error: MPI implementation does not support MPI_THREAD_FUNNELED\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
}
```

**Why `MPI_THREAD_FUNNELED`?**
- Only the **master thread** calls MPI routines (`MPI_Barrier`, `MPI_Bcast`, `MPI_Reduce`, `MPI_Wtime`) **outside** the `#pragma omp parallel` region.
- `MPI_THREAD_FUNNELED` has virtually zero locking overhead compared to `MPI_THREAD_MULTIPLE` and is universally supported by MPICH (`MPICH/3.4.2-GCC-10.3.0`).

### Compilation
```bash
mpicc -O3 -std=c11 -fopenmp -Wall -Wextra -o bin/quad_hybrid \
    src/hybrid/main_hybrid.c src/common/gauss_legendre.c src/common/integrands.c -lm
```

### Does Hybrid Always Beat Pure MPI? (Crucial for Critical Analysis!)
Do **not** assume hybrid is automatically faster — measure and analyze *why* one wins over the other on the cluster!

| Factor | Pure MPI ($P$ ranks × 1 thread) | Hybrid ($R$ ranks × $T$ threads, $R \cdot T = P$) |
| --- | --- | --- |
| **MPI reduction tree depth** | $\lceil \log_2 P \rceil$ | $\lceil \log_2 R \rceil$ (fewer inter-process messages) |
| **Memory footprint** | $P$ separate address spaces & MPI buffers | $R$ address spaces (less buffer/table duplication) |
| **Intra-node load balancing** | Static unless explicit MPI dynamic messaging | Easy dynamic load balancing via `schedule(dynamic)` / `guided` with zero MPI messages |
| **NUMA / thread overhead** | Each rank pinned locally to its core/socket | Spanning 1 rank across multiple NUMA sockets can hurt if memory/threads aren't bound (`OMP_PROC_BIND`) |
| **Expected outcome on uniform composite quadrature** | Very strong (communication is only 1 scalar `MPI_Reduce`!) | Comparable or slightly faster at high node counts (fewer MPI collectives), or slightly slower if thread fork/join overhead is noticeable on small problems |
| **Expected outcome on irregular / adaptive quadrature** | Needs complex dynamic MPI messaging | **Big win:** coarse static MPI split (or cyclic) + intra-node OpenMP `dynamic`/`guided` balances work cleanly with minimal overhead |

---

## 6. Summary Comparison Matrix (For Report Section 2 & 3)

| Strategy | Paradigm | Load Balance (Uniform) | Load Balance (Irregular/Adaptive) | Communication Overhead | Implementation Complexity |
| --- | --- | --- | --- | --- | --- |
| **Serial Baseline** | C | N/A | N/A | None | Low |
| **1A: MPI Static Block** | MPI | Excellent | Poor (hotspots) | Minimal ($O(\log p)$ scalar reduce) | Low |
| **1B: MPI Static Cyclic** | MPI | Excellent | Good (spreads hotspots) | Minimal ($O(\log p)$ scalar reduce) | Low |
| **1C: MPI Dynamic Master–Worker** | MPI | Good ($p-1$ workers) | Very Good (chunk-tunable) | Medium–High (point-to-point messages) | Medium–High |
| **2: OpenMP (`static`/`dynamic`/`guided`)** | OpenMP | Excellent | Excellent (`dynamic`/`guided`) | None (shared memory; single-node limit) | Low–Medium |
| **3: Hybrid MPI + OpenMP** | MPI + OpenMP | Excellent | Very Good–Excellent | Lowest inter-node ($O(\log R)$) | Medium |
