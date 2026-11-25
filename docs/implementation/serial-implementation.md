# Serial Implementation Guide (C Baseline)

This document describes how to design, implement, validate, and time the **sequential C implementation** of Gaussian quadrature. The serial code serves two critical purposes:
1. It is the **correctness reference** (verified against known analytical integrals).
2. It provides the **$T_{\text{serial}}$ baseline** for all speedup and efficiency calculations ($S = T_{\text{serial}} / T_{\text{parallel}}$).

> **Important:** The serial baseline must be **fair and well-optimized** (compiled with `-O3`, using efficient loops and no redundant work inside inner loops). Comparing a parallel version against a deliberately slow serial baseline invalidates the benchmark analysis.

---

## 1. Recommended Repository Code Layout

Keep shared math logic separate from the execution drivers so serial, MPI, OpenMP, and hybrid versions execute the **exact same integrand and quadrature kernel**:

## 2. Core Data Structures

```c
typedef struct {
    int n;           /* Quadrature order (number of Gauss-Legendre points per panel) */
    double *nodes;   /* Abscissas xi_i in (-1, 1), length n */
    double *weights; /* Positive weights w_i, length n, sum(w_i) = 2 */
} GaussRule;

typedef struct {
    double a, b;          /* Integration interval [a, b] */
    long long num_panels; /* Number of subintervals M (use 64-bit integer for large M) */
    int order;            /* Quadrature points per panel n */
    int func_id;          /* Selected benchmark integrand */
} IntegrationConfig;
```

**Why `long long` (or `uint64_t`) for `num_panels`?**  
For large 1D benchmarks, $M$ can approach or exceed $10^9$ (near the 32-bit signed `int` limit of $2.14 \times 10^9$). Using 64-bit loop indices prevents silent integer overflow.

---

## 3. Step-by-Step Serial Algorithm & Pseudocode

### Step 1 — Compute Gauss–Legendre Nodes and Weights (`gauss_legendre_init`)
Use Newton–Raphson on the Legendre polynomial $P_n(x)$ exploiting symmetry around $x = 0$:

```text
function gauss_legendre_init(n, nodes, weights):
    m = (n + 1) / 2
    for i = 0 to m - 1:
        // Asymptotic initial guess for the (i+1)-th root
        x = cos(PI * (i + 0.75) / (n + 0.5))

        repeat:
            // Three-term recurrence for P_n(x) and P_{n-1}(x)
            p_prev = 0.0      // P_{k-2}
            p_curr = 1.0      // P_{k-1}
            for k = 1 to n:
                p_next = ((2*k - 1) * x * p_curr - (k - 1) * p_prev) / k
                p_prev = p_curr
                p_curr = p_next
            // Now p_curr = P_n(x), p_prev = P_{n-1}(x)

            // Derivative P_n'(x)
            dp = n * (x * p_curr - p_prev) / (x * x - 1.0)

            dx = p_curr / dp
            x  = x - dx
        until abs(dx) < 1e-15

        // Exploit symmetry: roots in (-1, 1) are symmetric around 0
        nodes[i]         = -x
        nodes[n - 1 - i] =  x
        w = 2.0 / ((1.0 - x * x) * dp * dp)
        weights[i]         = w
        weights[n - 1 - i] = w
```

**Verification checks for Step 1:**
- For any $n \ge 1$, verify $\left|\sum_{i=0}^{n-1} w_i - 2.0\right| < 10^{-14}$.
- Verify symmetry: `nodes[i] == -nodes[n - 1 - i]` and `weights[i] == weights[n - 1 - i]`.
- Verify exactness on $f(x) = x^{2n - 2}$ over $[-1, 1]$ (exact value $\frac{2}{2n - 1}$).

---

### Step 2 — 1D Composite Gauss–Legendre Quadrature

```text
function integrate_1d_composite(f, a, b, M, rule):
    h = (b - a) / M
    half_h = 0.5 * h
    total = 0.0
    comp  = 0.0   // Kahan compensation term (see Section 4)

    for k = 0 to M - 1:
        mid = a + (k + 0.5) * h
        panel_sum = 0.0
        for i = 0 to rule.n - 1:
            x = mid + half_h * rule.nodes[i]
            panel_sum += rule.weights[i] * f(x)

        // Accumulate scaled panel contribution using Kahan summation
        y = (half_h * panel_sum) - comp
        t = total + y
        comp = (t - total) - y
        total = t

    return total
```

**Micro-optimization notes:**
- Computing `mid = a + (k + 0.5) * h` instead of accumulating `x_k += h` avoids floating-point drift across millions of panels AND makes iteration `k` completely independent of iteration `k - 1` (crucial for parallelization!).
- Keeping `nodes` and `weights` in small contiguous arrays (`n * 8` bytes, e.g., 128 bytes for $n = 16$) guarantees they stay resident in **L1 CPU cache** for the entire run.

---

### Step 3 — 3D Tensor-Product Composite Gauss–Legendre (Recommended HPC Workload)

For $\Omega = [a_x, b_x] \times [a_y, b_y] \times [a_z, b_z]$ split into $M_x \times M_y \times M_z$ cells with $n$ points per axis:

```text
function integrate_3d_composite(f3d, bounds, Mx, My, Mz, rule):
    hx = (bx - ax) / Mx;  half_hx = 0.5 * hx
    hy = (by - ay) / My;  half_hy = 0.5 * hy
    hz = (bz - az) / Mz;  half_hz = 0.5 * hz
    cell_scale = half_hx * half_hy * half_hz
    total = 0.0

    for kx = 0 to Mx - 1:
        mx = ax + (kx + 0.5) * hx
        for ky = 0 to My - 1:
            my = ay + (ky + 0.5) * hy
            for kz = 0 to Mz - 1:
                mz = az + (kz + 0.5) * hz
                cell_sum = 0.0
                for ix = 0 to n - 1:
                    x = mx + half_hx * rule.nodes[ix]
                    wx = rule.weights[ix]
                    for iy = 0 to n - 1:
                        y = my + half_hy * rule.nodes[iy]
                        wxy = wx * rule.weights[iy]
                        for iz = 0 to n - 1:
                            z = mz + half_hz * rule.nodes[iz]
                            cell_sum += wxy * rule.weights[iz] * f3d(x, y, z)
                total += cell_scale * cell_sum
    return total
```

**Why include a 3D variant?**
- Every cell $(k_x, k_y, k_z)$ is completely independent.
- Total evaluations scale as $M_x M_y M_z n^3$. For example, $M_x = M_y = M_z = 64$, $n = 8$ produces $1.34 \times 10^8$ evaluations; $M_x = M_y = M_z = 128$, $n = 8$ produces $1.07 \times 10^9$ evaluations — naturally reaching multi-second to multi-minute serial runtimes!

---

### Step 4 — Adaptive 1D Gauss–Legendre (For Studying Load Imbalance)

Compare the $n$-point integral on $[l, r]$ against the sum of two $n$-point integrals on $[l, m]$ and $[m, r]$ ($m = (l + r)/2$):

```text
function integrate_1d_adaptive(f, a, b, rule, tol, max_depth):
    stack = empty stack of (l, r, I_coarse, depth)
    I_root = eval_panel(f, a, b, rule)
    push(stack, (a, b, I_root, 0))
    total = 0.0

    while stack is not empty:
        (l, r, I_coarse, depth) = pop(stack)
        m = 0.5 * (l + r)
        I_left  = eval_panel(f, l, m, rule)
        I_right = eval_panel(f, m, r, rule)
        I_fine  = I_left + I_right
        err_est = abs(I_fine - I_coarse)

        // Local tolerance scaled proportionally to interval width (or absolute)
        local_tol = tol * (r - l) / (b - a)
        if err_est <= local_tol or depth >= max_depth:
            total += I_fine
        else:
            push(stack, (m, r, I_right, depth + 1))
            push(stack, (l, m, I_left,  depth + 1))
    return total
```

Using an explicit stack (instead of recursion) prevents stack overflow on sharp singularities and maps cleanly to parallel work-pools later.

---

## 4. Numerical Precision & Floating-Point Considerations

1. **Kahan (Compensated) Summation:** When summing $10^8$–$10^9$ panel values in `double`, naive summation `total += val` accumulates rounding error of order $O(\sqrt{M}\,\epsilon_{\text{mach}})$ or $O(M\,\epsilon_{\text{mach}})$. Kahan summation keeps rounding error near $O(\epsilon_{\text{mach}})$ with minimal overhead.
2. **Non-Associativity in Parallel Reductions:** Floating-point addition is **not associative**: $(a + b) + c \ne a + (b + c)$ at the last ULP. Expect serial and parallel results to agree within ~$10^{-13}$–$10^{-15}$ (relative error), not bit-for-bit identical. Always validate using:
   $$|I_{\text{parallel}} - I_{\text{exact}}| < \varepsilon_{\text{tol}} \quad \text{and} \quad |I_{\text{parallel}} - I_{\text{serial}}| < 10^{-12} \cdot \max(1, |I_{\text{exact}}|)$$
3. **Avoid `-ffast-math`:** Compiler flag `-ffast-math` breaks IEEE-754 guarantees and can optimize away Kahan summation (`(t - total) - y` becomes `0`). Use `-O3` without `-ffast-math`.

---

## 5. Recommended Test Functions (With Known Analytical Solutions)

Include a small suite of integrands selectable via CLI (`--func <id>`) so you can validate accuracy and test different computational profiles:

| ID | Integrand | Domain | Exact Integral | Purpose |
| --- | --- | --- | --- | --- |
| `poly` | $f(x) = x^{15} - 3x^8 + 2x^2 + 1$ | $[-1, 1]$ | $\frac{8}{3}$ | Exactness sanity check ($n = 8$ must give machine precision in 1 panel!) |
| `smooth_1d` | $f(x) = e^{-x^2}\cos(5x) + \sin(10x)\log(2 + x^2)$ | $[0, 10]$ | Reference via high-order quad / closed form for $\int e^x \sin(x)\,dx$ | Uniform smooth workload |
| `osc_1d` | $f(x) = \sin(50x)\,e^{-0.1x} + \cos(120x)/(1 + x^2)$ | $[0, 20]$ | High-precision reference | High-frequency oscillations requiring many panels $M$ |
| `peak_1d` | $f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x - x_0)^2}{2\sigma^2}}$ ($\sigma = 10^{-4}$, $x_0 = 0.2$) | $[0, 1]$ | $\approx 1.0$ (exact via $\text{erf}$) | **Localized sharp peak** — demonstrates adaptive refinement & load imbalance! |
| `heavy_3d` | $f(x, y, z) = \sin(\pi x)\sin(\pi y)\sin(\pi z) \cdot e^{-(x^2 + y^2 + z^2)/4}$ (or separable $\prod \cos/\exp$) | $[0, 1]^3$ | Product of 1D exact integrals | **3D HPC benchmark** (separable $g(x)h(y)k(z)$ gives an exact analytical value while still evaluating full 3D transcendentals!) |

> **Tip for `heavy_3d`:** Choosing a separable product like $f(x,y,z) = e^{-x}\sin(\pi x)\cdot e^{-y}\sin(\pi y)\cdot e^{-z}\sin(\pi z)$ (plus non-separable cross terms if desired, or known sums of separable terms) gives an **exact closed-form analytical integral** $\left(\frac{\pi(1 + e^{-1})}{1 + \pi^2}\right)^3$ so you can always verify numerical error down to $10^{-15}$!

---

## 6. Serial Compilation, CLI & Timing

### Recommended Compilation
```bash
gcc -O3 -std=c11 -Wall -Wextra -o bin/quad_serial \
    src/serial/main_serial.c src/common/gauss_legendre.c src/common/integrands.c -lm
```

### Recommended CLI Interface (Keep Identical Across Serial, MPI, OpenMP, Hybrid!)
```bash
./bin/quad_serial --mode 3d --func heavy_3d --order 8 --panels 100 --repeats 5
```
Keeping the exact same CLI flags and CSV/structured output format across all implementations makes automated benchmarking and plotting effortless:
```text
impl,mode,func,order,panels,ranks,threads,run_id,t_nodes_s,t_eval_s,t_total_s,result,abs_error
serial,3d,heavy_3d,8,100,1,1,1,0.000012,18.423105,18.423117,0.008412345678901,1.12e-15
```

### Wall-Clock Timing in Serial Code
As covered in Lecture 11 (Slide 37 / Note 09), measure **wall-clock time** (not `clock()` CPU time), excluding CLI parsing, allocation, and printing:
- Use `gettimeofday(&tv, NULL)` (standard in the course) or `clock_gettime(CLOCK_MONOTONIC, &ts)` (monotonic, immune to system clock adjustments).
- Record **two separate timings**:
  1. `t_nodes`: Phase 1 (computing Gauss–Legendre nodes & weights)
  2. `t_eval`: Phase 2 (evaluating the composite/3D/adaptive integral)
  3. `t_total = t_nodes + t_eval`

Reporting both phases separately shows clearly in the report what fraction of time is spent in node generation vs. integration (helpful for Amdahl's Law analysis!).
