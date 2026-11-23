# Gaussian Quadrature: Mathematical Foundations & Algorithms

This document explains the theory behind **Gaussian quadrature**, how quadrature nodes and weights are computed, how the method extends to composite, multi-dimensional, and adaptive integration, and why it is well suited for HPC parallelization.

---

## 1. The Numerical Integration Problem

We want to approximate the definite integral of a real-valued function $f$:

- **1D:**
  $$I = \int_{a}^{b} f(x)\,dx$$
- **$d$-Dimensional (hyper-rectangular domain $\Omega = \prod_{m=1}^{d} [a_m, b_m]$):**
  $$I = \int_{a_1}^{b_1} \cdots \int_{a_d}^{b_d} f(x_1, \dots, x_d)\,dx_1 \cdots dx_d$$

In many scientific computing and data science workloads (Bayesian marginalization, finite element assembly, physics/electromagnetic potentials, statistical expectations, option pricing), $f$ has no closed-form antiderivative or is expensive to evaluate, making numerical quadrature essential.

### Why Gaussian Quadrature vs. Newton–Cotes?
In class (Lecture 5 / Note 05), we saw the **Trapezoidal Rule**, which is a **Newton–Cotes** formula: it fixes evaluation points at equally spaced abscissas and chooses only the weights. An $n$-point Newton–Cotes rule is exact for polynomials of degree at most $n - 1$ (or $n$ for even $n$).

**Gaussian quadrature** frees **both** the $n$ nodes $x_1, \dots, x_n$ and the $n$ weights $w_1, \dots, w_n$. With $2n$ degrees of freedom, an $n$-point Gaussian quadrature rule achieves the **maximum possible algebraic degree of exactness: $2n - 1$**.

| Method (per panel / $n$ points) | Exact polynomial degree | Convergence on smooth $f$ |
| --- | --- | --- |
| Composite Trapezoidal ($M$ panels) | $1$ | $O(h^2)$ |
| Composite Simpson ($M$ panels) | $3$ | $O(h^4)$ |
| **Single $n$-point Gauss–Legendre** | **$2n - 1$** | Spectral / exponential for analytic $f$ |
| **Composite $n$-point Gauss–Legendre ($M$ panels)** | **$2n - 1$ per panel** | **$O(h^{2n})$** |

---

## 2. Gauss–Legendre Quadrature

The standard Gaussian rule for integrals with unit weight function $W(x) = 1$ on $[-1, 1]$ is **Gauss–Legendre quadrature**:

$$\int_{-1}^{1} g(\xi)\,d\xi \approx \sum_{i=1}^{n} w_i \, g(\xi_i)$$

### Change of Interval from $[a, b]$ to $[-1, 1]$
Any finite interval $[a, b]$ maps linearly to $[-1, 1]$ via:

$$x(\xi) = \frac{b - a}{2}\,\xi + \frac{a + b}{2}, \qquad dx = \frac{b - a}{2}\,d\xi$$

Thus:

$$\int_{a}^{b} f(x)\,dx \approx \frac{b - a}{2} \sum_{i=1}^{n} w_i \, f\!\left(\frac{b - a}{2}\,\xi_i + \frac{a + b}{2}\right)$$

### Definition of Nodes and Weights
The nodes $\xi_1, \dots, \xi_n \in (-1, 1)$ are the **$n$ distinct real roots of the $n$-th Legendre polynomial $P_n(\xi)$**, which satisfies the three-term recurrence relation:

$$P_0(\xi) = 1, \qquad P_1(\xi) = \xi$$
$$k\,P_k(\xi) = (2k - 1)\,\xi\,P_{k-1}(\xi) - (k - 1)\,P_{k-2}(\xi), \qquad k = 2, \dots, n$$

Its derivative is given by:

$$P_n'(\xi) = \frac{n}{\xi^2 - 1}\Big(\xi\,P_n(\xi) - P_{n-1}(\xi)\Big)$$

The corresponding quadrature weights are strictly positive and sum to $2$:

$$w_i = \frac{2}{(1 - \xi_i^2)\,[P_n'(\xi_i)]^2}, \qquad \sum_{i=1}^{n} w_i = 2$$

**Symmetry property:** Because $P_n(-\xi) = (-1)^n P_n(\xi)$, nodes and weights are symmetric around $0$:

$$\xi_{n+1-i} = -\xi_i, \qquad w_{n+1-i} = w_i$$

Only $m = \lfloor (n + 1) / 2 \rfloor$ roots need to be computed explicitly!

### Error Formula
For $f \in C^{2n}[a, b]$, the quadrature error on a single interval $[a, b]$ is:

$$E_n = \frac{(b - a)^{2n+1}\,(n!)^4}{(2n + 1)\,[(2n)!]^3}\,f^{(2n)}(\eta), \qquad \eta \in (a, b)$$

For a **composite** rule over $M$ uniform subintervals of width $h = (b - a)/M$, the error scales as $O(h^{2n})$:

$$E_{M, n} = \frac{(b - a)\,h^{2n}\,(n!)^4}{(2n + 1)\,[(2n)!]^3}\,f^{(2n)}(\bar{\eta})$$

---

## 3. Algorithms for Computing Nodes and Weights (Phase 1)

There are three well-known ways to compute $(\xi_i, w_i)$ for an arbitrary order $n$:

### Method A — Newton–Raphson on $P_n(\xi)$ (Recommended for Implementation)
For each $i = 1, \dots, \lfloor (n + 1) / 2 \rfloor$:
1. Start from the Tricomi / asymptotic initial guess:
   $$\xi_i^{(0)} = \cos\!\left(\pi\,\frac{i - 0.25}{n + 0.5}\right)$$
2. Iterate Newton's step until $|\Delta \xi| < \varepsilon$ (e.g., $10^{-15}$):
   $$\xi \leftarrow \xi - \frac{P_n(\xi)}{P_n'(\xi)}$$
   evaluating $P_n(\xi)$ and $P_n'(\xi)$ in $O(n)$ operations via the three-term recurrence.
3. Compute $w_i = \frac{2}{(1 - \xi_i^2)[P_n'(\xi_i)]^2}$ and mirror to index $n + 1 - i$.

- **Convergence:** Quadratic — typically converges to full IEEE-754 `double` precision (`~1e-16`) in **4 to 6 iterations** for any $n$!
- **Complexity:** $O(n)$ per root $\Rightarrow$ **$O(n^2)$ total** serial work, $O(n)$ memory.
- **Parallelism:** Every root $i$ is **100% independent** of every other root.

### Method B — Golub–Welsch Algorithm (Literature Reference)
Forms the symmetric tridiagonal **Jacobi matrix** $J_n \in \mathbb{R}^{n \times n}$ with zero diagonal and subdiagonal entries:

$$\beta_k = \frac{k}{\sqrt{4k^2 - 1}}, \qquad k = 1, \dots, n - 1$$

The eigenvalues of $J_n$ are the nodes $\xi_i$, and the weights are $w_i = 2\,v_{1,i}^2$, where $v_{1,i}$ is the first component of the normalized eigenvector for $\xi_i$.
- **Complexity:** $O(n^2)$ using specialized symmetric tridiagonal QR/divide-and-conquer eigensolvers (e.g., LAPACK `dstev`), $O(n^2)$ memory if all eigenvectors are stored.
- **Note for report:** Mention Golub–Welsch (1969) in the state-of-the-art analysis and explain why Newton–Raphson is faster, simpler in pure C without LAPACK dependencies, more accurate for large $n$, and embarrassingly parallel across roots.

### Method C — Glaser–Liu–Rokhlin / Hale–Townsend Asymptotic Methods ($O(n)$)
Modern fast methods (e.g., Glaser, Liu & Rokhlin 2007; Hale & Townsend 2013) compute nodes and weights in **$O(n)$** time using Prüfer transformations or asymptotic expansions, enabling $n > 10^6$. Worth citing in the State-of-the-Art section of the report.

---

## 4. Extending to HPC-Scale Workloads (Phase 2)

A single 1D rule with small $n$ (say $n = 16$) finishes in microseconds — far too small for HPC benchmarking. Real HPC integration workloads arise in three natural extensions of Gaussian quadrature:

### 4.1 Composite 1D Gauss–Legendre Quadrature
Partition $[a, b]$ into $M$ subintervals $[x_k, x_{k+1}]$ of width $h = (b - a)/M$ ($x_k = a + k h$) and apply an $n$-point Gauss–Legendre rule on every panel:

$$I_{M, n} = \frac{h}{2} \sum_{k=0}^{M-1} \sum_{i=1}^{n} w_i \, f\!\left(x_k + \frac{h}{2}(\xi_i + 1)\right)$$

- **Total evaluations:** $N_{\text{eval}} = M \cdot n$.
- **Compute profile:** Regular, uniform cost per panel (unless $f(x)$ itself has position-dependent cost). Ideal for studying **strong/weak scalability** with static domain decomposition.

### 4.2 Multi-Dimensional Tensor-Product Gauss–Legendre (2D / 3D)
In $d$ dimensions, partition each axis $m \in \{1, \dots, d\}$ into $M_m$ panels of width $h_m = (b_m - a_m)/M_m$ and apply an $n$-point rule along every axis. For 3D:

$$I \approx \frac{h_x h_y h_z}{8} \sum_{k_x=0}^{M_x-1} \sum_{k_y=0}^{M_y-1} \sum_{k_z=0}^{M_z-1} \sum_{i=1}^{n} \sum_{j=1}^{n} \sum_{l=1}^{n} w_i w_j w_l \, f\!\left(x_{k_x,i},\, y_{k_y,j},\, z_{k_z,l}\right)$$

- **Total evaluations:** $N_{\text{eval}} = (M_x M_y M_z) \cdot n^3$.
- **Why this is great for HPC:** Suffers from the **curse of dimensionality** — even moderate grids ($M_x = M_y = M_z = 100$, $n = 8 \Rightarrow 5.12 \times 10^8$ evaluations) genuinely require seconds to minutes serially, motivating distributed and hybrid parallelization naturally without artificial busy-wait loops!

### 4.3 Adaptive Gaussian Quadrature (Gauss–Kronrod / Panel Refinement)
When $f(x)$ has sharp localized peaks, high-frequency oscillations, or near-singularities, uniform subdivision wastes evaluations in smooth regions. **Adaptive quadrature** estimates the local error on each subinterval $[a_k, b_k]$ by comparing two rules:
- **Embedded Gauss–Kronrod pair** (e.g., $G_7$–$K_{15}$, the core of QUADPACK / `scipy.integrate.quad`): reuses the 7 Gauss–Legendre points inside a 15-point Kronrod rule, giving an integral estimate $I_{K15}$ and error estimate $e_k = |I_{K15} - I_{G7}|$ in just 15 evaluations.
- **Or two-level Gauss–Legendre:** compare $n$-point rule on $[a_k, b_k]$ vs. two $n$-point rules on the left and right halves $[a_k, m_k]$ and $[m_k, b_k]$.

If $e_k > \text{tol}_k$, the subinterval is split and pushed onto a work queue/stack; otherwise $I_{\text{fine}}$ is accepted.
- **Why this is great for the Parallel Design section:** Adaptive refinement produces a **highly irregular, unpredictable workload** across $[a, b]$. Static partitioning leads to severe **load imbalance**, motivating **dynamic load balancing** (master–worker / work-stealing in MPI; `schedule(dynamic)` / `schedule(guided)` or OpenMP tasks in shared memory).

---

## 5. Other Gaussian Quadrature Families (Context for Introduction)

For completeness in the report's Introduction, briefly note that Gauss–Legendre is part of a general family $\int_a^b W(x) f(x)\,dx \approx \sum w_i f(x_i)$ tied to orthogonal polynomials:

| Family | Domain | Weight $W(x)$ | Orthogonal Polynomial |
| --- | --- | --- | --- |
| **Gauss–Legendre** (our focus) | $[-1, 1]$ | $1$ | Legendre $P_n(x)$ |
| Gauss–Chebyshev (1st kind) | $[-1, 1]$ | $(1 - x^2)^{-1/2}$ | Chebyshev $T_n(x)$ |
| Gauss–Laguerre | $[0, \infty)$ | $e^{-x}$ | Laguerre $L_n(x)$ |
| Gauss–Hermite | $(-\infty, \infty)$ | $e^{-x^2}$ | Hermite $H_n(x)$ |
| Gauss–Jacobi | $[-1, 1]$ | $(1-x)^\alpha(1+x)^\beta$ | Jacobi $P_n^{(\alpha,\beta)}(x)$ |

---

## 6. Complexity Summary

| Phase / Variant | Serial Time Complexity | Memory Complexity |
| --- | --- | --- |
| Node & weight generation (Newton–Raphson) | $O(n^2)$ | $O(n)$ |
| 1D Composite Gauss–Legendre ($M$ panels, order $n$) | $O(n^2 + M \cdot n \cdot C_f)$ | $O(n)$ |
| $d$-D Tensor-Product Composite ($M$ panels/axis) | $O(n^2 + M^d \cdot n^d \cdot C_f)$ | $O(n)$ |
| Adaptive 1D (resulting in $M_{\text{adapt}}$ accepted panels) | $O(n^2 + M_{\text{adapt}} \cdot n \cdot C_f)$ | $O(\text{depth})$ or $O(M_{\text{active}})$ |

*(where $C_f$ is the cost of one evaluation of the integrand $f$.)*

Notice the extremely **low memory footprint** ($O(n)$ for nodes and weights) and **high arithmetic intensity**: Gaussian quadrature is predominantly **compute-bound**, not memory-bandwidth-bound — which explains why well-balanced parallel implementations achieve near-linear strong and weak scalability.

---

## 7. Key Literature References (For State-of-the-Art Section)

1. **Golub, G. H., & Welsch, J. H. (1969).** *Calculation of Gauss Quadrature Rules.* Mathematics of Computation, 23(106), 221–230.
2. **Davis, P. J., & Rabinowitz, P. (1984).** *Methods of Numerical Integration* (2nd ed.). Academic Press.
3. **Piessens, R., de Doncker-Kapenga, E., Überhuber, C. W., & Kahaner, D. K. (1983).** *QUADPACK: A Subroutine Package for Automatic Integration.* Springer-Verlag.
4. **Glaser, A., Liu, X., & Rokhlin, V. (2007).** *A fast algorithm for the calculation of the roots of special functions.* SIAM Journal on Scientific Computing, 29(4), 1420–1438.
5. **Hale, N., & Townsend, A. (2013).** *Fast and accurate computation of Gauss–Legendre and Gauss–Jacobi quadrature nodes and weights.* SIAM Journal on Scientific Computing, 35(2), A652–A674.
6. **de Doncker, E., Kapenga, J., & Genz, A. (1996) / ParInt Project.** *Parallel Numerical Integration (PARINT).* Western Michigan University — classic reference for distributed-memory adaptive numerical integration and dynamic load balancing with MPI.
7. **Pacheco, P. S. (2011).** *An Introduction to Parallel Programming.* Morgan Kaufmann (course textbook — Chapter 3 MPI trapezoidal rule & collective operations; Chapter 5 OpenMP reductions & scheduling).
