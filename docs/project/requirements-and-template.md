# Project Requirements & Report Template Recap

*Source: `HPC4DS_Lecture11.pdf` (Slides 4–10, 29–37) and Lecture 11 recording/notes — Prof. Sandro Fiore, Ph.D. (DISI, University of Trento, 2026–2027).*

---

## 1. Assignment Overview

- **Team ID:** 25
- **Assigned Topic:** Parallel numerical integration using Gaussian Quadrature
- **Goal:** Design, implement (in C or C++), and evaluate a parallel solution to the assigned algorithm using **MPI** and **OpenMP**, demonstrating significant, well-analyzed speedup over a verified sequential baseline on both single-node and multi-node cluster configurations.
- **Format:** Team project accompanied by a written technical report (**≥ 10 pages**), a link to the public/accessible Git repository, and an **oral examination / presentation**.

---

## 2. Required Report Structure (Project Template)

The professor defines four mandatory core sections for the final report:

### Section 1 — Introduction
- Provide an **accurate description of the assigned problem** (numerical integration via Gaussian quadrature, mathematical formulation, and real-world/scientific motivation).
- Present and analyze the **sequential algorithm** that solves the problem (asymptotic time/space complexity, numerical properties, and computational bottlenecks).
- Include **pseudocode, worked examples, graphs, figures, and application instances** where helpful.

### Section 2 — Parallel Design
- Perform a **preliminary study of the opportunities for parallelism** inherent in the sequential algorithm (e.g., task decomposition, granularity, computation-to-communication ratio).
- Conduct a **state-of-the-art analysis** of existing parallelization strategies for the problem, with **explicit citations to prior and related work**.
- Consider and compare **alternative parallel designs**, clearly motivating:
  - Which operations parallelize effectively and which do not (serial bottlenecks / Amdahl's limit).
  - How the choice of **data structures** affects communication and synchronization overhead.
- Discuss **hybrid parallelization strategies (MPI + OpenMP)** and explicitly analyze **data dependencies** (flow, anti, output, loop-carried dependencies, and how reductions/privatization handle them).

### Section 3 — Implementation
- **Language:** **C** (C++ is also allowed).
- **Code quality:** Code must be **properly commented**, clean, and well-structured.
- **Repository link:** A link to the Git repository **must** be included in the report.
- **Multiple strategies:** If multiple parallelization strategies are identified (e.g., static MPI vs. dynamic MPI, pure MPI vs. pure OpenMP vs. Hybrid MPI+OpenMP), implement them and discuss the **pros and cons** of each.
- **Code snippets:** Include concise snippets in the report highlighting the most critical and interesting parts (not full file dumps).
- **Hybrid note:** Slide 7 states that hybrid parallelization is *recommended though not mandatory*, while Slide 10 describes the project as a *parallel application using MPI and OpenMP*. Starting with MPI and then introducing OpenMP (both standalone single-node and hybrid MPI+OpenMP) directly satisfies both slides and strengthens the evaluation.

### Section 4 — Performance and Scalability Analysis
- Measure and analyze:
  - **Execution time** ($T_{\text{serial}}$ and $T_{\text{parallel}}$)
  - **Speedup** ($S = T_{\text{serial}} / T_{\text{parallel}}$)
  - **Efficiency** ($E = S / p$)
- Evaluate both **strong scalability** and **weak scalability** wherever possible.
- Run and compare **single-node** and **multi-node** benchmarks on the cluster.
- Test across **three problem sizes: Small, Medium, and Large** (defined meaningfully for our algorithm) and compare how scalability changes with problem size.
- Provide **critical analysis** of the results (explaining deviations from linear speedup via communication overhead, load imbalance, memory/cache effects, Amdahl's law, and cluster noise).

---

## 3. Mandatory Formal Requirements

| Requirement | Details |
| --- | --- |
| **Report length** | Expected size **≥ 10 pages**. |
| **AI statement** | An **AI usage statement is mandatory** in every report. |
| **Citations & references** | Must cite papers/literature for the state-of-the-art analysis and any data sources or code inspirations. |
| **Repository link** | Must provide a working link to the code repository. |
| **Oral examination** | **All team members must be present** during the oral exam and project presentation. |

---

## 4. Evaluation Criteria

According to Slide 8, grading evaluates:
1. **Clarity and effectiveness of presentation** (structure, figures, tables, writing, oral defense).
2. **Depth, correctness, and originality of theoretical analysis** (sequential complexity, mathematical accuracy, dependency & communication analysis).
3. **Technical skills and documentation of implementation** (clean C/C++ code, modularity, comments, build/run reproducibility).
4. **Number and quality of multiple parallel strategies**, if any (e.g., comparing static vs. dynamic partitioning, pure MPI vs. OpenMP vs. hybrid).
5. **Critical thinking in performance evaluation and analysis** (explaining *why* curves behave the way they do, not just pasting plots).
6. **Significance of results:** Parallel programming is about performance — the project **must achieve a significant speedup**.

---

## 5. Practical Rules & Tips from Lecture 11

### Avoiding Invalid Benchmarks (Professor's Explicit Warning)
- **Never submit benchmarks that run in `< 1 second`:** If the serial or parallel experiments complete in under a second, statistical noise (OS scheduling, paging, cluster I/O) dominates and scalability cannot be studied meaningfully. The professor explicitly warned that sub-second benchmark sections will be rejected.
- **Target realistic serial runtimes:** Choose problem sizes where the serial execution takes substantial time (e.g., tens of seconds for Small/Medium up to several minutes for Large), so even at the highest core counts the parallel runtime stays well above 1 second and measurements remain stable.
- **Repeat runs:** Repeat each benchmark configuration multiple times (e.g., 5–10 runs) and report mean and variance/standard deviation to quantify residual cluster noise (shared network and shared storage).
- **Use exclusive placement for benchmarks:** On the PBS cluster, use `-l place=pack:excl` (single-node) and `-l place=scatter:excl` (multi-node) so other users' jobs do not share CPU/memory bandwidth on your allocated nodes.

### Datasets & Replicating Existing Work (Slide 9)
- If real-world input data is not directly applicable (as in numerical integration), **generate synthetic benchmarks / test functions** systematically across **Small, Medium, and Large** problem sizes.
- If inspired by an existing paper or implementation, consider one of these patterns:
  - Reproduce its algorithm in C from pseudocode and compare findings on the cluster.
  - Extend an existing approach with your own improvement (delta) and quantify the gain.
  - Compare two distinct parallel approaches ported to the same cluster environment.

---

## 6. Project Checklist

- [ ] Sequential C implementation verified against analytical integrals (known exact values).
- [ ] Serial benchmarks collected for Small, Medium, and Large workloads (all comfortably > 1 s).
- [ ] Pure MPI implementation (static decomposition) verified against serial baseline.
- [ ] Second parallel strategy (e.g., dynamic/adaptive MPI or multi-dimensional decomposition) for comparative analysis.
- [ ] OpenMP implementation (comparing `static`, `dynamic`, `guided` schedules and coarse vs. fine grain).
- [ ] Hybrid MPI + OpenMP implementation (`MPI_THREAD_FUNNELED` + OpenMP parallel reduction).
- [ ] Strong and weak scalability experiments on single-node (`pack:excl`) and multi-node (`scatter:excl`) allocations, repeated multiple times.
- [ ] Report (≥ 10 pages) containing all 4 template sections, state-of-the-art citations, repo link, and mandatory AI statement.
