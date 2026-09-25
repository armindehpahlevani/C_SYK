# Performance report: `majorana_SYK_fast.py`

The task was to push full exact diagonalization of the q=4 Majorana SYK model
to the largest N an ordinary laptop can handle, without changing the physics.
All numbers below were measured on a 4-core Intel Xeon (2.8 GHz), 15 GB RAM,
OpenBLAS 0.3.30, NumPy 2.4, SciPy 1.17, Numba 0.67. A typical recent laptop
has similar per-core speed, so expect similar numbers within about 2x.

`majorana_SYK_reference.py` is the previous version, kept unchanged. It
serves as the correctness oracle.

---

## 1. Audit of the previous version

### Performance bottlenecks, largest first

| # | Bottleneck | Measured effect |
|---|---|---|
| 1 | Dense O(D^3) diagonalization of every parity block. | 85–95% of the time from N=24 on. N=26: 10.2 s of 11.3 s. |
| 2 | For N mod 8 = 0, the blocks are diagonalized as **complex** Hermitian matrices, although they are exactly **real** symmetric in a suitable basis (section 5). | At equal dimension, real is 3.4x faster at D=4096 and 4.3x faster at D=8192, with half the memory. |
| 3 | SciPy's default eigensolver uses the 1-stage Householder tridiagonalization. That step is limited by memory bandwidth, not by arithmetic. | At D=8192, the 2-stage LAPACK driver is 1.38x faster (complex) and 1.48x faster (real). |
| 4 | Sparse CSR build, then `.toarray()`, then an internal copy into Fortran order (f2py). That holds two dense matrices at once. | Peak memory at N=26: 785 MB, against 440 MB for the new code. |
| 5 | A Python loop over all C(N,4) couplings, doing NumPy vector operations for each term and building a dict of per-mask arrays. | Old build takes 0.59 s at N=24, 1.09 s at N=26 and about 7 s at N=28 per realization. |
| 6 | Realizations run one after another, with multithreaded BLAS on small matrices, which scales poorly. | N=20: 122 ms per realization with 1 job x 4 threads, 62 ms with 4 jobs x 1 thread. |
| 7 | The Hermiticity test builds `B - B^H`, an extra sparse copy, on an extra random realization. The single-realization spectrum is yet another realization. | Duplicate build and diagonalization work. |

### Scientific and statistical problems found

The Hamiltonian, the Majorana normalization and the coupling variance were
all correct. The new code reproduces the old eigenvalues to 1e-14 (section 4).

1. **Reference values for <r>.** The old code compared with 0.536 / 0.603 /
   0.676. Those are the 3x3 "surmise" values. The correct large-matrix values
   are GOE 0.5307, GUE 0.5996 and GSE 0.6744 (Atas et al., PRL 110, 084101
   (2013)). The difference matters at the precision reached: the measured
   ⟨r⟩ is 0.530 at N=16 and 0.675 at N=20.
2. **Removing Kramers degeneracies.** The old code deleted levels closer
   than `1e-10 * bandwidth`. The new code takes one level per pair
   (`e[0::2]`) and reports the largest pair splitting, which is about 1e-14,
   so the degeneracy is exact. A tolerance rule can accidentally delete a
   genuine near-degeneracy, or miss a pair if the tolerance is wrong.
3. **The symmetry reduction was assumed, not verified.** The old code
   dropped the odd block for N mod 8 = 2, 6 because a proof said it could.
   The new code verifies `[H, T] = 0` without building a matrix, on every
   run, for the actual couplings at the actual N.
4. **Standard error of E0/N** now uses the sample standard deviation
   (ddof=1). This is a tiny correction.

Both versions correctly computed ⟨r⟩ separately in each parity block.
Mixing sectors gives nonsense. On the same realization, the full unsplit
spectrum gives ⟨r⟩ = 0.41 for N=16, which the program prints as a warning
example.

---

## 2. Architectures considered

| | Architecture | Verdict |
|---|---|---|
| A | Old: sparse build, `toarray`, complex `eigvalsh`, one realization at a time | baseline |
| B | Build directly into the final Fortran-ordered LAPACK array with Numba; complex solver in every sector | avoids the copies, but the diagonalization cost is unchanged |
| C | **B + a real symmetry-adapted basis for N mod 8 = 0 + the 2-stage LAPACK solver + one realization per core** | **chosen** |
| D | Matrix-free Lanczos or KPM | only part of the spectrum. Rejected for the default, since the full spectrum is required. Available as an option in `syk_spectrum.py`. |
| E | Packed storage (`dspevd` / `zhpevd`) | halves the memory, but measured 1.5x (complex) to 7x (real) slower. Rejected. It would only help to fit N=32 into 8 GB. |

Eigensolver measurements on random matrices, eigenvalues only, 4 threads:

| matrix | SciPy `ev` | SciPy `evd` | SciPy `evr` | **2-stage** | packed |
|---|---|---|---|---|---|
| real 2048 | 0.35 s | 0.42 s | 0.33 s | 0.43 s | 2.29 s |
| real 4096 | 2.47 s | 2.50 s | 2.44 s | 2.51 s | 17.1 s |
| real 8192 | | | 21.6 s | **14.6 s** | |
| complex 2048 | 1.18 s | 1.12 s | 1.21 s | 1.02 s | 1.26 s |
| complex 4096 | 8.76 s | 8.48 s | 8.48 s | **6.55 s** | 12.6 s |
| complex 8192 | | | 62.3 s | **45.1 s** | |

The 2-stage routines are not wrapped by SciPy. The program calls them
through `ctypes` from the OpenBLAS library that SciPy/NumPy already ship,
checks them against SciPy at start-up, and falls back to SciPy if they are
missing or disagree. They are used for D >= 4096, where they are faster.

I also tried two alternative assembly kernels (branch-free signs, and
column-blocked SIMD). Both were 4–50% slower than the simple kernel that was
kept. Assembly is below 2% of the runtime from N=26 on.

---

## 3. The new architecture

```
config -> validation (all tests, target N included; ~2 s)
       -> precomputation once per N: Pauli strings of all C(N,4) terms,
          grouping by flip mask, parity blocks and index maps, the T-symmetry
          map sigma(s), the real-basis index maps
       -> per realization (streamed; one realization per core):
            couplings from its own seed stream (order-independent)
            -> Numba assembly straight into the final dense LAPACK array
               (real 2-component basis for GOE; complex otherwise)
            -> in-place eigensolver (2-stage LAPACK for D >= 4096)
            -> E0, per-sector <r>, then the matrix is released
            -> eigenvalues written to the .npy file on disk (memmap)
       -> statistics, histogram, files
```

Work per realization and sector, with D = 2^(N/2 - 1):

| N mod 8 | RMT | sectors diagonalized | matrix type | memory | cost relative to the old code |
|---|---|---|---|---|---|
| 0 | GOE | 2 | real D x D | 8 D^2 bytes | about 1/4 (real) x 1/1.4 (2-stage) |
| 2, 6 | GUE | 1 (odd = copy of even) | complex D x D | 16 D^2 bytes | about 1/1.4 (the old code also skipped the odd block) |
| 4 | GSE | 2 | complex D x D | 16 D^2 bytes | about 1/1.4 |

No valid symmetry reduces the GSE case further with standard LAPACK. The
quaternion structure halves the independent levels, but there is no LAPACK
quaternion eigensolver.

Scaling: time grows as D^3 and memory as D^2. Every step N -> N+2 doubles D,
which makes each sector 8x slower and 4x larger. How many sectors, and of
which type, changes with N mod 8. That's why the cost per step is uneven.
Parallel realizations speed things up almost linearly with the number of
cores, as long as the matrices fit in RAM (section 4).

---

## 4. Benchmarks (one realization, fresh process each)

Time for one realization with all 4 cores. Diagonalization dominates from
N=24 on.

| N | N mod 8 | RMT | Hilbert dim | block dim D | sectors x type | old build | old diag | **old total** | new build | new diag | **new total** | speed-up | peak memory old -> new |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 0 | GOE | 256 | 128 | 2 x real | 0.022 s | 0.004 s | 0.027 s | 0.014 s | 0.003 s | 0.017 s | 1.6x | 155 -> 193 MB |
| 18 | 2 | GUE | 512 | 256 | 1 x complex | 0.024 s | 0.016 s | 0.040 s | 0.005 s | 0.015 s | 0.020 s | 2.0x | 158 -> 187 MB |
| 20 | 4 | GSE | 1024 | 512 | 2 x complex | 0.104 s | 0.12 s | 0.23 s | 0.019 s | 0.17 s | 0.19 s | 1.2x | 170 -> 179 MB |
| 22 | 6 | GUE | 2048 | 1024 | 1 x complex | 0.188 s | 0.33 s | 0.52 s | 0.074 s | 0.24 s | 0.31 s | 1.7x | 196 -> 192 MB |
| 24 | 0 | GOE | 4096 | 2048 | 2 x real | 0.59 s | 2.41 s | 2.99 s | 0.19 s | 0.70 s | 0.89 s | **3.4x** | 325 -> 209 MB |
| 26 | 2 | GUE | 8192 | 4096 | 1 x complex | 1.09 s | 10.2 s | 11.3 s | 0.48 s | 6.07 s | 6.54 s | 1.7x | 785 -> 440 MB |
| 28 | 4 | GSE | 16384 | 8192 | 2 x complex | 6.1 s | 142.5 s | 148.6 s | 1.6 s | 86.0 s | 87.7 s | 1.7x | 2.5 -> 1.2 GB |
| 30 | 6 | GUE | 32768 | 16384 | 1 x complex | | | (estimated ~550 s, ~8.6 GB) | 3.5 s | 309 s | **312 s (5.2 min)** | ~1.8x | ~8.6 -> **4.5 GB** |
| 32 | 0 | GOE | 65536 | 32768 | 2 x real | | | **impossible**: needs 2 x 17 GB complex | 9.0 s | 1469 s | **1478 s (24.6 min)** | new capability | -> **8.8 GB** |

For the same couplings, E0 of the old and new code agree to within
1e-16 to 1.3e-14 at N=16–26.

In the new code, building the Hamiltonian ("new build") costs 0.1–2% of the
diagonalization time from N=24 on. Every further gain has to come from the
eigensolver, and the only ones available are the real basis and the 2-stage
driver above.

### Choosing jobs and threads

The table below gives wall time per realization, measured with many
realizations:

| N | 4 jobs x 1 thread | 2 x 2 | 1 job x 4 threads |
|---|---|---|---|
| 20 | **62 ms** | 88 ms | 122 ms |
| 22 | **149 ms** | 177 ms | 220 ms |
| 24 | **568 ms** | 638 ms | 779 ms |
| 26 | **4.8 s** | 5.2 s | 6.2 s |
| 28 | **67 s** | | 83 s |

By default the program runs one realization per physical core, with 1 BLAS
thread each. It lowers the number of jobs until `jobs x matrix memory` fits
in 75% of the available RAM. With fewer realizations than cores, it gives
the spare cores to BLAS instead, so jobs x threads never exceeds the number
of cores and they don't compete. Override with `--jobs` and
`--blas-threads`.

### Throughput with many realizations (automatic parallel layout)

This is the realistic use case: wall time per realization.

| N | realizations | old (serial) | new (auto layout) | speed-up |
|---|---|---|---|---|
| 16 | 200 | 29.3 ms | 11.9 ms | 2.5x |
| 20 | 100 | 205 ms | 48 ms | 4.2x |
| 22 | 50 | 315 ms | 145 ms | 2.2x |
| 24 | 20 | 2.93 s | 0.54 s | **5.4x** |
| 26 | 4 | 11.3 s | 4.75 s | 2.4x |
| 28 | 4 | 148.6 s | 67.4 s | 2.2x |

Example: the default run (N=22, 200 realizations, all validation tests)
takes **34 s** in total, of which 28 s are the realizations.

---

## 5. Maximum practical N

`--auto-max-N` with the default limits (300 s per realization, 75% of
RAM) gives:

```
Largest successfully completed N: 28
  N=30: ~5.9 min per realization, ~4.3 GB RAM -> feasible but slow
  N=32: ~23.6 min per realization, ~8.3 GB RAM -> feasible but slow
  N=34: ~378.0 min per realization, ~64.4 GB RAM -> UNSAFE (RAM)
```

The extrapolation is reliable: the measured values at N=30 (5.2 min) and
N=32 (24.6 min) match it.

| N | status on this machine (15 GB, 4 cores) | time per realization | RAM |
|---|---|---|---|
| <= 28 | **benchmarked**; many realizations are practical | <= 1.5 min (67 s with 4 in parallel) | <= 1.2 GB per job |
| 30 | **benchmarked** | 5.2 min | 4.5 GB |
| 32 | **benchmarked** (one realization) | 24.6 min | 8.8 GB |
| 34 | **unsafe**: one complex block of dimension 65536 needs 69 GB (34 GB even in packed storage) | ~6 h (estimated) | 64 GB |

On a typical laptop:
- **8 GB RAM:** N <= 30, and N=30 only one realization at a time.
- **16 GB RAM:** N <= 32. That is the absolute full-spectrum limit, and the
  real GOE basis is what makes it possible, because it halves the memory.
- **Around 200 realizations in a day:** N <= 28 (4 x 1.2 GB in parallel).
- **About 50 realizations overnight:** N = 30.

N=34 and above need a different method. The spectrum is only accessible
partially: Lanczos for the lowest levels, KPM for the density of states
(both in `syk_spectrum.py`).

---

## 6. Validation (runs automatically at every start)

| Test | Result |
|---|---|
| Pauli algebra: Hermitian, P^2 = I, anticommutators, XY = iZ | PASS |
| Clifford algebra {chi_a, chi_b} = delta_ab, all pairs, at the target N | PASS (exact) |
| Majorana matrices identical to the original kron construction (N=8) | PASS |
| N=8 matrix elements, fast vs. dense | PASS, max difference 0 |
| N=8 eigenvalues, fast (real basis) vs. dense | PASS, 2e-16 |
| H = H^dagger at the target N, realization 0, every state (no matrix built) | PASS, 0 |
| [H, T] = 0 at the target N (justifies every symmetry reduction) | PASS, 0 |
| T^2 = (-1)^(n(n-1)/2) | PASS |
| N=8, 16: eigenvalues of the real GOE block == complex block, R = R^T | PASS, 6e-15 |
| N=10, 14: odd-block spectrum == even-block spectrum | PASS, 1e-15 |
| N=12: every level doubly degenerate | PASS, 9e-16 |
| Same couplings -> same spectrum as `majorana_SYK_reference.py`, 13 realizations, N=8..18 | PASS, 7e-15 |
| Same seed -> bitwise identical couplings and spectrum | PASS |
| Realization i independent of execution order and job count | PASS |
| Numba kernels == NumPy fallback kernels | PASS, 0 |
| 2-stage LAPACK == SciPy | PASS |

Benchmark mode also compares E0 between the old and new code, for the same
couplings, at N=16–26 (and 28). The differences are between 1e-16 and
1e-14.

---

## 7. The physics behind the optimizations

**Parity.** Every term in H is a product of four Majoranas, so it flips an
even number of fermion occupations. H therefore commutes with
(-1)^F = product of Z_K. It is block diagonal, with two blocks of dimension
D = 2^(N/2 - 1). Diagonalizing two blocks of size D instead of one of size
2D saves 4x in time and 4x in memory.

**Why N mod 8 matters.** Take U = the product of the N/2 even Majoranas
chi_0 chi_2 ... chi_{N-2} (a real signed permutation that flips every
qubit), and K = complex conjugation. The even Majoranas are real and the
odd ones are imaginary. As a result, T = U K maps chi_a to
(-1)^(n-1) chi_a for every a, where n = N/2. Since H has four Majoranas per
term and real couplings, T H T^-1 = H. The type of T depends on n:
- U flips all n qubits, so T keeps fermion parity when n is even and swaps
  the two parity blocks when n is odd.
- T^2 = U^2 = (-1)^(n(n-1)/2).

This gives three cases, repeating with period 8 in N:
- **N mod 8 = 2, 6** (n odd): T is an antiunitary map from the even block
  to the odd block that commutes with H. The two blocks have identical
  spectra, so only one is computed. Inside a block there is no antiunitary
  symmetry left, so the class is **GUE** (beta = 2).
- **N mod 8 = 0** (T^2 = +1 inside each block): the vectors
  a_r = (|r> + sigma_r |r̄>)/sqrt2 and b_r = i(|r> - sigma_r |r̄>)/sqrt2
  satisfy T a = a and T b = b. In this basis every matrix element is real,
  so H is a real symmetric matrix. That is the 4x saving. The class is
  **GOE** (beta = 1).
- **N mod 8 = 4** (T^2 = -1 inside each block): Kramers' theorem gives
  exactly doubly degenerate levels. The class is **GSE** (beta = 4).

**Why the random-matrix class matters.** The level-repulsion exponent beta
sets how level spacings are distributed. The mean spacing ratio ⟨r⟩
therefore depends on the class: GOE 0.5307, GUE 0.5996, GSE 0.6744, and
Poisson 0.3863 for uncorrelated levels. A result has to be compared with the
right class for its N.

**Why ⟨r⟩ must respect the symmetries.** Levels from different symmetry
sectors don't repel, so combining sectors shifts ⟨r⟩ towards Poisson. Exact
degeneracies give a spacing of zero, so r = 0 for those levels. The program
therefore computes r separately in each independent sector. For N mod 8 = 4
it first removes the Kramers partner of each level. For N mod 8 = 2, 6 it
uses only the even block, because the odd block is a copy. The full spectrum
(both blocks, all degeneracies) is kept for the density of states and the
`.npy` file.
