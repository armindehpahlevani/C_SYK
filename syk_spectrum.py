#!/usr/bin/env python3
"""
syk_spectrum.py -- eigenvalues and density of states of the Majorana SYK model
===============================================================================

A self-contained Python replacement for the exact-diagonalization part of the
C++ code in this repository, designed to run on an ordinary laptop.

Model (same conventions as MajoranaKitaevHamiltonian.cc in this repo):

    H = sum_{i<j<k<l} J_ijkl  chi_i chi_j chi_k chi_l ,   {chi_a, chi_b} = delta_ab
    J_ijkl  Gaussian, mean 0, variance  3! J^2 / N^3

N Majoranas -> N/2 complex fermions -> Hilbert space dimension 2^(N/2).
H conserves fermion parity, so it splits into an even and an odd block of
size 2^(N/2 - 1) each; we build and diagonalize each block separately.

Three modes
-----------
  full     all eigenvalues (dense diagonalization).        Practical: N <= 30
  lowest   the k lowest eigenvalues (Lanczos, scipy eigsh). Practical: N <= 34
  kpm      density of states via the Kernel Polynomial     Practical: N <= 34
           Method (Chebyshev expansion, no diagonalization)

Measured cost (one disorder sample, 4-core machine, 16 GB RAM):
  N=24  full: ~3 s           N=26  full: ~12 s          N=28  full: ~2 min
  N=30  full: ~10 min, ~5 GB RAM      N=30  lowest (k=4): ~45 s
  N=32  kpm (300 moments): ~5 min, ~2 GB    N=34: roughly 5-10x slower than N=32
N=48 is NOT reachable on a laptop in any language: a single parity block has
dimension 2^23 = 8.4 million and the Hamiltonian has ~10^11 nonzero entries.

Examples
--------
  python syk_spectrum.py --N 24 --samples 20                 # full spectrum, 20 samples
  python syk_spectrum.py --N 28 --samples 5 --plot           # + histogram plot
  python syk_spectrum.py --N 32 --mode lowest --k 10         # 10 lowest eigenvalues
  python syk_spectrum.py --N 32 --mode kpm --moments 400     # density of states

Requirements:  pip install numpy scipy matplotlib
"""

import argparse
import math
import os
import time
from itertools import combinations

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ---------------------------------------------------------------------------
# Fock-space helpers
# ---------------------------------------------------------------------------

def popcount(x):
    """Number of 1-bits of every entry of an integer array."""
    if hasattr(np, "bitwise_count"):          # numpy >= 2.0
        return np.bitwise_count(x).astype(np.int64)
    x = x.astype(np.int64)
    count = np.zeros_like(x)
    while np.any(x):
        count += x & 1
        x = x >> 1
    return count


def majorana_action(a, states):
    """
    Action of chi_a on Fock basis states |s> (Jordan-Wigner):
        chi_a |s> = factor * sign(s) * |s XOR mask>
    Returns (mask, factor, sign array of +-1).
    chi_{2j}   = (c_j + c_j^dag) / sqrt2
    chi_{2j+1} = i (c_j - c_j^dag) / sqrt2
    """
    j = a // 2
    mask = 1 << j
    string = 1 - 2 * (popcount(states & (mask - 1)) & 1)    # JW string
    if a % 2 == 0:
        return mask, 1 / math.sqrt(2), string
    occupied = (states >> j) & 1
    # c|1>=|0>, c^dag|0>=|1>  ->  (c - c^dag) gives +1 on occupied, -1 on empty
    return mask, 1j / math.sqrt(2), string * (2 * occupied - 1)


def pair_tables(N):
    """
    For every pair a<b precompute the action of chi_a chi_b on ALL basis
    states:  chi_a chi_b |s> = factor_ab * sign_ab(s) |s XOR mask_ab>.
    Signs are stored as int8 to save memory.
    """
    dim = 2 ** (N // 2)
    s = np.arange(dim, dtype=np.int64)
    single = [majorana_action(a, s) for a in range(N)]
    pairs = {}
    for a, b in combinations(range(N), 2):
        mb, fb, sb = single[b]
        ma, fa, sa = single[a]
        sign = sb * sa[s ^ mb]                     # chi_b first, then chi_a
        pairs[(a, b)] = (ma ^ mb, fa * fb, sign.astype(np.int8))
    return pairs


def random_couplings(N, J, rng):
    """Gaussian J_ijkl for i<j<k<l with variance 3! J^2 / N^3."""
    sigma = math.sqrt(6.0 * J ** 2 / N ** 3)
    quads = list(combinations(range(N), 4))
    return quads, rng.normal(0.0, sigma, size=len(quads))


# ---------------------------------------------------------------------------
# Hamiltonian construction (one parity block)
# ---------------------------------------------------------------------------

def build_block(N, quads, Js, pairs, parity):
    """
    Build the Hamiltonian restricted to states of given fermion parity
    (0 = even, 1 = odd) as a scipy sparse CSR matrix.

    Every term chi_i chi_j chi_k chi_l maps a basis state to one other basis
    state (s -> s XOR mask).  Terms sharing the same mask are summed into one
    coefficient array, so the matrix has (#masks) x (block dim) nonzeros.
    """
    dim = 2 ** (N // 2)
    all_states = np.arange(dim, dtype=np.int64)
    block = all_states[(popcount(all_states) & 1) == parity]
    position = np.full(dim, -1, dtype=np.int64)
    position[block] = np.arange(len(block))

    coef = {}                                   # mask -> coefficient array
    for (i, j, k, l), Jv in zip(quads, Js):
        m_kl, f_kl, s_kl = pairs[(k, l)]
        m_ij, f_ij, s_ij = pairs[(i, j)]
        mask = m_kl ^ m_ij
        # chi_i chi_j chi_k chi_l |s> = f_ij f_kl s_kl(s) s_ij(s^m_kl) |s^mask>
        val = (Jv * f_ij * f_kl) * (s_kl[block] * s_ij[block ^ m_kl])
        if mask in coef:
            coef[mask] += val
        else:
            coef[mask] = val.astype(np.complex128)

    D = len(block)
    cols = np.arange(D)
    rows_all, cols_all, data_all = [], [], []
    for mask, val in coef.items():
        rows_all.append(position[block ^ mask])
        cols_all.append(cols)
        data_all.append(val)
    H = sp.csr_matrix(
        (np.concatenate(data_all),
         (np.concatenate(rows_all), np.concatenate(cols_all))),
        shape=(D, D))
    return H


def blocks_needed(N):
    """
    Particle-hole symmetry: for N mod 8 = 2 or 6 the even and odd blocks have
    identical spectra, so only one needs to be computed.
    """
    return [0] if N % 8 in (2, 6) else [0, 1]


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------

def full_spectrum(H):
    return scipy.linalg.eigvalsh(H.toarray(), overwrite_a=True,
                                 check_finite=False)


def lowest_eigenvalues(H, k):
    vals = spla.eigsh(H, k=k, which="SA", return_eigenvectors=False)
    return np.sort(vals)


def kpm_moments(H, n_moments, n_random, rng):
    """
    Chebyshev moments mu_n = Tr T_n(H~) / D of the rescaled Hamiltonian
    H~ = (H - b)/a, estimated with random-phase vectors.
    """
    D = H.shape[0]
    emax = spla.eigsh(H, k=1, which="LA", return_eigenvectors=False,
                      tol=1e-4)[0]
    emin = spla.eigsh(H, k=1, which="SA", return_eigenvectors=False,
                      tol=1e-4)[0]
    a = (emax - emin) / (2 * 0.98)              # small safety margin
    b = (emax + emin) / 2
    Ht = (H - b * sp.identity(D, format="csr")) / a

    mu = np.zeros(n_moments)
    half = (n_moments + 1) // 2
    for _ in range(n_random):
        v0 = np.exp(2j * np.pi * rng.random(D)) / math.sqrt(D)
        v1 = Ht @ v0
        m0 = np.vdot(v0, v0).real
        m1 = np.vdot(v0, v1).real
        mu[0] += m0
        if n_moments > 1:
            mu[1] += m1
        # doubling trick with v_n = T_n(H~) v_0:
        #   mu_{2n} = 2<v_n|v_n> - mu_0,   mu_{2n+1} = 2<v_n|v_{n+1}> - mu_1
        for n in range(1, half):
            if 2 * n < n_moments:
                mu[2 * n] += 2 * np.vdot(v1, v1).real - m0
            if 2 * n + 1 < n_moments:
                v2 = 2 * (Ht @ v1) - v0
                mu[2 * n + 1] += 2 * np.vdot(v1, v2).real - m1
                v0, v1 = v1, v2
    return mu / n_random, a, b


def kpm_density(mu, a, b, n_points=1000):
    """Reconstruct the normalized density of states with a Jackson kernel."""
    M = len(mu)
    n = np.arange(M)
    g = ((M - n + 1) * np.cos(np.pi * n / (M + 1))
         + np.sin(np.pi * n / (M + 1)) / np.tan(np.pi / (M + 1))) / (M + 1)
    x = np.cos(np.pi * (np.arange(n_points) + 0.5) / n_points)
    T = np.cos(np.outer(np.arccos(x), n))            # T_n(x)
    coeffs = mu * g
    coeffs[1:] *= 2
    rho_x = (T @ coeffs) / (np.pi * np.sqrt(1 - x ** 2))
    E = a * x + b
    rho_E = rho_x / a
    order = np.argsort(E)
    return E[order], rho_E[order]


# ---------------------------------------------------------------------------
# Spectral statistics
# ---------------------------------------------------------------------------

def mean_r_ratio(evs):
    """
    Mean ratio of consecutive level spacings <r> (Oganesyan-Huse), computed
    within one symmetry block.  Reference values: Poisson 0.386,
    GOE 0.536, GUE 0.603, GSE 0.676.  Exact degeneracies are removed first.
    """
    evs = np.sort(evs)
    evs = evs[np.concatenate(([True], np.diff(evs) > 1e-10 * np.ptp(evs)))]
    s = np.diff(evs)
    if len(s) < 2:
        return float("nan")
    r = np.minimum(s[1:], s[:-1]) / np.maximum(s[1:], s[:-1])
    return float(np.mean(r))


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Eigenvalues / density of states of the SYK model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--N", type=int, default=24,
                   help="number of Majorana fermions (even)")
    p.add_argument("--J", type=float, default=1.0, help="coupling strength")
    p.add_argument("--samples", type=int, default=1,
                   help="number of disorder realizations")
    p.add_argument("--seed", type=int, default=None, help="random seed")
    p.add_argument("--mode", choices=["full", "lowest", "kpm"], default="full")
    p.add_argument("--k", type=int, default=6,
                   help="[lowest] eigenvalues per parity block")
    p.add_argument("--moments", type=int, default=300,
                   help="[kpm] number of Chebyshev moments (energy resolution)")
    p.add_argument("--random-vectors", type=int, default=5,
                   help="[kpm] random vectors for the stochastic trace")
    p.add_argument("--bins", type=int, default=100,
                   help="histogram bins for the density of states")
    p.add_argument("--out", default="syk_results", help="output directory")
    p.add_argument("--plot", action="store_true",
                   help="save a density-of-states plot (needs matplotlib)")
    args = p.parse_args()

    N = args.N
    if N % 2 or N < 4:
        p.error("N must be an even number >= 4")
    if args.mode == "full" and N > 30:
        print(f"WARNING: full diagonalization at N={N} needs a dense "
              f"{2**(N//2-1)}^2 matrix "
              f"({16 * 4**(N//2-1) / 1e9:.0f} GB). Consider --mode kpm.")

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    parities = blocks_needed(N)

    t0 = time.time()
    pairs = pair_tables(N)
    print(f"N={N}: Hilbert space 2^{N//2} = {2**(N//2)}, "
          f"parity block dim {2**(N//2-1)}, {math.comb(N, 4)} couplings")
    print(f"blocks computed: {['even', 'odd'][0:len(parities)]}"
          + ("  (odd block = even block by symmetry)" if len(parities) == 1
             else ""))

    all_evs = []            # full / lowest mode
    kpm_rho = []            # kpm mode
    for sample in range(args.samples):
        quads, Js = random_couplings(N, args.J, rng)
        evs_sample = []
        for parity in parities:
            ts = time.time()
            H = build_block(N, quads, Js, pairs, parity)
            tb = time.time() - ts
            if args.mode == "full":
                evs = full_spectrum(H)
            elif args.mode == "lowest":
                evs = lowest_eigenvalues(H, args.k)
            else:
                mu, a, b = kpm_moments(H, args.moments,
                                       args.random_vectors, rng)
                kpm_rho.append((mu, a, b))
                evs = None
            label = "even" if parity == 0 else "odd"
            msg = (f"  sample {sample+1}/{args.samples} {label}: "
                   f"build {tb:.1f}s, solve {time.time()-ts-tb:.1f}s")
            if evs is not None:
                msg += f", E0/N = {evs.min()/N:.5f}"
                if args.mode == "full":
                    msg += f", <r> = {mean_r_ratio(evs):.3f}"
                evs_sample.append(evs)
            print(msg, flush=True)
            del H

        if evs_sample:
            if len(parities) == 1:              # add the identical odd block
                evs_sample = evs_sample * 2
            evs_all = np.sort(np.concatenate(evs_sample))
            all_evs.append(evs_all)
            np.savetxt(os.path.join(args.out,
                                    f"evs_N{N}_{args.mode}_s{sample}.txt"),
                       evs_all, header=f"SYK eigenvalues N={N} J={args.J}")

    # ---- disorder-averaged density of states -----------------------------
    if args.mode == "full":
        evs = np.concatenate(all_evs)
        hist, edges = np.histogram(evs, bins=args.bins, density=True)
        E = 0.5 * (edges[1:] + edges[:-1])
        rho = hist
        e0 = np.array([e.min() for e in all_evs])
        print(f"\nground state E0/N = {e0.mean()/N:.5f} "
              f"+- {e0.std()/N/math.sqrt(len(e0)):.5f}  "
              f"(large-N limit: -0.04063)")
    elif args.mode == "kpm":
        # average densities on a common energy grid
        lo = min(b - a for _, a, b in kpm_rho)
        hi = max(b + a for _, a, b in kpm_rho)
        E = np.linspace(lo, hi, 1000)
        rho = np.zeros_like(E)
        for mu, a, b in kpm_rho:
            Ek, rk = kpm_density(mu, a, b)
            rho += np.interp(E, Ek, rk, left=0, right=0)
        rho /= len(kpm_rho)
    else:
        lows = np.array([e[:args.k] for e in all_evs])
        print("\nlowest eigenvalues (disorder average):")
        print(np.array2string(lows.mean(axis=0), precision=6))
        E = rho = None

    if E is not None:
        path = os.path.join(args.out, f"dos_N{N}_{args.mode}.txt")
        np.savetxt(path, np.column_stack([E, rho]),
                   header=f"E  rho(E)   (normalized to 1), N={N}, "
                          f"{args.samples} samples")
        print(f"density of states written to {path}")
        if args.plot:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(E, rho, lw=1.5)
            ax.set_xlabel("E / J")
            ax.set_ylabel(r"$\rho(E)$")
            ax.set_title(f"SYK density of states, N={N}, "
                         f"{args.samples} sample(s), {args.mode}")
            fig.tight_layout()
            png = os.path.join(args.out, f"dos_N{N}_{args.mode}.png")
            fig.savefig(png, dpi=150)
            print(f"plot saved to {png}")

    print(f"total time {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
