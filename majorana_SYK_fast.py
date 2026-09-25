#!/usr/bin/env python3
"""
majorana_SYK_fast.py -- full exact diagonalization of the q=4 Majorana SYK
model, optimized for an ordinary laptop (CPU only).

Model (unchanged from majorana_SYK_reference.py)
------------------------------------------------
    chi_{2K}   = Z_0 ... Z_{K-1} X_K / sqrt2
    chi_{2K+1} = Z_0 ... Z_{K-1} Y_K / sqrt2          {chi_a, chi_b} = delta_ab
    H = sum_{i<j<k<l} J_ijkl chi_i chi_j chi_k chi_l,  <J_ijkl^2> = 3! J^2 / N^3

Architecture (measurements and reasoning: SYK_PERFORMANCE_REPORT.md)
--------------------------------------------------------------------
config -> validation -> precomputation (Pauli strings, masks, basis, symmetry
maps; once per N) -> per realization: couplings -> Numba assembly directly into
the final Fortran-ordered LAPACK array -> symmetry-reduced dense
diagonalization -> spectral statistics -> streaming to disk.

Symmetry reduction used (all proven in the validation output, not assumed)
-------------------------------------------------------------------------
Fermion parity splits H into two blocks of dimension D = 2^(N/2-1).  The
antiunitary operator T = U K (K = complex conjugation, U = product of all
even Majoranas) commutes with H and T^2 = (-1)^(n(n-1)/2), n = N/2:
  N mod 8 = 0 : T keeps parity, T^2=+1 -> each block is REAL symmetric in a
                T-invariant basis (GOE).  Both blocks diagonalized as real
                matrices: ~4x faster and half the memory of complex.
  N mod 8 = 2,6: T maps even <-> odd -> the odd block has exactly the even
                block's spectrum (GUE).  Only the even block is diagonalized.
  N mod 8 = 4 : T keeps parity, T^2=-1 -> Kramers pairs in each block (GSE).
                Both blocks diagonalized; one level per pair used for <r>.

Usage
-----
    python majorana_SYK_fast.py                         # defaults below
    python majorana_SYK_fast.py --N 26 --realizations 100
    python majorana_SYK_fast.py --benchmark             # timing table
    python majorana_SYK_fast.py --auto-max-N            # find laptop limit
    python majorana_SYK_fast.py --help                  # all options

Requirements:  pip install numpy scipy matplotlib numba
Optional:      pip install psutil threadpoolctl   (better RAM / BLAS info)
"""

# =============================================================================
# CONFIGURATION (command-line options override these)
# =============================================================================
N = 22                        # number of Majoranas (even, >= 8)
J = 1.0                       # coupling strength
SEED = 12345                  # master seed; realization i uses stream i
N_REALIZATIONS = 200          # disorder realizations

N_JOBS = None                 # parallel realizations; None = automatic
BLAS_THREADS = None           # BLAS/Numba threads per job; None = automatic

MAX_MEMORY_FRACTION = 0.75    # never plan to use more of the available RAM
MAX_RUNTIME_PER_REALIZATION = 300.0   # seconds, for --auto-max-N
AUTO_BENCHMARK = False        # True = behave like --auto-max-N
BENCHMARK_N = [16, 18, 20, 22, 24, 26, 28]
OLD_BENCHMARK_MAX_N = 26      # also time the reference code up to this N
SHOW_PLOT = True              # open the histogram window at the end
# =============================================================================

import argparse
import ctypes
import glob
import itertools
import json
import math
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
import scipy
import scipy.linalg

try:
    from numba import njit, prange
    import numba
    HAVE_NUMBA = True
except ImportError:                                   # pure NumPy fallback
    HAVE_NUMBA = False

# Mean spacing ratio for large random matrices (Atas et al., PRL 110, 084101
# (2013), numerical large-N values; the 3x3 "surmise" values 0.536/0.603/0.676
# printed by the old code are slightly off).
R_REFERENCE = {"Poisson": 0.3863, "GOE": 0.5307, "GUE": 0.5996, "GSE": 0.6744}

TWO_STAGE_MIN_DIM = 4096      # below this, SciPy's driver is as fast (measured)
BASE_PROCESS_BYTES = 250e6    # Python + NumPy + SciPy + Numba per process


# =============================================================================
# SYSTEM INFORMATION
# =============================================================================

def cpu_name():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def physical_cores():
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return n
    except ImportError:
        pass
    return os.cpu_count() or 1


def available_memory():
    """Bytes of RAM currently available (None if unknown)."""
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        pass
    if sys.platform == "win32":
        class MS(ctypes.Structure):
            _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong),
                        ("total", ctypes.c_ulonglong),
                        ("avail", ctypes.c_ulonglong)] + \
                       [(f"x{i}", ctypes.c_ulonglong) for i in range(5)]
        m = MS()
        m.len = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.avail
    return None


def peak_rss():
    """Peak resident memory of this process in bytes (None if unknown)."""
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r if sys.platform == "darwin" else r * 1024
    except ImportError:
        pass
    try:
        import psutil
        return psutil.Process().memory_info().peak_wset
    except (ImportError, AttributeError):
        return None


def blas_description():
    try:
        from threadpoolctl import threadpool_info
        libs = [f"{i['internal_api']} {i.get('version', '')} "
                f"({i['num_threads']} threads)"
                for i in threadpool_info() if i["user_api"] == "blas"]
        if libs:
            return "; ".join(sorted(set(libs)))
    except ImportError:
        pass
    try:
        cfg = np.show_config(mode="dicts")
        b = cfg["Build Dependencies"]["blas"]
        return f"{b.get('name')} {b.get('version', '')} (NumPy build)"
    except Exception:
        return "unknown (pip install threadpoolctl for details)"


def fmt_bytes(b):
    if b is None:
        return "unknown"
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if abs(b) < 1024 or unit == "PB":
            return f"{b:.1f} {unit}" if unit != "B" else f"{b:.0f} B"
        b /= 1024


def set_thread_env(threads):
    """Thread counts for BLAS/Numba in processes started AFTER this call."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
                "NUMBA_NUM_THREADS"):
        os.environ[var] = str(threads)


def limit_threads_here(threads):
    """Best-effort thread limit for the current process."""
    if HAVE_NUMBA:
        numba.set_num_threads(min(threads, numba.config.NUMBA_NUM_THREADS))
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=threads)
    except ImportError:
        return None


# =============================================================================
# SYMMETRY STRUCTURE
# =============================================================================

def symmetry_info(N):
    """Symmetry class of q=4 Majorana SYK with N Majoranas (n = N/2 qubits).

    T = U K commutes with H; U = product of the n even Majoranas (a real
    signed permutation flipping every bit), U^2 = (-1)^(n(n-1)/2); U keeps
    fermion parity iff n is even.
    """
    n = N // 2
    m = N % 8
    info = dict(mod8=m, U2=(-1) ** (n * (n - 1) // 2),
                T_keeps_parity=(n % 2 == 0))
    if m == 0:
        info.update(rmt="GOE", parities=(0, 1), real_basis=True,
                    mirror=False, kramers=False,
                    note="T^2=+1 inside each parity block: blocks are real "
                         "symmetric in a T-invariant basis; blocks independent")
    elif m == 4:
        info.update(rmt="GSE", parities=(0, 1), real_basis=False,
                    mirror=False, kramers=True,
                    note="T^2=-1 inside each parity block: every level is a "
                         "Kramers pair; blocks independent")
    else:
        info.update(rmt="GUE", parities=(0,), real_basis=False,
                    mirror=True, kramers=False,
                    note="T maps even <-> odd parity: odd-block spectrum = "
                         "even-block spectrum; only even block computed")
    return info


def sector_dtype_bytes(sym):
    return 8 if sym["real_basis"] else 16


def cost_units(N):
    """Relative diagonalization cost (real flops ~ D^3, complex ~ 4 D^3)."""
    sym = symmetry_info(N)
    D = 2 ** (N // 2 - 1)
    return len(sym["parities"]) * D ** 3 * (1 if sym["real_basis"] else 4)


def estimate_memory(N, jobs=1):
    """Planned peak bytes: one dense sector matrix per job + LAPACK work."""
    sym = symmetry_info(N)
    D = 2 ** (N // 2 - 1)
    item = sector_dtype_bytes(sym)
    per_job = D * D * item + 128 * D * item + 64 * 2 ** (N // 2) \
        + BASE_PROCESS_BYTES
    main = BASE_PROCESS_BYTES if jobs > 1 else 0     # coordinating process
    return jobs * per_job + main, D * D * item


# =============================================================================
# PAULI-STRING REPRESENTATION (precomputed once per N)
# =============================================================================
# Basis |s>, bit K of s = qubit K.  P(x, z) = X^x Z^z acts as
#     P(x, z)|s> = (-1)^popcount(z & s) |s ^ x>,
#     P(x1,z1) P(x2,z2) = (-1)^popcount(z1 & x2) P(x1^x2, z1^z2).

def parity_array(v):
    v = np.asarray(v, dtype=np.int64)
    if hasattr(np, "bitwise_count"):
        return (np.bitwise_count(v) & 1).astype(np.int64)
    v = v ^ (v >> 32)
    v = v ^ (v >> 16)
    v = v ^ (v >> 8)
    v = v ^ (v >> 4)
    v = v ^ (v >> 2)
    v = v ^ (v >> 1)
    return v & 1


def majorana_strings(N):
    """chi_a = c_a P(x_a, z_a)."""
    n = N // 2
    x = np.zeros(N, np.int64)
    z = np.zeros(N, np.int64)
    c = np.zeros(N, np.complex128)
    for K in range(n):
        x[2 * K] = x[2 * K + 1] = 1 << K
        z[2 * K] = (1 << K) - 1                     # Z..Z X
        z[2 * K + 1] = (1 << (K + 1)) - 1           # Z..Z (X Z), Y = i X Z
        c[2 * K] = 1 / math.sqrt(2)
        c[2 * K + 1] = 1j / math.sqrt(2)
    return x, z, c


def product_strings(x, z, c, idx):
    """Pauli string of chi_idx[:,0] chi_idx[:,1] ... for every row of idx."""
    ax = np.zeros(len(idx), np.int64)
    az = np.zeros(len(idx), np.int64)
    ac = np.ones(len(idx), np.complex128)
    for col in range(idx.shape[1]):
        a = idx[:, col]
        ac = ac * c[a] * (1 - 2 * parity_array(az & x[a]))
        ax ^= x[a]
        az ^= z[a]
    return ax, az, ac


class SYKStructure:
    """Everything that does not depend on the disorder realization."""

    def __init__(self, N):
        if N % 2 or N < 8:
            raise ValueError("N must be even and >= 8")
        self.N, self.n = N, N // 2
        self.dim = 1 << self.n
        self.D = self.dim >> 1
        self.sym = symmetry_info(N)
        self.n_terms = math.comb(N, 4)
        self.sigma_J = math.sqrt(6.0 / N ** 3)          # times J

        mx, mz, mc = majorana_strings(N)
        quads = np.array(list(itertools.combinations(range(N), 4)),
                         dtype=np.int64)
        qx, qz, qc = product_strings(mx, mz, mc, quads)

        # group the C(N,4) terms by flip mask x: one matrix element per group
        order = np.argsort(qx, kind="stable")
        masks, starts = np.unique(qx[order], return_index=True)
        self.term_order = order
        self.z = np.ascontiguousarray(qz[order])
        self.phase = np.ascontiguousarray(qc[order])
        self.gmask = masks.astype(np.int64)
        self.gstart = np.append(starts, len(order)).astype(np.int64)

        states = np.arange(self.dim, dtype=np.int64)
        par = parity_array(states)
        self.block_states = [np.ascontiguousarray(states[par == b])
                             for b in (0, 1)]
        self.pos = np.empty(self.dim, np.int64)
        for b in (0, 1):
            self.pos[self.block_states[b]] = np.arange(self.D)

        # antiunitary T = U K,  U|s> = sigma(s) |s ^ all_ones>
        self.all_ones = self.dim - 1
        ux, uz, uc = self._u_string(mx, mz)
        assert ux == self.all_ones and abs(uc.imag) < 1e-12
        self.sigma = (uc.real * (1 - 2 * parity_array(uz & states))
                      ).astype(np.float64)

        if self.sym["real_basis"]:
            top = 1 << (self.n - 1)
            self.top_bit = top
            self.reps = [np.ascontiguousarray(bs[(bs & top) == 0])
                         for bs in self.block_states]
            self.rep_idx = np.empty(self.dim, np.int64)
            for r in self.reps:
                self.rep_idx[r] = np.arange(len(r))
                self.rep_idx[r ^ self.all_ones] = np.arange(len(r))

    def _u_string(self, mx, mz):
        """U = prod_K (sqrt2 chi_{2K}) as a single Pauli string."""
        idx = np.arange(0, self.N, 2)[None, :]
        ones = np.ones(self.N, np.complex128)
        ux, uz, uc = product_strings(mx, mz, ones, idx)
        return int(ux[0]), int(uz[0]), complex(uc[0])

    def term_weights(self, couplings):
        """w_t = J_t * phase_t in mask-grouped order."""
        return couplings[self.term_order] * self.phase

    def build_sector(self, parity, w, real_basis=None):
        """Dense sector matrix, Fortran order, ready for in-place LAPACK."""
        real = self.sym["real_basis"] if real_basis is None else real_basis
        if real:
            A = np.zeros((self.D, self.D), np.float64, order="F")
            fill_goe_real(A, self.reps[parity], self.rep_idx, self.top_bit,
                          self.all_ones, self.sigma, self.gmask, self.gstart,
                          self.z, w)
        else:
            A = np.zeros((self.D, self.D), np.complex128, order="F")
            fill_complex(A, self.block_states[parity], self.pos, self.gmask,
                         self.gstart, self.z, w)
        return A


def realization_couplings(N, J_, seed, idx):
    """Independent, reproducible stream per realization index."""
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(idx,)))
    return rng.normal(scale=J_ * math.sqrt(6.0 / N ** 3),
                      size=math.comb(N, 4))


# =============================================================================
# HAMILTONIAN ASSEMBLY KERNELS (Numba, with a NumPy fallback)
# =============================================================================
# Matrix element <s^x|H|s> = sum_{t in group(x)} w_t (-1)^popcount(z_t & s).

def _fill_complex_np(H, states, pos, gmask, gstart, z, w):
    cols = np.arange(len(states))
    for g in range(len(gmask)):
        acc = np.zeros(len(states), np.complex128)
        for t in range(gstart[g], gstart[g + 1]):
            acc += w[t] * (1 - 2 * parity_array(z[t] & states))
        H[pos[states ^ gmask[g]], cols] = acc


def _fill_goe_real_np(R, reps, rep_idx, top_bit, all_ones, sigma, gmask,
                      gstart, z, w):
    D2 = len(reps)
    p = np.arange(D2)
    for g in range(len(gmask)):
        acc = np.zeros(D2, np.complex128)
        for t in range(gstart[g], gstart[g + 1]):
            acc += w[t] * (1 - 2 * parity_array(z[t] & reps))
        u = reps ^ gmask[g]
        q = rep_idx[u]
        # s = +1 if u is a representative, else sigma of its representative
        isrep = (u & top_bit) == 0
        sg = np.where(isrep, 1.0, sigma[u ^ all_ones])
        sb = np.where(isrep, 1.0, -1.0)
        re, im = acc.real, acc.imag
        R[q, p] += sg * re
        R[D2 + q, p] += sb * sg * im
        R[q, D2 + p] += -sg * im
        R[D2 + q, D2 + p] += sb * sg * re


def _check_symmetries_np(dim, all_ones, sigma, gmask, gstart, z, w):
    s = np.arange(dim, dtype=np.int64)
    herm = tsym = mx = 0.0

    def coef(g, st):
        acc = np.zeros(len(st), np.complex128)
        for t in range(gstart[g], gstart[g + 1]):
            acc += w[t] * (1 - 2 * parity_array(z[t] & st))
        return acc

    sb = s ^ all_ones
    for g in range(len(gmask)):
        x = gmask[g]
        h, ht, hb = coef(g, s), coef(g, s ^ x), coef(g, sb)
        ub = (s ^ x) ^ all_ones
        herm = max(herm, np.abs(h - ht.conj()).max())
        tsym = max(tsym, np.abs(h - sigma[ub] * sigma[sb] * hb.conj()).max())
        mx = max(mx, np.abs(h).max())
    return herm, tsym, mx


if HAVE_NUMBA:
    @njit(cache=True, inline="always")
    def _parity(v):
        v ^= v >> 32
        v ^= v >> 16
        v ^= v >> 8
        v ^= v >> 4
        v ^= v >> 2
        v ^= v >> 1
        return v & 1

    @njit(cache=True, inline="always")
    def _coef(g, s, gstart, z, w):
        acc = 0j
        for t in range(gstart[g], gstart[g + 1]):
            if _parity(z[t] & s):
                acc -= w[t]
            else:
                acc += w[t]
        return acc

    @njit(parallel=True, cache=True)
    def _fill_complex_nb(H, states, pos, gmask, gstart, z, w):
        for c in prange(states.shape[0]):
            s = states[c]
            for g in range(gmask.shape[0]):
                H[pos[s ^ gmask[g]], c] = _coef(g, s, gstart, z, w)

    @njit(parallel=True, cache=True)
    def _fill_goe_real_nb(R, reps, rep_idx, top_bit, all_ones, sigma, gmask,
                          gstart, z, w):
        # Real basis a_r = (|r> + sigma_r|rbar>)/sqrt2, b_r = i(|r> - ...)/sqrt2
        # (T a_r = a_r, T b_r = b_r).  With h = <u|H|r>, u = r^x:
        #   u = r'    : R[a',a] += Re h, R[b',a] += Im h,
        #               R[a',b] -= Im h, R[b',b] += Re h
        #   u = r'bar : same with factors sigma_r' * (1, -1, -1, -1)
        D2 = reps.shape[0]
        for p in prange(D2):
            r = reps[p]
            for g in range(gmask.shape[0]):
                h = _coef(g, r, gstart, z, w)
                u = r ^ gmask[g]
                q = rep_idx[u]
                re = h.real
                im = h.imag
                if (u & top_bit) == 0:
                    R[q, p] += re
                    R[D2 + q, p] += im
                    R[q, D2 + p] -= im
                    R[D2 + q, D2 + p] += re
                else:
                    sg = sigma[u ^ all_ones]
                    R[q, p] += sg * re
                    R[D2 + q, p] -= sg * im
                    R[q, D2 + p] -= sg * im
                    R[D2 + q, D2 + p] -= sg * re

    @njit(parallel=True, cache=True)
    def _check_symmetries_nb(dim, all_ones, sigma, gmask, gstart, z, w):
        herm = np.zeros(dim)
        tsym = np.zeros(dim)
        mx = np.zeros(dim)
        for s in prange(dim):
            sb = s ^ all_ones
            eh = 0.0
            et = 0.0
            m = 0.0
            for g in range(gmask.shape[0]):
                x = gmask[g]
                h = _coef(g, s, gstart, z, w)            # <s^x|H|s>
                ht = _coef(g, s ^ x, gstart, z, w)       # <s|H|s^x>
                hb = _coef(g, sb, gstart, z, w)          # <sb^x|H|sb>
                ub = (s ^ x) ^ all_ones
                eh = max(eh, abs(h - ht.conjugate()))
                et = max(et, abs(h - sigma[ub] * sigma[sb] * hb.conjugate()))
                m = max(m, abs(h))
            herm[s] = eh
            tsym[s] = et
            mx[s] = m
        return herm.max(), tsym.max(), mx.max()

    fill_complex = _fill_complex_nb
    fill_goe_real = _fill_goe_real_nb
    check_symmetries_kernel = _check_symmetries_nb
else:
    fill_complex = _fill_complex_np
    fill_goe_real = _fill_goe_real_np
    check_symmetries_kernel = _check_symmetries_np


def use_numpy_kernels():
    global fill_complex, fill_goe_real, check_symmetries_kernel
    fill_complex = _fill_complex_np
    fill_goe_real = _fill_goe_real_np
    check_symmetries_kernel = _check_symmetries_np


# =============================================================================
# DENSE EIGENSOLVER (LAPACK, eigenvalues only, in place)
# =============================================================================

class Eigensolver:
    """Eigenvalues of a dense symmetric/Hermitian Fortran-ordered matrix.

    Uses LAPACK's 2-stage tridiagonalization (dsyevd_2stage / zheevd_2stage,
    1.4-1.5x faster at D >= 4096) when the BLAS library shipped with
    SciPy/NumPy exports it; it is validated against SciPy at start-up.
    Otherwise SciPy's eigvalsh.  The input matrix is destroyed.
    """

    def __init__(self, allow_two_stage=True):
        self.two_stage = {}
        self.two_stage_lib = None
        if allow_two_stage:
            try:
                self._detect()
            except Exception:
                self.two_stage = {}

    def description(self):
        if self.two_stage:
            return (f"LAPACK 2-stage (d/z syevd_2stage) for D >= "
                    f"{TWO_STAGE_MIN_DIM}, SciPy eigvalsh below; "
                    f"{os.path.basename(self.two_stage_lib)}")
        return "SciPy eigvalsh (2-stage LAPACK not found)"

    @staticmethod
    def _candidate_libs():
        paths = []
        try:
            from threadpoolctl import threadpool_info
            paths += [i["filepath"] for i in threadpool_info()
                      if i["user_api"] == "blas"]
        except ImportError:
            pass
        for mod in (scipy, np):
            base = os.path.dirname(mod.__file__)
            for pat in ("../*.libs/*openblas*", ".dylibs/*openblas*",
                        "../*.libs/*mkl*", "*.libs/*openblas*"):
                paths += glob.glob(os.path.join(base, pat))
        seen, out = set(), []
        for p in paths:
            p = os.path.realpath(p)
            if p not in seen and os.path.isfile(p):
                seen.add(p)
                out.append(p)
        return out

    def _detect(self):
        for path in self._candidate_libs():
            try:
                lib = ctypes.CDLL(path)
            except OSError:
                continue
            found = {}
            for kind, routine in (("d", "dsyevd_2stage"),
                                  ("z", "zheevd_2stage")):
                for name, itype in ((f"scipy_{routine}_", ctypes.c_int32),
                                    (f"{routine}_", ctypes.c_int32),
                                    (f"scipy_{routine}_64_", ctypes.c_int64),
                                    (f"{routine}_64_", ctypes.c_int64)):
                    fn = getattr(lib, name, None)
                    if fn is not None:
                        found[kind] = (fn, itype)
                        break
            if len(found) == 2:
                self.two_stage = found
                self.two_stage_lib = path
                if self._validate():
                    return
                self.two_stage = {}
        self.two_stage_lib = None

    def _validate(self):
        rng = np.random.default_rng(1)
        for kind in ("d", "z"):
            n = 300
            M = rng.normal(size=(n, n))
            if kind == "z":
                M = M + 1j * rng.normal(size=(n, n))
            M = np.asfortranarray(M + M.conj().T)
            ref = np.linalg.eigvalsh(M)
            got = np.sort(self._call_two_stage(M.copy(order="F")))
            if not np.allclose(got, ref, atol=1e-10 * np.abs(ref).max()):
                return False
        return True

    def _call_two_stage(self, A):
        n = A.shape[0]
        cplx = np.iscomplexobj(A)
        fn, itype = self.two_stage["z" if cplx else "d"]
        ival = lambda v: ctypes.byref(itype(v))
        ptr = lambda a: a.ctypes.data_as(ctypes.c_void_p)
        w = np.empty(n)
        info = itype(0)
        work = np.empty(1, A.dtype)
        rwork = np.empty(1)
        iwork = np.empty(1, np.int64 if itype is ctypes.c_int64 else np.int32)

        def call(lw, lr, li):
            args = [ctypes.c_char_p(b"N"), ctypes.c_char_p(b"L"), ival(n),
                    ptr(A), ival(n), ptr(w), ptr(work), ival(lw)]
            if cplx:
                args += [ptr(rwork), ival(lr)]
            args += [ptr(iwork), ival(li), ctypes.byref(info),
                     ctypes.c_size_t(1), ctypes.c_size_t(1)]
            fn(*args)

        call(-1, -1, -1)                                   # workspace query
        lw, lr, li = int(work[0].real), int(rwork[0]), int(iwork[0])
        work = np.empty(max(lw, 1), A.dtype)
        rwork = np.empty(max(lr, 1))
        iwork = np.empty(max(li, 1), iwork.dtype)
        call(len(work), len(rwork), len(iwork))
        if info.value != 0:
            raise np.linalg.LinAlgError(f"2-stage LAPACK info={info.value}")
        return w

    def eigvalsh(self, A):
        if self.two_stage and A.shape[0] >= TWO_STAGE_MIN_DIM \
                and A.flags.f_contiguous:
            return np.sort(self._call_two_stage(A))
        return scipy.linalg.eigh(A, eigvals_only=True, overwrite_a=True,
                                 check_finite=False)


# =============================================================================
# ONE REALIZATION  (couplings -> assembly -> diagonalization -> statistics)
# =============================================================================

def r_ratios(levels):
    s = np.diff(levels)
    a, b = s[1:], s[:-1]
    big = np.maximum(a, b)
    ok = big > 0
    return np.minimum(a, b)[ok] / big[ok]


def sector_statistics(sym, e):
    """r values of one sorted sector spectrum (Kramers pairs removed)."""
    split = 0.0
    if sym["kramers"]:
        split = float(np.abs(e[1::2] - e[0::2]).max())
        e = e[0::2]
    r = r_ratios(e)
    m = len(r)
    rc = r[m // 4: 3 * m // 4]                      # central half of spectrum
    return r.sum(), len(r), rc.sum(), len(rc), split


def compute_realization(st, solver, J_, seed, idx):
    t0 = time.perf_counter()
    w = st.term_weights(realization_couplings(st.N, J_, seed, idx))
    t_build = time.perf_counter() - t0
    t_diag = 0.0
    sectors = []
    for b in st.sym["parities"]:
        t = time.perf_counter()
        A = st.build_sector(b, w)
        t_build += time.perf_counter() - t
        t = time.perf_counter()
        e = np.sort(solver.eigvalsh(A))
        del A                                        # release before next
        t_diag += time.perf_counter() - t
        sectors.append(e)

    stats = [sector_statistics(st.sym, e) for e in sectors]
    parts = sectors * 2 if st.sym["mirror"] else sectors
    spectrum = np.sort(np.concatenate(parts))
    return dict(idx=idx, spectrum=spectrum, E0=float(spectrum[0]),
                r_sum=sum(s[0] for s in stats), r_n=sum(s[1] for s in stats),
                rc_sum=sum(s[2] for s in stats),
                rc_n=sum(s[3] for s in stats),
                kramers_split=max(s[4] for s in stats),
                t_build=t_build, t_diag=t_diag)


def warm_up(solver, J_=1.0, seed=0):
    """Compile (first run) or load from cache (later runs) all kernels."""
    for Nw in (8, 10):                     # real GOE kernel, complex kernel
        compute_realization(SYKStructure(Nw), solver, J_, seed, 0)


# ---- worker processes (spawned; each builds its own precomputed tables) ----
_WORKER = {}


def _worker_init(N_, J_, seed, allow_two_stage, force_numpy):
    if force_numpy:
        use_numpy_kernels()
    _WORKER.update(st=SYKStructure(N_), solver=Eigensolver(allow_two_stage),
                   J=J_, seed=seed)
    warm_up(_WORKER["solver"], J_, seed)


def _worker_run(idx):
    w = _WORKER
    return compute_realization(w["st"], w["solver"], w["J"], w["seed"], idx)


# =============================================================================
# VALIDATION
# =============================================================================

def pauli_matrices():
    return {"I": np.eye(2, dtype=complex),
            "X": np.array([[0, 1], [1, 0]], dtype=complex),
            "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
            "Z": np.array([[1, 0], [0, -1]], dtype=complex)}


def kron_list(ms):
    out = ms[0]
    for m in ms[1:]:
        out = np.kron(out, m)
    return out


def dense_majoranas(N):
    """Original kron construction (qubit 0 = most significant index bit)."""
    p = pauli_matrices()
    n = N // 2
    return [(1 / np.sqrt(2)) * kron_list(
        [p["Z"] if i < K else p[P] if i == K else p["I"] for i in range(n)])
        for K in range(n) for P in ("X", "Y")]


def dense_reference_H(N, couplings):
    ch = dense_majoranas(N)
    H = np.zeros((2 ** (N // 2),) * 2, dtype=complex)
    for Jv, (i, j, k, l) in zip(couplings,
                                itertools.combinations(range(N), 4)):
        H += Jv * (ch[i] @ ch[j] @ ch[k] @ ch[l])
    return H


def bit_reverse(s, n):
    return int(format(s, f"0{n}b")[::-1], 2)


def full_complex_H(st, w):
    """Full 2^n x 2^n H in our basis from the two complex parity blocks."""
    H = np.zeros((st.dim, st.dim), complex)
    for b in (0, 1):
        bs = st.block_states[b]
        H[np.ix_(bs, bs)] = st.build_sector(b, w, real_basis=False)
    return H


def spectrum_new(st, solver, w):
    parts = [np.linalg.eigvalsh(st.build_sector(b, w))
             for b in st.sym["parities"]]
    if st.sym["mirror"]:
        parts = parts * 2
    return np.sort(np.concatenate(parts))


class Validator:
    def __init__(self):
        self.results = []

    def record(self, name, ok, detail=""):
        ok = None if ok is None else bool(ok)
        self.results.append((name, ok, detail))
        tag = "PASS" if ok is True else ("SKIP" if ok is None else "FAIL")
        print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""),
              flush=True)

    def failed(self):
        return [r for r in self.results if r[1] is False]


def run_validation(st, solver, J_, seed):
    V = Validator()
    tol = 1e-10

    # A. Pauli matrices
    p = pauli_matrices()
    I, X, Y, Z = p["I"], p["X"], p["Y"], p["Z"]
    ok = all(np.allclose(M, M.conj().T) for M in p.values()) \
        and all(np.allclose(M @ M, I) for M in (X, Y, Z)) \
        and all(np.allclose(A @ B + B @ A, 0)
                for A, B in ((X, Y), (X, Z), (Y, Z))) \
        and np.allclose(X @ Y, 1j * Z)
    V.record("Pauli algebra (Hermitian, P^2=I, anticommutation, XY=iZ)", ok)

    # C. Clifford algebra at the target N (exact, on the Pauli strings)
    x, z, c = majorana_strings(st.N)
    pc = lambda v: bin(int(v)).count("1") & 1
    ok = True
    for a in range(st.N):
        herm = np.isclose(np.conj(c[a]) * (-1) ** pc(x[a] & z[a]), c[a])
        square = np.isclose(c[a] ** 2 * (-1) ** pc(z[a] & x[a]), 0.5)
        ok &= bool(herm and square)
        for b in range(a + 1, st.N):
            ok &= (pc(z[a] & x[b]) + pc(x[a] & z[b])) % 2 == 1
    V.record(f"Clifford algebra {{chi_a,chi_b}}=delta_ab, all a,b, N={st.N}",
             bool(ok), f"{st.N * (st.N + 1) // 2} pairs, exact")

    # Majorana convention vs. dense kron construction (N=8)
    st8 = SYKStructure(8)
    dense = dense_majoranas(8)
    rev = np.array([bit_reverse(s, 4) for s in range(16)])
    x8, z8, c8 = majorana_strings(8)
    s = np.arange(16)
    ok = True
    for a in range(8):
        M = np.zeros((16, 16), complex)
        M[s ^ x8[a], s] = c8[a] * (1 - 2 * parity_array(z8[a] & s))
        ok &= np.allclose(M, dense[a][np.ix_(rev, rev)])
    V.record("Majorana matrices identical to the kron construction (N=8)", ok)

    # B. fast vs dense, N=8: matrix elements and eigenvalues
    Jc = realization_couplings(8, J_, seed, 0)
    w8 = st8.term_weights(Jc)
    H_ref = dense_reference_H(8, Jc)
    H_new = full_complex_H(st8, w8)
    err = np.abs(H_new - H_ref[np.ix_(rev, rev)]).max()
    V.record("N=8 Hamiltonian matrix elements, fast vs dense", err < tol,
             f"max |dH| = {err:.1e}")
    e_ref = np.linalg.eigvalsh(H_ref)
    e_new = spectrum_new(st8, solver, w8)
    err = np.abs(e_new - e_ref).max()
    V.record("N=8 eigenvalues, fast (real GOE basis) vs dense", err < tol,
             f"max |dE| = {err:.1e}")

    # D. Hermiticity + antiunitary symmetry at the target N (matrix-free)
    w = st.term_weights(realization_couplings(st.N, J_, seed, 0))
    t = time.perf_counter()
    herm, tsym, scale = check_symmetries_kernel(
        st.dim, st.all_ones, st.sigma, st.gmask, st.gstart, st.z, w)
    dt = time.perf_counter() - t
    V.record(f"H = H^dagger, N={st.N}, realization 0 (all {st.dim} states)",
             herm <= tol * scale, f"max err {herm:.1e}, {dt:.1f} s")
    V.record(f"[H, T] = 0 for T = U K, N={st.N}", tsym <= tol * scale,
             f"max err {tsym:.1e}")
    u2 = st.sigma * st.sigma[np.arange(st.dim) ^ st.all_ones]
    V.record(f"T^2 = {st.sym['U2']:+d} (N mod 8 = {st.sym['mod8']})",
             bool(np.all(u2 == st.sym["U2"])))

    # symmetry reductions on explicit small examples (all four classes)
    for Ns in (8, 10, 12, 14, 16):
        s2 = SYKStructure(Ns)
        ws = s2.term_weights(realization_couplings(Ns, J_, seed, 1))
        blocks = [np.linalg.eigvalsh(s2.build_sector(b, ws, real_basis=False))
                  for b in (0, 1)]
        sym = s2.sym
        if sym["real_basis"]:
            reals = [np.linalg.eigvalsh(s2.build_sector(b, ws))
                     for b in (0, 1)]
            R = s2.build_sector(0, ws)
            err = max(np.abs(r - c).max() for r, c in zip(reals, blocks))
            err = max(err, np.abs(R - R.T).max())
            msg = "real symmetric block == complex block"
        elif sym["mirror"]:
            err = np.abs(blocks[0] - blocks[1]).max()
            msg = "odd-block spectrum == even-block spectrum"
        else:
            err = max(np.abs(e[1::2] - e[0::2]).max() for e in blocks)
            msg = "every level doubly degenerate (Kramers)"
        V.record(f"N={Ns} ({sym['rmt']}): {msg}", err < tol,
                 f"max err {err:.1e}")

    # 5 & 7. old (reference file) vs new, same couplings
    try:
        import majorana_SYK_reference as ref
    except ImportError:
        ref = None
    if ref is None:
        V.record("reference implementation comparison", None,
                 "majorana_SYK_reference.py not found next to this file")
    else:
        worst, count = 0.0, 0
        for Ns, reps in ((8, 2), (10, 2), (12, 3), (14, 3), (16, 2), (18, 1)):
            s2 = SYKStructure(Ns)
            for k in range(reps):
                Jc = realization_couplings(Ns, J_, seed + 7, k)
                e_old = ref.diagonalize(ref.build_hamiltonian(Ns, Jc))
                e_new = spectrum_new(s2, solver, s2.term_weights(Jc))
                worst = max(worst, np.abs(e_old - e_new).max())
                count += 1
        V.record("same couplings -> same spectrum as majorana_SYK_reference",
                 worst < tol, f"{count} realizations, N=8..18, "
                              f"max |dE| = {worst:.1e}")

    # 6. reproducibility
    s12 = SYKStructure(12)
    a = compute_realization(s12, solver, J_, seed, 3)
    b = compute_realization(s12, solver, J_, seed, 3)
    same_J = np.array_equal(realization_couplings(12, J_, seed, 3),
                            realization_couplings(12, J_, seed, 3))
    V.record("same seed -> identical couplings and spectrum",
             same_J and np.array_equal(a["spectrum"], b["spectrum"]),
             "bitwise identical")
    c1 = compute_realization(s12, solver, J_, seed, 1)
    c3 = compute_realization(s12, solver, J_, seed, 3)
    V.record("realization i independent of execution order / job count",
             np.array_equal(c3["spectrum"], a["spectrum"])
             and not np.array_equal(c1["spectrum"], a["spectrum"]),
             "one random stream per realization index")

    # Numba kernels vs pure NumPy kernels
    if HAVE_NUMBA and fill_complex is _fill_complex_nb:
        worst = 0.0
        for Ns in (12, 16):
            s2 = SYKStructure(Ns)
            ws = s2.term_weights(realization_couplings(Ns, J_, seed, 2))
            for real in ((False, True) if s2.sym["real_basis"] else (False,)):
                A1 = s2.build_sector(0, ws, real_basis=real)
                A2 = np.zeros_like(A1, order="F")
                if real:
                    _fill_goe_real_np(A2, s2.reps[0], s2.rep_idx, s2.top_bit,
                                      s2.all_ones, s2.sigma, s2.gmask,
                                      s2.gstart, s2.z, ws)
                else:
                    _fill_complex_np(A2, s2.block_states[0], s2.pos, s2.gmask,
                                     s2.gstart, s2.z, ws)
                worst = max(worst, np.abs(A1 - A2).max())
        V.record("Numba kernels == NumPy kernels", worst < tol,
                 f"max diff {worst:.1e}")

    # eigensolver
    if solver.two_stage:
        V.record("2-stage LAPACK eigensolver agrees with SciPy", True,
                 "checked at start-up, real and complex")
    return V


# =============================================================================
# PARALLEL STRATEGY AND MEMORY SAFETY
# =============================================================================

def choose_parallelism(N_, n_real, jobs, threads, max_frac):
    """(jobs, threads per job), never jobs x threads > physical cores.

    Measured on 4 cores (per-realization wall time, jobs x threads):
      N=20:  4x1  62 ms   2x2  88 ms   1x4  122 ms
      N=24:  4x1 568 ms   2x2 638 ms   1x4  779 ms
      N=26:  4x1 4.8 s    2x2 5.2 s    1x4  6.2 s
      N=28:  4x1 67 s                  1x4  83 s
    Dense eigensolvers scale poorly with threads, so one realization per
    core wins whenever the matrices fit in RAM.  The number of jobs is
    therefore limited by n_realizations and by memory only.
    """
    cores = physical_cores()
    if jobs is None:
        jobs = max(1, min(cores, n_real))
        avail = available_memory()
        if avail is not None:
            per_job = estimate_memory(N_, 1)[0]
            jobs = max(1, min(jobs, int(max_frac * avail // per_job)))
    if threads is None:
        threads = max(1, cores // jobs)
    return jobs, threads


def memory_check(N_, jobs, max_frac, verbose=True):
    need, matrix = estimate_memory(N_, jobs)
    avail = available_memory()
    if verbose:
        sym = symmetry_info(N_)
        D = 2 ** (N_ // 2 - 1)
        print(f"Memory estimate      : {fmt_bytes(need)} "
              f"({jobs} x [{D}x{D} "
              f"{'float64' if sym['real_basis'] else 'complex128'} = "
              f"{fmt_bytes(matrix)} + workspace])")
        print(f"Available RAM        : {fmt_bytes(avail)} "
              f"(limit {max_frac:.0%} = "
              f"{fmt_bytes(None if avail is None else max_frac * avail)})")
    if avail is not None and need > max_frac * avail:
        return False
    return True


# =============================================================================
# OUTPUT
# =============================================================================

def plot_histogram(evs, N_, n_real, show, bins=60):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(7, 5))
    plt.hist(evs, bins=bins, density=True, edgecolor="black", alpha=0.7)
    plt.xlabel("Energy E")
    plt.ylabel("Density of states")
    plt.title(f"SYK energy spectrum, N={N_}, {n_real} realizations")
    fname = f"syk_histogram_N{N_}.png"
    fig.savefig(fname, dpi=150)
    if show and plt.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)
    return fname


def print_spectrum(spec, N_):
    fname = f"syk_spectrum_realization0_N{N_}.txt"
    np.savetxt(fname, spec, header=f"all {len(spec)} eigenvalues, N={N_}, "
                                   f"realization 0")
    if len(spec) <= 4096:
        with np.printoptions(threshold=sys.maxsize, precision=6,
                             linewidth=100):
            print(spec)
    else:
        with np.printoptions(precision=6):
            print(spec)
        print(f"(complete list of {len(spec)} eigenvalues in {fname})")
    return fname


# =============================================================================
# MAIN RUN
# =============================================================================

def run(args):
    t_start = time.perf_counter()
    bar = "=" * 60
    N_, J_, seed, n_real = args.N, args.J, args.seed, args.realizations
    sym = symmetry_info(N_)
    jobs, threads = choose_parallelism(N_, n_real, args.jobs,
                                       args.blas_threads,
                                       args.max_memory_fraction)
    print(bar)
    print("Majorana SYK q=4 -- exact diagonalization (fast)")
    print(bar)
    print(f"N = {N_}   J = {J_}   realizations = {n_real}   seed = {seed}")
    print()
    print(f"CPU                  : {cpu_name()}")
    print(f"cores                : {physical_cores()} physical, "
          f"{os.cpu_count()} logical")
    print(f"BLAS                 : {blas_description()}")
    print(f"assembly kernels     : "
          f"{'Numba ' + numba.__version__ if HAVE_NUMBA and not args.no_numba else 'NumPy (install numba for speed)'}")
    print(f"parallel layout      : {jobs} job(s) x {threads} thread(s)")
    print()
    D = 2 ** (N_ // 2 - 1)
    print(f"Hilbert dimension    : 2^{N_ // 2} = {2 ** (N_ // 2)}")
    print(f"parity block dim D   : {D}")
    print(f"N mod 8              : {sym['mod8']}  ->  {sym['note']}")
    print(f"sectors diagonalized : {len(sym['parities'])} x "
          f"{'real symmetric' if sym['real_basis'] else 'complex Hermitian'}"
          f" {D}x{D}")
    if not memory_check(N_, jobs, args.max_memory_fraction):
        print("\nABORTED: this N does not fit safely in RAM. Use a smaller N,"
              " fewer --jobs, or raise --max-memory-fraction.")
        return 1

    if args.no_numba:
        use_numpy_kernels()
    limiter = limit_threads_here(threads if jobs == 1 else physical_cores())
    t = time.perf_counter()
    st = SYKStructure(N_)
    solver = Eigensolver(not args.no_two_stage)
    t_pre = time.perf_counter() - t
    print(f"eigensolver          : {solver.description()}")
    print(f"precomputation       : {t_pre:.2f} s "
          f"({st.n_terms} couplings in {len(st.gmask)} flip-mask groups)")

    if not args.no_validate:
        print("\nValidation:")
        V = run_validation(st, solver, J_, seed)
        if V.failed():
            print("\nVALIDATION FAILED -- results would not be trustworthy.")
            return 1

    # ---- streaming realizations ---------------------------------------
    dim = st.dim
    out_npy = f"syk_eigenvalues_N{N_}.npy"
    evs_file = np.lib.format.open_memmap(out_npy, mode="w+",
                                         dtype=np.float64,
                                         shape=(n_real * dim,))
    E0 = np.empty(n_real)
    r_real = np.empty(n_real)
    rc_real = np.empty(n_real)
    r_sum = r_n = rc_sum = rc_n = 0.0
    split = 0.0
    t_build = t_diag = 0.0
    spectrum0 = None

    print(f"\nDiagonalizing {n_real} realizations "
          f"({jobs} parallel job(s)) ...", flush=True)
    t_loop = time.perf_counter()

    def consume(res):
        nonlocal r_sum, r_n, rc_sum, rc_n, split, t_build, t_diag, spectrum0
        i = res["idx"]
        evs_file[i * dim:(i + 1) * dim] = res["spectrum"]
        E0[i] = res["E0"]
        r_real[i] = res["r_sum"] / res["r_n"]
        rc_real[i] = res["rc_sum"] / max(res["rc_n"], 1)
        r_sum += res["r_sum"]
        r_n += res["r_n"]
        rc_sum += res["rc_sum"]
        rc_n += res["rc_n"]
        split = max(split, res["kramers_split"])
        t_build += res["t_build"]
        t_diag += res["t_diag"]
        if i == 0:
            spectrum0 = res["spectrum"]
        progress[0] += 1
        done = progress[0]
        el = time.perf_counter() - t_loop
        if el - progress[1] > 1.0 or done == n_real:
            progress[1] = el
            print(f"  {done}/{n_real}  elapsed {el:.1f} s, "
                  f"remaining ~{el / done * (n_real - done):.0f} s   ",
                  end="\r", flush=True)

    progress = [0, -10.0]
    if jobs == 1:
        for i in range(n_real):
            consume(compute_realization(st, solver, J_, seed, i))
    else:
        set_thread_env(threads)
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
                max_workers=jobs, mp_context=ctx, initializer=_worker_init,
                initargs=(N_, J_, seed, not args.no_two_stage,
                          args.no_numba)) as ex:
            chunk = max(1, min(16, n_real // (4 * jobs)))
            for res in ex.map(_worker_run, range(n_real), chunksize=chunk):
                consume(res)
    evs_file.flush()
    wall = time.perf_counter() - t_loop
    print()

    # ---- report -----------------------------------------------------------
    print("\nPerformance:")
    print(f"  Hamiltonian construction : {t_build:.2f} s total, "
          f"{t_build / n_real * 1e3:.1f} ms per realization")
    print(f"  Diagonalization          : {t_diag:.2f} s total, "
          f"{t_diag / n_real * 1e3:.1f} ms per realization")
    print(f"  Wall time (realizations) : {wall:.2f} s "
          f"({wall / n_real * 1e3:.1f} ms per realization)")
    print(f"  Peak memory (main proc.) : {fmt_bytes(peak_rss())}")

    print(f"\nSpectrum of realization 0 ({dim} eigenvalues):")
    spec_file = print_spectrum(spectrum0, N_)

    print("\nGround state:")
    pm = lambda x, d: (f"+- {x:.{d}f}" if n_real > 1
                       else "(no error bar: 1 realization)")
    err = E0.std(ddof=1) / math.sqrt(n_real) if n_real > 1 else 0.0
    print(f"  E0/N = {E0.mean() / N_:.6f} {pm(err / N_, 6)}   "
          f"(mean +- std. error over {n_real} realizations)")

    print("\nSpectral statistics:")
    print(f"  N mod 8 = {sym['mod8']}")
    if sym["mirror"]:
        sect = "1 (even parity; odd parity is an exact copy)"
    else:
        sect = "2 (even and odd parity, independent)"
    print(f"  independent symmetry sectors = {sect}")
    if sym["kramers"]:
        scale = np.abs(E0).max()
        ok = split <= 1e-9 * scale
        print(f"  Kramers pairs removed (max pair splitting {split:.1e}"
              + (", i.e. exact degeneracy)" if ok else
                 ") -- WARNING: pairs are NOT degenerate, GSE statistics "
                 "invalid"))
    print(f"  RMT class = {sym['rmt']}")
    r_mean = r_sum / r_n
    r_err = r_real.std(ddof=1) / math.sqrt(n_real) if n_real > 1 else 0.0
    rc_err = rc_real.std(ddof=1) / math.sqrt(n_real) if n_real > 1 else 0.0
    print(f"  <r> = {r_mean:.4f} {pm(r_err, 4)}   (all levels)")
    print(f"  <r> = {rc_sum / rc_n:.4f} {pm(rc_err, 4)}   "
          f"(central half of each sector)")
    print("  reference values = " + ", ".join(
        f"{k} {v}" for k, v in R_REFERENCE.items()))
    naive = r_ratios(np.asarray(spectrum0)).mean() if dim > 3 else np.nan
    print(f"  (for comparison: naive <r> of the unsplit full spectrum of "
          f"realization 0 = {naive:.3f} -- meaningless, mixes sectors)")

    np.savez(f"syk_statistics_N{N_}.npz", N=N_, J=J_, seed=seed,
             E0=E0, r_per_realization=r_real,
             r_central_per_realization=rc_real, rmt_class=sym["rmt"])
    fig = plot_histogram(evs_file, N_, n_real,
                         show=args.show and not args.no_show)
    print("\nFiles:")
    print(f"  {out_npy}  (flat array of {n_real} x {dim}; "
          f".reshape({n_real}, {dim}))")
    print(f"  {fig}")
    print(f"  {spec_file}")
    print(f"  syk_statistics_N{N_}.npz  (E0 and <r> per realization)")
    print(f"\nTotal time: {time.perf_counter() - t_start:.1f} s")
    print(bar)
    del limiter
    return 0


# =============================================================================
# BENCHMARK AND AUTOMATIC MAXIMUM-N SEARCH
# =============================================================================

def _bench_one(N_, impl, seed, J_):
    """Runs inside a fresh subprocess; prints one JSON line."""
    out = dict(N=N_, impl=impl)
    if impl == "new":
        warm_up(Eigensolver(), J_, seed)
        t = time.perf_counter()
        st = SYKStructure(N_)
        solver = Eigensolver()
        out["precompute"] = time.perf_counter() - t
        res = compute_realization(st, solver, J_, seed, 0)
        out.update(build=res["t_build"], diag=res["t_diag"], E0=res["E0"])
    else:
        import majorana_SYK_reference as ref
        t = time.perf_counter()
        pairs = ref.pair_tables(N_)
        out["precompute"] = time.perf_counter() - t
        Jc = realization_couplings(N_, J_, seed, 0)
        build = diag = 0.0
        evs = []
        for p in ref.parity_blocks(N_):
            t = time.perf_counter()
            B = ref.build_hamiltonian_block(N_, Jc, pairs, p)
            build += time.perf_counter() - t
            t = time.perf_counter()
            evs.append(scipy.linalg.eigvalsh(B.toarray(), overwrite_a=True,
                                             check_finite=False))
            diag += time.perf_counter() - t
            del B
        out.update(build=build, diag=diag, E0=float(min(e.min()
                                                        for e in evs)))
    out["total"] = out["build"] + out["diag"]
    out["peak"] = peak_rss()
    print("BENCH_JSON " + json.dumps(out), flush=True)


def _bench_subprocess(N_, impl, timeout, seed, J_):
    cmd = [sys.executable, os.path.abspath(__file__), "--_bench-one", str(N_),
           "--_impl", impl, "--seed", str(seed), "--J", str(J_)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return dict(N=N_, impl=impl, error=f"timeout > {timeout:.0f} s")
    for line in p.stdout.splitlines():
        if line.startswith("BENCH_JSON "):
            return json.loads(line[len("BENCH_JSON "):])
    err = (p.stderr.strip().splitlines() or ["killed (out of memory?)"])[-1]
    return dict(N=N_, impl=impl, error=f"failed: {err[:80]}")


def _print_bench_row(r):
    sym = symmetry_info(r["N"])
    D = 2 ** (r["N"] // 2 - 1)
    kind = "real" if sym["real_basis"] and r["impl"] == "new" else "cplx"
    head = (f"{r['N']:>4} {2 ** (r['N'] // 2):>8} {D:>6} "
            f"{len(sym['parities'])}x{kind:<4} {sym['rmt']:<4} ")
    if "error" in r:
        print(head + f"  {r['error']}")
        return
    print(head + f"{r['build']:>9.3f} {r['diag']:>9.2f} {r['total']:>9.2f} "
                 f"{fmt_bytes(r['peak']):>10}")


def benchmark(Ns, old_max, max_frac, max_runtime, seed, J_):
    avail = available_memory()
    print(f"CPU: {cpu_name()}  ({physical_cores()} cores)")
    print(f"BLAS: {blas_description()}")
    print(f"RAM available: {fmt_bytes(avail)}\n")
    hdr = (f"{'N':>4} {'dim':>8} {'D':>6} {'sectors':<7} {'RMT':<4} "
           f"{'build[s]':>9} {'diag[s]':>9} {'total[s]':>9} {'peak mem':>10}")
    results = {"new": {}, "old": {}}
    for impl in ("new", "old"):
        print(("NEW implementation" if impl == "new" else
               "OLD implementation (majorana_SYK_reference.py)") +
              ", one realization per N (fresh process each):")
        print(hdr)
        for N_ in Ns:
            if impl == "old" and N_ > old_max:
                continue
            need = estimate_memory(N_)[0] if impl == "new" else \
                2 * 16 * 4 ** (N_ // 2 - 1) + BASE_PROCESS_BYTES
            if avail is not None and need > max_frac * avail:
                print(f"{N_:>4}  skipped: needs ~{fmt_bytes(need)}")
                continue
            r = _bench_subprocess(N_, impl, max_runtime * 3 + 120, seed, J_)
            results[impl][N_] = r
            _print_bench_row(r)
        print()
    print("Speed-up per realization (old total / new total):")
    for N_, r in results["old"].items():
        n = results["new"].get(N_)
        if n and "error" not in r and "error" not in n:
            mem = ""
            if r.get("peak") and n.get("peak"):
                mem = (f", peak memory {fmt_bytes(r['peak'])} -> "
                       f"{fmt_bytes(n['peak'])}")
            print(f"  N={N_}: old {r['total']:.3f} s, new {n['total']:.3f} s,"
                  f" speed-up {r['total'] / n['total']:.1f}x{mem}; "
                  f"same couplings: |E0_old - E0_new| = "
                  f"{abs(r['E0'] - n['E0']):.1e}")
    return results


def auto_max_n(max_frac, max_runtime, seed, J_, start=16):
    avail = available_memory()
    print(f"Automatic maximum-N search: runtime limit {max_runtime:.0f} s "
          f"per realization, memory limit {max_frac:.0%} of "
          f"{fmt_bytes(avail)}\n")
    print(f"{'N':>4} {'dim':>8} {'D':>6} {'sectors':<7} {'RMT':<4} "
          f"{'build[s]':>9} {'diag[s]':>9} {'total[s]':>9} {'peak mem':>10}")
    best, last = None, None
    N_ = start
    while True:
        need = estimate_memory(N_)[0]
        if avail is not None and need > max_frac * avail:
            print(f"{N_:>4}  STOP: needs ~{fmt_bytes(need)} > "
                  f"{fmt_bytes(max_frac * avail)} allowed")
            break
        if last is not None:
            pred = last["total"] * cost_units(N_) / cost_units(last["N"])
            if pred > max_runtime:
                print(f"{N_:>4}  STOP: predicted {pred:.0f} s per "
                      f"realization > limit {max_runtime:.0f} s "
                      f"(memory ~{fmt_bytes(need)} would fit)")
                break
        r = _bench_subprocess(N_, "new", max_runtime * 1.5 + 60, seed, J_)
        _print_bench_row(r)
        if "error" in r:
            break
        best, last = N_, r
        N_ += 2
    print(f"\nLargest successfully completed N: {best}")
    if last is not None:
        print("Extrapolation beyond it (cost ~ sectors x D^3, complex = 4x "
              "real):")
        for Nx in range(best + 2, best + 7, 2):
            need = estimate_memory(Nx)[0]
            t = last["total"] * cost_units(Nx) / cost_units(last["N"])
            safe = avail is None or need <= max_frac * avail
            print(f"  N={Nx}: ~{t / 60:.1f} min per realization, "
                  f"~{fmt_bytes(need)} RAM -> "
                  f"{'feasible but slow' if safe else 'UNSAFE (RAM)'}")
    return best


# =============================================================================
# COMMAND LINE
# =============================================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Full exact diagonalization of the q=4 Majorana SYK "
                    "model (laptop-optimized).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--N", type=int, default=N)
    p.add_argument("--J", type=float, default=J)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--realizations", type=int, default=N_REALIZATIONS)
    p.add_argument("--jobs", type=int, default=N_JOBS,
                   help="parallel realizations (default: automatic)")
    p.add_argument("--blas-threads", type=int, default=BLAS_THREADS,
                   help="threads per job (default: cores / jobs)")
    p.add_argument("--max-memory-fraction", type=float,
                   default=MAX_MEMORY_FRACTION)
    p.add_argument("--max-runtime", type=float,
                   default=MAX_RUNTIME_PER_REALIZATION,
                   help="seconds per realization for --auto-max-N")
    p.add_argument("--benchmark", nargs="*", type=int, metavar="N",
                   help=f"benchmark mode (default N list {BENCHMARK_N})")
    p.add_argument("--old-max-N", type=int, default=OLD_BENCHMARK_MAX_N,
                   help="largest N to benchmark the reference code")
    p.add_argument("--auto-max-N", action="store_true",
                   default=AUTO_BENCHMARK)
    p.add_argument("--no-validate", action="store_true")
    p.add_argument("--no-numba", action="store_true",
                   help="use the pure NumPy kernels")
    p.add_argument("--no-two-stage", action="store_true",
                   help="always use SciPy's eigensolver")
    p.add_argument("--show", action="store_true", default=SHOW_PLOT,
                   help=argparse.SUPPRESS)
    p.add_argument("--no-show", action="store_true",
                   help="do not open the histogram window")
    p.add_argument("--_bench-one", type=int, help=argparse.SUPPRESS)
    p.add_argument("--_impl", default="new", help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if a.N % 2 or a.N < 8:
        p.error("N must be even and >= 8")
    if a.realizations < 1:
        p.error("--realizations must be >= 1")
    return a


def main(argv=None):
    args = parse_args(argv)
    if args._bench_one is not None:
        _bench_one(args._bench_one, args._impl, args.seed, args.J)
        return 0
    if args.benchmark is not None:
        benchmark(args.benchmark or BENCHMARK_N, args.old_max_N,
                  args.max_memory_fraction, args.max_runtime, args.seed,
                  args.J)
        return 0
    if args.auto_max_N:
        auto_max_n(args.max_memory_fraction, args.max_runtime, args.seed,
                   args.J)
        return 0
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
