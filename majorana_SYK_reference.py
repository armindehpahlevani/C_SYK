"""Majorana SYK model: spectrum of the q=4 SYK Hamiltonian -- fast laptop version.

Produces the same outputs as majorana_SYK_1.py:
  1. Pauli-matrix tests
  2. fast Hamiltonian vs. dense reference check (N=8)
  3. Clifford-algebra test {chi_a, chi_b} = delta_ab
  4. Hermiticity test of H
  5. eigenvalues of one realization
  6. eigenvalues of many disorder realizations + density-of-states histogram
     (saved as syk_histogram_N{N}.png)
and additionally prints the ground-state energy E0/N and the level-spacing
ratio <r> (random-matrix class check), and saves all eigenvalues to
syk_eigenvalues_N{N}.npy.

Why it is faster than majorana_SYK_1.py
---------------------------------------
* H conserves fermion parity, so it is block diagonal.  We build and
  diagonalize the two blocks of size 2^(N/2-1) separately: diagonalization
  costs dim^3, so this is ~4x faster and needs 4x less memory.
* For N mod 8 = 2 or 6 the two blocks have identical spectra (particle-hole
  symmetry), so only one block is diagonalized: another 2x.
* H is built as a sparse matrix directly from bit masks; no dense matrix
  products and no 2^(N/2) x 2^(N/2) intermediate arrays.

Practical sizes on a laptop (one realization): N=24 ~3 s, N=26 ~12 s,
N=28 ~2 min, N=30 ~10 min (needs ~5 GB RAM).

Conventions: chi_{2K} = Z..Z X I..I / sqrt2,  chi_{2K+1} = Z..Z Y I..I / sqrt2,
{chi_a, chi_b} = delta_ab,  H = sum_{i<j<k<l} J_ijkl chi_i chi_j chi_k chi_l,
<J_ijkl^2> = 3! J^2 / N^3.

Usage:  edit N and n_realizations at the bottom of the file, then
        python majorana_SYK_fast.py
Needs:  pip install numpy scipy matplotlib
"""

import itertools
import time

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import matplotlib.pyplot as plt


# ------------------------------------------------------- dense references
# Only for validating the fast path at small N.  Never call these with a
# large N: each matrix is (2**(N//2))**2 complex entries.


def pauli_matrices():  # define the Pauli matrices
    return {
        'I': np.array([[1, 0], [0, 1]], dtype=complex),
        'X': np.array([[0, 1], [1, 0]], dtype=complex),
        'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
        'Z': np.array([[1, 0], [0, -1]], dtype=complex),
    }


def kron_list(matrices_list):  # tensor product of matrices
    out = matrices_list[0]
    for matrix in matrices_list[1:]:
        out = np.kron(out, matrix)
    return out


def test_pauli_matrices():
    p = pauli_matrices()
    I, X, Y, Z = p['I'], p['X'], p['Y'], p['Z']

    for name, M in p.items():
        assert np.allclose(M, M.conj().T), f"{name} is not Hermitian"

    assert np.allclose(X @ X, I)
    assert np.allclose(Y @ Y, I)
    assert np.allclose(Z @ Z, I)
    assert np.allclose(X @ Y + Y @ X, 0)
    assert np.allclose(X @ Z + Z @ X, 0)
    assert np.allclose(Y @ Z + Z @ Y, 0)
    assert np.allclose(X @ Y, 1j * Z)

    print("pauli_matrices tests passed")


def build_majoranas(N):  # dense reference construction
    M = N // 2
    p = pauli_matrices()
    majoranas = []
    for K in range(M):
        for P in ('X', 'Y'):
            majoranas.append((1 / np.sqrt(2)) * kron_list(
                [p['Z'] if i < K else p[P] if i == K else p['I']
                 for i in range(M)]))
    return majoranas


def build_hamiltonian_dense_reference(N, couplings):
    """Original O(dim**3) construction -- small N only, for validation."""
    majoranas = build_majoranas(N)
    dim = 2 ** (N // 2)
    H = np.zeros((dim, dim), dtype=complex)
    quads = itertools.combinations(range(N), 4)
    for Jv, (i, j, k, l) in zip(couplings, quads):
        H += Jv * (majoranas[i] @ majoranas[j] @ majoranas[k] @ majoranas[l])
    return H


# ------------------------------------------------- fast bit-mask Majoranas
# Basis state |s>: bit K of the integer s is the occupation of qubit K.
# Each Majorana maps a basis state to exactly one basis state:
#     chi_a |s> = factor_a * sign_a(s) * |s XOR mask_a>


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
    """(mask, factor, sign array) of chi_a acting on the given states."""
    K = a // 2
    mask = 1 << K
    string = 1 - 2 * (popcount(states & (mask - 1)) & 1)   # Z on qubits < K
    if a % 2 == 0:                                         # X on qubit K
        return mask, 1 / np.sqrt(2), string
    occupied = (states >> K) & 1                           # Y on qubit K
    return mask, 1j / np.sqrt(2), string * (1 - 2 * occupied)


def pair_tables(N):
    """chi_a chi_b |s> = factor * sign(s) |s XOR mask> for every pair a<b."""
    s = np.arange(2 ** (N // 2), dtype=np.int64)
    single = [majorana_action(a, s) for a in range(N)]
    pairs = {}
    for a, b in itertools.combinations(range(N), 2):
        ma, fa, sa = single[a]
        mb, fb, sb = single[b]
        sign = sb * sa[s ^ mb]                     # chi_b acts first
        pairs[(a, b)] = (ma ^ mb, fa * fb, sign.astype(np.int8))
    return pairs


def random_couplings(N, J, rng):
    sigma = np.sqrt(6 * J**2 / N**3)
    n_terms = len(list(itertools.combinations(range(N), 4)))
    return rng.normal(scale=sigma, size=n_terms)


def build_hamiltonian_block(N, couplings, pairs, parity):
    """Sparse H restricted to states with fermion parity 0 (even) or 1 (odd)."""
    dim = 2 ** (N // 2)
    states = np.arange(dim, dtype=np.int64)
    block = states[(popcount(states) & 1) == parity]
    position = np.full(dim, -1, dtype=np.int64)
    position[block] = np.arange(len(block))

    coef = {}                          # terms with the same mask are summed
    quads = itertools.combinations(range(N), 4)
    for Jv, (i, j, k, l) in zip(couplings, quads):
        m_kl, f_kl, s_kl = pairs[(k, l)]
        m_ij, f_ij, s_ij = pairs[(i, j)]
        mask = m_ij ^ m_kl
        val = (Jv * f_ij * f_kl) * (s_kl[block] * s_ij[block ^ m_kl])
        if mask in coef:
            coef[mask] += val
        else:
            coef[mask] = val.astype(np.complex128)

    D = len(block)
    cols = np.arange(D)
    rows = np.concatenate([position[block ^ m] for m in coef])
    data = np.concatenate(list(coef.values()))
    return sp.csr_matrix((data, (rows, np.tile(cols, len(coef)))),
                         shape=(D, D))


def parity_blocks(N):
    """For N mod 8 = 2, 6 the odd block is a copy of the even block."""
    return [0] if N % 8 in (2, 6) else [0, 1]


def build_hamiltonian(N, couplings, pairs=None):
    """Full H (both parity blocks) as a list of sparse blocks."""
    pairs = pair_tables(N) if pairs is None else pairs
    return [build_hamiltonian_block(N, couplings, pairs, p) for p in (0, 1)]


# -------------------------------------------------------------- diagnostics


def test_majoranas(N):
    """Clifford algebra {chi_a, chi_b} = delta_ab, checked on every state."""
    dim = 2 ** (N // 2)
    s = np.arange(dim, dtype=np.int64)
    ops = [majorana_action(a, s) for a in range(N)]
    assert len(ops) == N, f"Expected {N} majoranas, got {len(ops)}"

    for a in range(N):
        ma, fa, sa = ops[a]
        # Hermitian: <s^m|chi|s> = conj(<s|chi|s^m>)
        assert np.allclose(fa * sa, np.conj(fa * sa[s ^ ma])), \
            f"chi_{a} is not Hermitian"
        for b in range(a, N):
            mb, fb, sb = ops[b]
            ab = fa * fb * sb * sa[s ^ mb]         # chi_a chi_b |s>
            ba = fb * fa * sa * sb[s ^ ma]         # chi_b chi_a |s>
            expected = 1.0 if a == b else 0.0
            assert np.allclose(ab + ba, expected), \
                f"anticommutator(chi_{a}, chi_{b}) != {expected}"

    print(f"All Clifford algebra tests passed for N={N}, dimension={dim}")


def test_fast_matches_dense(N=8, J=1.0, seed=0):
    """Fast builder must give the same spectrum as the dense one."""
    couplings = random_couplings(N, J, np.random.default_rng(seed))
    blocks = build_hamiltonian(N, couplings)
    ev_fast = np.sort(np.concatenate([np.linalg.eigvalsh(B.toarray())
                                      for B in blocks]))
    H_ref = build_hamiltonian_dense_reference(N, couplings)
    ev_ref = np.linalg.eigvalsh(H_ref)
    assert np.allclose(ev_fast, ev_ref), \
        "fast Hamiltonian disagrees with the dense reference"
    print(f"fast/dense Hamiltonian agreement verified for N={N}")


def test_hamiltonian(N, J=1.0, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    blocks = build_hamiltonian(N, random_couplings(N, J, rng))
    dim = 2 ** (N // 2)
    for B in blocks:
        assert abs(B - B.conj().T).max() < 1e-12, "Hamiltonian is not Hermitian!"
    assert sum(B.shape[0] for B in blocks) == dim
    print(f"Hamiltonian tests passed for N={N}, dimension={dim}")
    return blocks


# ---------------------------------------------------------- diagonalization


def diagonalize(blocks):
    """All eigenvalues of H given as parity blocks, sorted."""
    evs = [scipy.linalg.eigvalsh(B.toarray(), overwrite_a=True,
                                 check_finite=False) for B in blocks]
    return np.sort(np.concatenate(evs))


def mean_r_ratio(evs):
    """<r> within one symmetry block.  GOE 0.536, GUE 0.603, GSE 0.676."""
    evs = np.sort(evs)
    evs = evs[np.concatenate(([True], np.diff(evs) > 1e-10 * np.ptp(evs)))]
    s = np.diff(evs)
    r = np.minimum(s[1:], s[:-1]) / np.maximum(s[1:], s[:-1])
    return float(np.mean(r))


def run_multiple_realizations(N, n_realizations, J=1.0, seed=None,
                              verbose=True):
    """Diagonalize n_realizations SYK Hamiltonians; return all eigenvalues."""
    rng = np.random.default_rng(seed)
    dim = 2 ** (N // 2)
    pairs = pair_tables(N)                     # reused by every realization
    parities = parity_blocks(N)

    if verbose:
        print(f"diagonalizing {n_realizations} realizations of dim {dim} "
              f"(parity blocks of {dim // 2}"
              + (", odd block = even block by symmetry" if len(parities) == 1
                 else "") + ")")

    all_eigenvalues = np.empty(n_realizations * dim)
    ground = np.empty(n_realizations)
    r_values = []
    t0 = time.time()
    for n in range(n_realizations):
        couplings = random_couplings(N, J, rng)
        evs = []
        for p in parities:
            B = build_hamiltonian_block(N, couplings, pairs, p)
            e = scipy.linalg.eigvalsh(B.toarray(), overwrite_a=True,
                                      check_finite=False)
            r_values.append(mean_r_ratio(e))
            evs.append(e)
        if len(parities) == 1:
            evs = evs * 2
        evs = np.sort(np.concatenate(evs))
        all_eigenvalues[n * dim:(n + 1) * dim] = evs
        ground[n] = evs[0]
        if verbose:
            print(f"  {n + 1}/{n_realizations}  ({time.time() - t0:.1f} s)",
                  end="\r", flush=True)

    if verbose:
        print()
        print(f"ground state energy E0/N = {ground.mean() / N:.5f} "
              f"+- {ground.std() / N / np.sqrt(n_realizations):.5f}")
        print(f"level-spacing ratio <r> = {np.mean(r_values):.3f}  "
              f"(GOE 0.536, GUE 0.603, GSE 0.676)")
    return all_eigenvalues


def plot_energy_histogram(all_eigenvalues, N, n_realizations, bins=60):
    plt.figure(figsize=(7, 5))
    plt.hist(all_eigenvalues, bins=bins, density=True, edgecolor='black',
             alpha=0.7)
    plt.xlabel("Energy E")
    plt.ylabel("Density of states")
    plt.title(f"SYK energy spectrum, N={N}, {n_realizations} realizations")
    plt.savefig(f"syk_histogram_N{N}.png", dpi=150)
    print(f"Histogram saved as syk_histogram_N{N}.png")
    plt.show()


if __name__ == "__main__":

    N = 22                 # number of Majoranas (even); try 24, 26, 28
    n_realizations = 200   # disorder realizations for the histogram

    test_pauli_matrices()
    test_fast_matches_dense(N=8)
    test_majoranas(N)

    H_test = test_hamiltonian(N)
    eigs_test = diagonalize(H_test)
    print(f"Eigenvalues (N={N}):")
    print(eigs_test)
    del H_test

    eigs_N0 = run_multiple_realizations(N, n_realizations=n_realizations)
    np.save(f"syk_eigenvalues_N{N}.npy", eigs_N0)
    print(f"all eigenvalues saved as syk_eigenvalues_N{N}.npy")

    plot_energy_histogram(eigs_N0, N, n_realizations=n_realizations)
