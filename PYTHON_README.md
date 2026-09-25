# Python SYK spectrum calculator (`syk_spectrum.py`)

`syk_spectrum.py` is a single Python file that computes the eigenvalues and the
density of states of the Majorana SYK model (q = 4). It runs on an ordinary
laptop. It uses the same conventions as the C++ code in this repository:
`{chi_a, chi_b} = delta_ab` and `<J_ijkl^2> = 3! J^2 / N^3`.

## 1. Install (one time)

1. Install Python 3 from https://www.python.org/downloads/. On Windows, tick
   "Add Python to PATH".
2. Open a terminal (Windows: "Command Prompt"; macOS: "Terminal").
3. Install the three libraries it needs:

       pip install numpy scipy matplotlib

## 2. Run

In the terminal, go to the folder that holds `syk_spectrum.py`
(for example `cd Downloads/C_SYK`), then run:

| Goal | Command |
|---|---|
| All eigenvalues, N=24, averaged over 20 disorder samples, with a plot | `python syk_spectrum.py --N 24 --samples 20 --plot` |
| All eigenvalues, N=28 | `python syk_spectrum.py --N 28 --samples 3 --plot` |
| The 10 lowest eigenvalues, N=32 | `python syk_spectrum.py --N 32 --mode lowest --k 10` |
| Density of states, N=32 (no full diagonalization) | `python syk_spectrum.py --N 32 --mode kpm --plot` |
| Show all options | `python syk_spectrum.py --help` |

Results go to the folder `syk_results/`:
- `evs_N..._s0.txt`: the eigenvalues of each disorder sample
- `dos_N....txt`: the density of states as two columns, E and rho(E),
  normalized to 1
- `dos_N....png`: a plot of the density of states (when you pass `--plot`)

The terminal also shows the ground-state energy per Majorana, E0/N, and the
level-spacing ratio <r>. You can use <r> to check the random-matrix class:
GOE 0.53 (N mod 8 = 0), GUE 0.60 (N mod 8 = 2, 6), GSE 0.68 (N mod 8 = 4).

## 3. The three modes

| Mode | What it gives | Method | Practical N on a laptop |
|---|---|---|---|
| `full` | every eigenvalue | dense diagonalization of each parity block | up to 30 |
| `lowest` | the k lowest eigenvalues | Lanczos (`scipy.sparse.linalg.eigsh`) | up to about 34 |
| `kpm` | a smooth density of states | Kernel Polynomial Method (Chebyshev moments) | up to about 34 |

## 4. Why it is fast

- **Parity blocks.** H conserves fermion parity, so the program works in two
  blocks, each half the size of the full space. Dense diagonalization costs
  dim^3, so this alone makes it 4x faster and uses 4x less memory.
- **Particle-hole symmetry.** When N mod 8 = 2 or 6, the two blocks have the
  same spectrum, so the program computes only one.
- **Bit operations.** Each Majorana product sends a basis state to a single
  other basis state. The program builds H directly from bit masks instead of
  multiplying large matrices.
- **KPM.** For the density of states you do not need the eigenvalues. KPM
  needs only a few hundred matrix-vector products.

## 5. What is not possible on a laptop

N = 48 is out of reach for exact diagonalization on a laptop, in Python or
any other language. One parity block has dimension 2^23, about 8 million, and
H has about 10^11 nonzero entries. Even the GPU-cluster code in this
repository computes only the lowest eigenvalues at that size, never the full
spectrum.

For the density of states at large N, the usual approach combines exact
results up to N of about 32 with the analytic large-N form: the Q-Hermite
density of Garcia-Garcia and Verbaarschot, and Cotler et al.
