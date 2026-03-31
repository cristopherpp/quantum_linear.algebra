import numpy as np
from quantum_gates import zero, one, H, kron

# First quantum algorithm with exponential quantum speed-up
# Just using linear algebra

# XOR-power of Hadamard
def H_n(n):
    Hn = H
    for _ in range(n - 1):
        Hn = np.kron(Hn, H)
    return Hn

# deutsch_jozsa oracle:
# Uf |x,y> = |x, y XOR f(x)>
def dj_oracle(f, n):
    N = 2**n
    dim = 2**(n + 1)
    U = np.zeros((dim, dim))

    for x in range(N):
        for y in [0, 1]:
            input_state = 2*x + y
            output_state = 2*x + (y ^ f(x))
            U[output_state, input_state] = 1

    return U

# Deutsch-Jozsa Algorithm
def deutsch_jozsa(f, n):
    # |0 ... 0>|1>
    psi = zero
    for _ in range(n - 1):
        psi = kron(psi, zero)
    psi = kron(psi, one)

    # Apply hadamard to all qubits
    H_all = kron(H_n(n), H)
    psi = H_all @ psi

    # Apply oracle
    Uf = dj_oracle(f, n)
    psi = Uf @ psi

    # Apply Hadamard to input register
    psi = kron(H_n(n), np.eye(2)) @ psi

    # Measure input register
    probs = []
    for i in range(2**n):
        probs.append(np.sum(np.abs(psi[2 * i : 2 * i + 2]) * 2))

    # If only |0 ... 0> has probability -> constant
    #
    #
    if probs[0] > 0.999:
        return "Constant"
    else:
        return "Balanced"

## TESTINGGGGG!!!!!!!
def f0(x): return 0
def f1(x): return 1

def parity(x): return bin(x).count("1") % 2

print(deutsch_jozsa(f0, 3)) # Constant
print(deutsch_jozsa(f1, 3)) # Constant
print(deutsch_jozsa(parity, 3)) # Balanced
