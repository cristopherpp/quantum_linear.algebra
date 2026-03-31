import numpy as np

# Basis
zero = np.array([1, 0])
one = np.array([0,1])

# Hadamard gate (H)
# is a foundational 1-qubit quantum gate that creates equal superposition, transforming basis states |0⟩ or |1⟩ into:
#  (|0⟩ + |1⟩) / square root of (2) or (|0⟩ - |1⟩) / square root of (2)
#  Basically essential for quantum algorithms acting as a 2 * 2 unitary Hermitian matrix
H = (1/np.sqrt(2)) * np.array([[1,1], [1,-1]])

# Tensor product
def kron(a, b):
    return np.kron(a, b)

# David Deutsch 🐐🐐🐐 oracle
def oracle(f):
    U = np.zeros((4, 4))
    for x in [0, 1]:
        for y in [0, 1]:
            input_state = 2*x + y
            output_state = 2* x + (y ^ f(x))
            U[output_state, input_state] = 1
    return U

# since I have to copy the ""⟩"" I'll just use "">"" 

# Running Deutsch 🐐🐐🐐🐐🐐🐐🐐🐐
#
def deutsch(f): # 🐐🐐
    # Initial state |0>|1>
    psi = kron(zero, one)

    # Apply Hadamard to both qubits
    H2 = kron(H, H)
    psi = H2 @ psi

    # Apply oracle
    Uf = oracle(f)
    psi = Uf @ psi

    # Apply Hadamard to first qubit only
    H1 = kron(H, np.eye(2))
    psi = H1 @ psi

    # Measuring first qubit
    prob0 = abs(psi[0])**2 + abs(psi[1])**2
    prob1 = abs(psi[2])**2 + abs(psi[3])**2

    if prob0 > prob1:
        return "Constant"
    else:
        return "Balanced"


#  Testing constant
print(deutsch(lambda x: 0))
print(deutsch(lambda x: 1))


print(deutsch(lambda x: x))
print(deutsch(lambda x: 1-x))

# What just happened?:w
# output expected:
# Constant
# Constant
# Balanced
# Balanced

# Deutsch algorithm exist to prove a point it's just conceptual
# Deutsch algorithm proves: "A quantum computer can extract globla information about a function with fewer queries than any classical computer"
# It computes: "Are f(0) and f(1) the same?"
