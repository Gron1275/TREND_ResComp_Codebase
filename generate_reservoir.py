from numpy.random import default_rng
from scipy import sparse
from scipy.sparse import linalg


def rescale(A, spectral_radius):

    if not isinstance(A, sparse.csr.csr_matrix):
        A = sparse.csr_matrix(A)

    eigenvalues, _ = linalg.eigs(A)
    max_eigenvalue = max(abs(eigenvalues))
    A_scaled = A / max_eigenvalue * spectral_radius
    return A_scaled


def erdos_renyi_reservoir(size, degree, radius,seed=0):

    rng = default_rng(seed)
    nonzero_mat = rng.random((size, size)) < degree / size  # matrix of 1s
    random_weights_mat = -1 + 2 * rng.random((size, size))  # uniform distribution on [-1, 1] at each matrix element
    A = nonzero_mat * random_weights_mat
    A[A == -0.0] = 0.0
    sA = sparse.csr_matrix(A)

    try:
        sA_rescaled = rescale(sA, radius)
        return sA_rescaled
    except linalg.eigen.arpack.ArpackNoConvergence:
        # If cannot find eigenvalues, just start over
        return erdos_renyi_reservoir(size, degree, radius)
        # May cause infinite recursion but hopefully I avoid this! Sorry if it does


def generate_W_in(num_inputs, res_size, sigma,seed = 0):

    rng = default_rng(seed)
    W_in = sigma * (-1 + 2 * rng.random((res_size, num_inputs)))
    return W_in
