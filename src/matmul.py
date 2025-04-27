import torch
import torch.nn.functional as F


def matmul(A, B, method='naive', **kwargs):
    """
    Multiply two matrices.
    :param A: (N, M) torch tensor.
    :param B: (M, K) torch tensor.
    :param method:
    :return:
        Output matrix with shape (N, K)
    """
    method = method.lower()
    if method in ['naive', 'pytorch', 'torch']:
        return naive(A, B)
    elif method == 'svd':
        return svd(A, B, **kwargs)
    elif method in ['log', 'logmatmul']:
        return logmatmul(A, B, **kwargs)
    else:
        raise ValueError("Invalid [method] value: %s" % method)


def naive(A, B, **kwargs):
    return A @ B


def svd(A, B, rank_A=None, rank_B=None, return_approx_matrices=False):
    """
    Applies low-rank approximation (SVD) to both matrix A and B with rank rank_A
    and rank_B respectively.
    :param A: (N, M) pytorch tensor
    :param B: (M, K) pytorch tensor
    :param rank_A: None or int. None means use original A matrix.
    :param rank_B: None or int. None means use original B matrix.
    :param return_approx_matrices: bool. If True, return the approximated matrices without multiplying them.
    :return: a (N, K) pytorch tensor
    """
    if rank_A is None and rank_B is None:
        return A @ B
    
    def get_svd_approximation(matrix, rank):
        if rank is None:
            return matrix
            
        left_singular_vectors, singular_values, right_singular_vectors = torch.svd(matrix, some=True)
        rank = min(
            rank,
            min(matrix.shape)
        )
        return (
            left_singular_vectors[
                :,
                : rank
            ]
            @ (
                singular_values[: rank].reshape(-1, 1)
                * (
                    right_singular_vectors[
                        :,
                        : rank
                    ].T
                )
            )
        )
    
    A_approx = get_svd_approximation(A, rank_A)
    B_approx = get_svd_approximation(B, rank_B)
    
    if return_approx_matrices:
        return A_approx, B_approx
    
    return A_approx @ B_approx


def logmatmul(A, B, **kwargs):
    """Implements log-matrix multiplication, which is done by converting the matrices to log-space and then summing instead of multiplying."""
    rows_A, cols_A = A.shape
    rows_B, cols_B = B.shape
    
    assert cols_A == rows_B, f"Matrices' dimensions not compatible for multiplication, A cols: {cols_A} and B rows: {rows_B} are not equal"
    
    sign_A = torch.sign(A)
    sign_B = torch.sign(B)
    
    epsilon = 1e-30
    log_A = torch.log2(torch.abs(A) + epsilon)
    log_B = torch.log2(torch.abs(B) + epsilon)
    
    sign_A = sign_A.view(
        rows_A,
        cols_A,
        1
    )
    sign_B = sign_B.view(
        1,
        rows_B,
        cols_B
    )
    
    log_A = log_A.view(
        rows_A,
        cols_A,
        1
    )
    log_B = log_B.view(
        1,
        rows_B,
        cols_B
    )
    
    sum_of_logs = log_A + log_B
    product_of_signs = sign_A * sign_B
    
    element_products = product_of_signs * (2 ** sum_of_logs)
    
    result = torch.sum(element_products, dim=1)
    
    return result
    