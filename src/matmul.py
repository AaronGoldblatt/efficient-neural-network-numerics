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


def svd(A, B, rank_A=None, rank_B=None):
    """
    Applies low-rank approximation (SVD) to both matrix A and B with rank rank_A
    and rank_B respectively.
    :param A: (N, M) pytorch tensor
    :param B: (M, K) pytorch tensor
    :param rank_A: None or int. None means use original A matrix.
    :param rank_B: None or int. None means use original B matrix.
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
    
    return A_approx @ B_approx


def logmatmul(A, B, **kwargs):
    """ TODO: use log multiplication for matrix-matrix multiplication """
    raise NotImplementedError