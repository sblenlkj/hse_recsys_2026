"""Popularity, ItemKNN и EASE для бинарных взаимодействий."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits


def popularity_scores(matrix: sparse.csr_matrix) -> np.ndarray:
    return np.asarray(matrix.sum(axis=0)).ravel().astype(np.float32)


def fit_itemknn(matrix: sparse.csr_matrix, neighbors: int = 100, shrinkage: float = 10.0) -> sparse.csr_matrix:
    """Item-item cosine similarity; в каждой колонке остаются top-N соседей."""
    similarity = (matrix.T @ matrix).tocoo().astype(np.float32)
    norm = np.sqrt(np.maximum(np.asarray(matrix.sum(axis=0)).ravel(), 1e-12))
    similarity.data /= norm[similarity.row] * norm[similarity.col] + shrinkage
    similarity = similarity.tocsc()
    similarity.setdiag(0)
    similarity.eliminate_zeros()
    result = sparse.lil_matrix(similarity.shape, dtype=np.float32)
    for column in range(similarity.shape[1]):
        start, end = similarity.indptr[column:column + 2]
        values, indices = similarity.data[start:end], similarity.indices[start:end]
        keep = np.argsort(values, kind="stable")[-neighbors:]
        result[indices[keep], column] = values[keep]
    return result.tocsr()


def fit_ease(matrix: sparse.csr_matrix, regularization: float, blas_threads: int = 4) -> np.ndarray:
    """Обучает EASE: B = -P / diag(P), diag(B)=0."""
    gram = (matrix.T @ matrix).toarray().astype(np.float64)
    gram.flat[:: gram.shape[0] + 1] += regularization
    with threadpool_limits(limits=blas_threads):
        precision = np.linalg.inv(gram)
    weights = -precision / np.diag(precision)[None, :]
    np.fill_diagonal(weights, 0.0)
    return weights.astype(np.float32)


def score(matrix: sparse.csr_matrix, weights: sparse.spmatrix | np.ndarray) -> np.ndarray:
    product = matrix @ weights
    return product.toarray() if sparse.issparse(product) else np.asarray(product)
