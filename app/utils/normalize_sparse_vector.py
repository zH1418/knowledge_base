from __future__ import annotations

import numpy as np


def normalize_sparse_vector(
    sparse_vec: dict[int, float] | None,
) -> dict[int, float]:
    if not sparse_vec:
        return {}

    values = np.array(list(sparse_vec.values()), dtype=np.float64)

    l2_norm = np.linalg.norm(values)
    if l2_norm < 1e-9:
        return sparse_vec

    normalized = (values / l2_norm).tolist()  # .tolist() → Python float

    return {dim: float(val) for (dim, _), val in zip(sparse_vec.items(), normalized)}
