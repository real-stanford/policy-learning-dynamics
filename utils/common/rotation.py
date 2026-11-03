from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def matrix_to_rotation_6d(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float64)
    assert array.shape[-2:] == (3, 3), f"Expected (..., 3, 3), got {array.shape}."
    return array[..., :2, :].reshape(array.shape[:-2] + (6,))


def rotvec_to_rotation_6d(rotvec: np.ndarray) -> np.ndarray:
    rotvec_array = np.asarray(rotvec, dtype=np.float64)
    assert rotvec_array.shape[-1] == 3, (
        f"Expected last dimension 3, got {rotvec_array.shape}."
    )
    matrix = Rotation.from_rotvec(rotvec_array.reshape(-1, 3)).as_matrix()
    matrix = matrix.reshape(rotvec_array.shape[:-1] + (3, 3))
    return matrix_to_rotation_6d(matrix).astype(np.float32)


def rotation_6d_to_matrix(rotation_6d: np.ndarray) -> np.ndarray:
    reshaped = np.asarray(rotation_6d, dtype=np.float64)
    assert reshaped.shape[-1] == 6, f"Expected last dimension 6, got {reshaped.shape}."
    a1 = reshaped[..., 0:3]
    a2 = reshaped[..., 3:6]

    a1_norm = np.linalg.norm(a1, axis=-1, keepdims=True)
    a1_norm = np.maximum(a1_norm, 1e-8)
    b1 = a1 / a1_norm
    proj = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - proj * b1
    b2_norm = np.linalg.norm(b2, axis=-1, keepdims=True)
    b2_norm = np.maximum(b2_norm, 1e-8)
    b2 = b2 / b2_norm
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack((b1, b2, b3), axis=-2)


def rotation_6d_to_rotvec(rotation_6d: np.ndarray) -> np.ndarray:
    matrix = rotation_6d_to_matrix(rotation_6d)
    flat = matrix.reshape(-1, 3, 3)
    try:
        rotvec = Rotation.from_matrix(flat).as_rotvec()
    except (ValueError, np.linalg.LinAlgError):
        # Fallback: fix degenerate matrices individually
        rotvec = np.zeros((flat.shape[0], 3), dtype=np.float64)
        for i in range(flat.shape[0]):
            try:
                rotvec[i] = Rotation.from_matrix(flat[i]).as_rotvec()
            except (ValueError, np.linalg.LinAlgError):
                U, _, Vt = np.linalg.svd(flat[i])
                R = U @ Vt
                if np.linalg.det(R) < 0:
                    U[:, -1] *= -1
                    R = U @ Vt
                try:
                    rotvec[i] = Rotation.from_matrix(R).as_rotvec()
                except (ValueError, np.linalg.LinAlgError):
                    pass  # zero rotvec (identity)
    return rotvec.reshape(matrix.shape[:-2] + (3,)).astype(np.float32)
