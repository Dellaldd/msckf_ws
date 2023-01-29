import quaternion
import numpy as np
def quaternion_rotate_vector(
    quat: quaternion.quaternion, v: np.ndarray
) -> np.ndarray:
    r"""Rotates a vector by a quaternion
    Args:
        quaternion: The quaternion to rotate by
        v: The vector to rotate
    Returns:
        np.ndarray: The rotated vector
    """
    vq = quaternion.quaternion(0, 0, 0, 0)
    vq.imag = v
    return (quat * vq * quat.inverse()).imag

a = quaternion.quaternion(0, 1, 0, 0)
print(quaternion.as_euler_angles(a))
b = np.array([0,0,1])
# print(quaternion.as_rotation_matrix(a))
print(quaternion_rotate_vector(a,b))
