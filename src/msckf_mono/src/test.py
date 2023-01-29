import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
# import quaternion

# def quaternion_rotate_vector(quat, v):
#     """Rotates a vector by a quaternion
#     Args:
#         quaternion: The quaternion to rotate by
#         v: The vector to rotate
#     Returns:
#         np.ndarray: The rotated vector
#     """
#     vq = quaternion.quaternion(0, 1, 0, 0)
#     vq.imag = v
#     print(quat * vq * quat.inverse())
#     return (quat * vq * quat.inverse()).imag

# a = quaternion.quaternion(1, 0, 0, 0)
# b = np.array([0,2,2])
# # print(quaternion.as_rotation_matrix(a))
# print(quaternion_rotate_vector(a,b))


# def quaternion_rotate_vector(quat, v):
#     """Rotates a vector by a quaternion
#     Args:
#         quaternion: The quaternion to rotate by
#         v: The vector to rotate
#     Returns:
#         np.ndarray: The rotated vector
#     """
#     vq = Quaternion(0, v[0], v[1], v[2])
    
#     print(quat.inverse)
#     return (quat * vq * quat.inverse).vector

# a = Quaternion(0, 1, 0, 0)
# b = np.array([0,2,2])
# # print(quaternion.as_rotation_matrix(a))
# print(quaternion_rotate_vector(a,b))

# def quaternion_from_two_vectors(v0, v1) :
#     r"""Computes the quaternion representation of v1 using v0 as the origin.
#     """
#     v0 = v0 / np.linalg.norm(v0)
#     v1 = v1 / np.linalg.norm(v1)
#     c = v0.dot(v1)
#     # Epsilon prevents issues at poles.
#     if c < (-1 + 1e-8):
#         c = max(c, -1)
#         m = np.stack([v0, v1], 0)
#         _, _, vh = np.linalg.svd(m, full_matrices=True)
#         axis = vh.T[:, 2]
#         w2 = (1 + c) * 0.5
#         w = np.sqrt(w2)
#         axis = axis * np.sqrt(1 - w2)
#         return Quaternion(w, *axis)

#     axis = np.cross(v0, v1)
#     s = np.sqrt((1 + c) * 2)
#     return Quaternion(s * 0.5, *(axis / s)) 

# a = np.array([1,0,0])
# b = np.array([-1,0,0])
# print(quaternion_from_two_vectors(a,b))

def quaternion2euler(quaternion):
    r = R.from_quat(quaternion)
    euler = r.as_euler('xyz', degrees=True)
    return euler

a = Quaternion(1,0,0,0)
print(quaternion2euler(a.elements))
