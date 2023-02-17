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

# def quaternion2euler(quaternion):
#     r = R.from_quat(quaternion)
#     euler = r.as_euler('xyz', degrees=True)
#     return euler

# a = Quaternion(1,0,0,0)
# print(quaternion2euler(a.elements))
# from scipy.linalg import expm
# a = np.array([1,0])
# print(np.diag(a))
# # print(np.exp(a))
import cv2
# extrinsic = np.array([[0.05812254, 0.9969995, 0.05112498, 0.043909],
#                     [-0.02821786, -0.04955038, 0.99837293, -0.026862],
#                     [0.99791058, -0.05947061, 0.02525319, -0.006717],
#                     [0., 0., 0., 1.]])

# rvec, _ = cv2.Rodrigues(extrinsic[:3, :3])
# print(rvec)

# a = np.mat([[1],[2],[3]],dtype=float)
# rot_mat,_ = cv2.Rodrigues(a)
# print(rot_mat)

# a= [[1,2],[2,3]]
# b = np.sum(np.array(a)**2,axis=1)

# print(np.argwhere(b > 1))

# intrinsics = np.array([457.587, 456.134, 379.999, 255.238])
# K = np.eye(3)
# K[0,0] = intrinsics[0]
# K[1,1] = intrinsics[1]
# K[0,2] = intrinsics[2]
# K[1,2] = intrinsics[3]

# dist_coeffs = [-0.28368365,  0.07451284, -0.00010473, -3.55590700e-05]
# dist_coeffs = np.float32(dist_coeffs)

# feature = [[1,2],[2,3]]
# feature = np.array(feature,dtype = float)
# print(type(feature))
# print(type(dist_coeffs))
# # feature = np.ndarray[int, np.dtype[np.generic]]
# feature = cv2.undistortPoints(feature,K,dist_coeffs)
# feature = np.squeeze(feature).tolist()
# print(type(feature))
# print(feature)

# from queue import Queue
# x = Queue(5)
# x.put(1)
# x.put(2)
# x.put(2)
# x.put(2)
# x.put(2)
# print(x.qsize())
# b = x.get()

# print(b)

# i = [1,2,3,4]
# a = np.where(np.array([1,2,3,4])==5)[0]
# i.pop(1)
# print(i)
# print(i.index(6))

# print(np.iinfo(np.int8).min)
# a = np.array([[1,2,3,4],[5,6,7,8]])
# b = np.repeat(a,[2,1],axis = 1)
# print(a[:,-2:])
# a = np.count_nonzero([[0,0,7,0,0],[3,0,0,2,19],[0,0,0,0,0]], axis=1)
# print(a)
# print(np.count_nonzero(a))
# a = Quaternion(1,1,0,0)
# print(a)
# print(type(a.normalised))

# from module_msckf import Camera
# a = Camera()
# b = Camera()
# c = Camera()
# x = [a,b,c]
# y = np.array(x)
# print(y)
# index = [0,1]
# z = np.delete(y,index).tolist()
# print(type(z))

# a = np.ones((2,3))
# b = np.ones((3,2))
# c = np.ones((2,2))
# print(np.mat(a)*np.mat(b)*np.mat(c))

# from module_msckf import FeatureTrack
# f1 = FeatureTrack()
# f2 = FeatureTrack()
# f3 = FeatureTrack()
# feature_tracks = [f1,f2,f3]
# f = feature_tracks[0]
# f.cam_state_indices.append(1)
# print(feature_tracks[0].cam_state_indices)

# from scipy.stats import chi2
# print(chi2.pdf(1))
# from numpy import random
# a = random.chisquare(df=1, size=100)
# print(np.percentile(a,5))

# from scipy.stats import chi2
# # boost::math::quantile(norm, 0.5) <0.5 
# a=chi2.ppf(q=0.05, df=3) 
# print(a)
# cam_states = [1,12,3,4,5,6]
# print(cam_states.count(0))

import rospy
from sensor_msgs.msg import *
from cv_bridge import CvBridge
# class getImage():
#     def __init__(self):
#         self.count = 0
#         self.image_sub = rospy.Subscriber("/cam0/image_raw", Image,self.img_Cb)
#         self.total = 0

#     def img_Cb(self,msg):
#         bridge = CvBridge()
#         img = bridge.imgmsg_to_cv2(msg, "mono8")
#         if self.total > 100:
            
#             if self.count%2 == 0 and self.count< 20:
#                 path = "image/" + str(self.count/2) + ".png"
#                 cv2.imwrite(path,img)
#                 print(self.count)
#             self.count += 1
#         self.total += 1
        
# def main():
#     print("start")
#     rospy.init_node('get_image_node', anonymous=True)
#     getimage = getImage()
#     rospy.spin()

# if __name__ == '__main__':
#     try:
#         main()
#     except rospy.ROSInterruptException:
#         pass

index = [1,1,2,3,4,5,6]
index = list(set(index))
print(index)


