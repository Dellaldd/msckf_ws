from queue import Queue
import rospy
from sensor_msgs.msg import *
import numpy as np
from cv_bridge import CvBridge
from msckf import Msckf
from queue import Queue 

import math

def quaternion_rotate_vector(quat, v):
    """Rotates a vector by a quaternion
    Args:
        quaternion: The quaternion to rotate by
        v: The vector to rotate
    Returns:
        np.ndarray: The rotated vector
    """
    vq = quaternion.quaternion(0, 0, 0, 0)
    vq.imag = v
    return (quat * vq * quat.inverse()).imag

def quaternion_from_two_vectors(v0, v1) :
    r"""Computes the quaternion representation of v1 using v0 as the origin.
    """
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)
    c = v0.dot(v1)
    # Epsilon prevents issues at poles.
    if c < (-1 + 1e-8):
        c = max(c, -1)
        m = np.stack([v0, v1], 0)
        _, _, vh = np.linalg.svd(m, full_matrices=True)
        axis = vh.T[:, 2]
        w2 = (1 + c) * 0.5
        w = np.sqrt(w2)
        axis = axis * np.sqrt(1 - w2)
        return np.quaternion(w, *axis)

    axis = np.cross(v0, v1)
    s = np.sqrt((1 + c) * 2)
    return np.quaternion(s * 0.5, *(axis / s)) 

class Current_imu():
    def __init__(self):
        self.a = np.zeros((3,))
        self.omega = np.zeros((3,))
        self.dt = 0
        self.current_time = 0

class Init_imu_state():
    def __init__(self):
        self.b_g = 0
        self.g = 0
        self.b_a = 0
        self.q_IG = np.quaternion(1, 0, 0, 0)
        self.q_IG_null = np.quaternion(1, 0, 0, 0)
        self.p_I_G = np.zeros((3,))
        self.v_I_G = np.zeros((3,))
        

class Vio():
    def __init__(self):
        self.img_queue = Queue(10)
        self.imu_queue = Queue(200)
        self.imu_sub = rospy.Subscriber("/imu0", Imu, self.imu_Cb)
        self.image_sub = rospy.Subscriber("/cam0/image_raw", Image,self.img_Cb)
        img = Image()
        self.prev_imu_time = 0
        self.current_imu = Current_imu()
        self.imu_calibrate = False
        self.done_stand_still_time = 0
        self.cur_img_time = 0
        self.msckf = Msckf()
        self.init_imu_state = Init_imu_state()

        # param
        self.stand_still_time = 8
        self.T_cam_imu = np.array([ 0.0148655429818, -0.999880929698,   0.00414029679422, -0.021640145497],
        [ 0.999557249008,   0.0149672133247,  0.025715529948,   -0.064676986768],
        [-0.0257744366974,  0.00375618835797, 0.999660727178,    0.009810730590],
        [             0.0,  0.0,              0.0,               1.000000000000])
        self.R_cam_imu = self.T_cam_imu[:3,:3]
        self.p_cam_imu = self.T_cam_imu[:3,3]

    def imu_Cb(self,msg):
        cur_imu_time = msg.header.stamp.to_sec()
        # first get imu time
        if(self.prev_imu_time == 0.0):
            self.prev_imu_time = cur_imu_time
            self.done_stand_still_time = cur_imu_time + self.stand_still_time  
        else:
            self.current_imu.a[0] = msg.linear_acceleration.x
            self.current_imu.a[1] = msg.linear_acceleration.y
            self.current_imu.a[2] = msg.linear_acceleration.z

            self.current_imu.omega[0] = msg.angular_velocity.x
            self.current_imu.omega[1] = msg.angular_velocity.y
            self.current_imu.omega[2] = msg.angular_velocity.z

            self.current_imu.dt = cur_imu_time - self.prev_imu_time
            self.current_imu.current_time = cur_imu_time
            
            self.imu_queue.put(self.current_imu)

            self.prev_imu_time_ = cur_imu_time

    def img_Cb(self,msg):
        self.cur_img_time = msg.header.stamp.to_sec()
        bridge = CvBridge()
        img = bridge.imgmsg_to_cv2(msg, "mono8")

        if not self.imu_calibrate:
            if self.imu_queue.qsize() % 100 == 0:
                rospy.INFO("Has", self.img_queue.qsize(), "readings")
            if self.can_initialize_imu():
                self.initialize_imu()
                self.imu_calibrate = True
                
        else:
            pass
            imu_since_prev_img = self.find_frame_end()
            for imu in imu_since_prev_img:
                self.msckf.propagate()
                gyro_measurement = 

    def find_frame_end(self):
        imu_since_prev_img = Queue(10)
        imu = Current_imu()
        while (self.imu_queue.not_empty()):
            imu = self.imu_queue.get()
            if imu.current_time > self.cur_img_time:
                return imu_since_prev_img
            else:
                imu_since_prev_img.put(imu)
                

    def can_initialize_imu(self):
        return self.prev_imu_time > self.done_stand_still_time

    def initialize_imu(self):
        num = 0
        accel_accum = np.zeros((3,))
        gyro_accum = np.zeros((3,))
        while(self.imu_queue.not_empty()):
            accel_accum = accel_accum + self.img_queue.get().a
            gyro_accum = gyro_accum + self.img_queue.get().omega
        accel_mean = accel_accum / num
        gyro_mean = gyro_accum / num
        self.init_imu_state.b_g = gyro_mean
        self.init_imu_state.g = np.array([0,0,-9.81])
        self.init_imu_state.q_IG = quaternion_from_two_vectors(-self.init_imu_state.g,accel_mean)
        self.init_imu_state.b_a = quaternion_rotate_vector(self.init_imu_state.q_IG,self.init_imu_state.g)+accel_mean
        print("\nInitial IMU State:","\n--p_I_G:",self.init_imu_state.p_I_G,
        "\n--q_IG: ",quaternion.as_euler_angles(self.init_imu_state.q_IG),
        "\n--v_I_G ",self.init_imu_state.v_I_G,
        "\n--b_a ",self.init_imu_state.b_a,
        "\n--b_g ", self.init_imu_state.b_g,
        "\n--g ",self.init_imu_state.g)

    def load_parameters(self):
        pass

def main():
    rospy.init_node('msckf_mono_node', anonymous=True)
    rate = rospy.Rate(100)
    vio = Vio()
    rospy.spin()
    
if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass