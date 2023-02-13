from queue import Queue
import rospy
from sensor_msgs.msg import *
import numpy as np
from cv_bridge import CvBridge
from msckf import Msckf
from queue import Queue 
from pyquaternion import Quaternion
from corner_detector import TrackHandle
from module_msckf import Current_imu, Imu_state
from scipy.spatial.transform import Rotation as R

def quaternion_rotate_vector(quat, v):
    """Rotates a vector by a quaternion
    Args:
        quaternion: The quaternion to rotate by
        v: The vector to rotate
    Returns:
        np.ndarray: The rotated vector
    """
    vq = Quaternion(0, v[0], v[1], v[2])
    return (quat * vq * quat.inverse).vector

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
        return Quaternion(w, *axis)

    axis = np.cross(v0, v1)
    s = np.sqrt((1 + c) * 2)
    return Quaternion(s * 0.5, *(axis / s)) 

def quaternion2euler(quaternion):
    r = R.from_quat(quaternion)
    euler = r.as_euler('xyz', degrees=True)
    return euler
 

class Vio():
    def __init__(self):
        # self.img_qeuue = Queue(80)
        self.imu_queue = Queue(250)
        self.imu_sub = rospy.Subscriber("/imu0", Imu, self.imu_Cb)
        self.image_sub = rospy.Subscriber("/cam0/image_raw", Image,self.img_Cb)
        self.prev_imu_time = 0
        
        self.current_imu = Current_imu()
        self.imu_calibrate = False
        self.done_stand_still_time = 0
        self.cur_img_time = 0
        self.msckf = Msckf()
        self.init_imu_state = Imu_state()
        
        self.state_k = 0
        
        # param
        self.stand_still_time = 1
        self.T_cam_imu = np.array([[ 0.0148655429818, -0.999880929698,   0.00414029679422, -0.021640145497],
        [0.999557249008,   0.0149672133247,  0.025715529948,   -0.064676986768],
        [-0.0257744366974,  0.00375618835797, 0.999660727178,    0.009810730590],
        [0.0,  0.0,0.0,1.0]])
        self.R_cam_imu = self.T_cam_imu[:3,:3]
        self.p_cam_imu = self.T_cam_imu[:3,3]

        self.n_grid_rows = 8
        self.n_grid_cols = 8
        self.ransac_threshold = 0.000002

        self.intrinsics = np.array([457.587, 456.134, 379.999, 255.238])
        self.K = np.eye(3)
        self.K[0,0] = self.intrinsics[0]
        self.K[1,1] = self.intrinsics[1]
        self.K[0,2] = self.intrinsics[2]
        self.K[1,2] = self.intrinsics[3]

        self.dist_coeffs = np.array([-0.28368365,  0.07451284, -0.00010473, -3.55590700e-05])
        
        self.distortion_model = "radtan"
        self.trackhandle = TrackHandle(self.K, self.dist_coeffs, self.distortion_model)
        self.setup_track_handler()
        

    def setup_track_handler(self):
        self.trackhandle.detector.set_grid_size(self.n_grid_rows,self.n_grid_cols)
        self.trackhandle.set_ransac_threshold(self.ransac_threshold)

    def imu_Cb(self,msg):
        cur_imu_time = msg.header.stamp.to_sec()
        # first get imu time
        if(self.prev_imu_time == 0.0):
            self.prev_imu_time = cur_imu_time
            self.done_stand_still_time = cur_imu_time + self.stand_still_time 
            # print("first get imu time",self.prev_imu_time) 
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
            # print("imu_time:",self.current_imu.current_time)
            self.prev_imu_time = cur_imu_time

    def img_Cb(self,msg):
        self.cur_img_time = msg.header.stamp.to_sec()
        # print("img_time:",self.cur_img_time)
        bridge = CvBridge()
        img = bridge.imgmsg_to_cv2(msg, "mono8")
        
        if not self.imu_calibrate:
            if self.imu_queue.qsize() % 100 == 0 and self.imu_queue.qsize() > 0:
                print("Has", self.imu_queue.qsize(), "readings")
            if self.can_initialize_imu():
                self.initialize_imu()
                self.imu_calibrate = True
                print("finish initialize!")
                self.imu_queue.queue.clear()
                print("after clear imu_queue:",self.imu_queue.qsize())
                self.setup_msckf()
        else:
            imu_since_prev_img = self.find_frame_end()
        # imu_queue_.erase(imu_queue_.begin(), frame_end)
            print("imu_since_prev_img:",imu_since_prev_img.qsize())
        
            n = imu_since_prev_img.qsize()
            for i in range(n):
                imu = imu_since_prev_img.get()
                self.msckf.propagate(imu)
                gyro_measurement = np.dot(self.R_cam_imu.transpose(), (imu.omega-self.init_imu_state.b_g))
                self.trackhandle.add_gyro_reading(gyro_measurement)

            self.trackhandle.set_current_image(img, self.cur_img_time)
            cur_features, cur_ids = self.trackhandle.tracked_features()
            new_features, new_ids = self.trackhandle.get_new_features()
            # self.msckf.augmentState(self.state_k,self.cur_img_time)
            # self.msckf.update(cur_features, cur_ids)
            # self.msckf.addFeatures(new_features,new_ids)
            # self.msckf.marginalize()
            # self.msckf.pruneEmptyStates()

    def setup_msckf(self):
        self.state_k = 0
        self.msckf.initialize(self.init_imu_state)

    def find_frame_end(self):
        imu_since_prev_img = Queue()
        imu = Current_imu()
        n = self.imu_queue.qsize()
        print("imu_queue:",n)
        
        for i in range(n):
            imu = self.imu_queue.get()
            # print("cur_img_time:",self.cur_img_time)
            if imu.current_time < self.cur_img_time:
                # print("imu_time:",imu.current_time)
                # print(i,":add a new one!")
                imu_since_prev_img.put(imu)
            else:
                imu_since_prev_img.put(imu)
                return imu_since_prev_img
        return imu_since_prev_img

    def can_initialize_imu(self):
        return self.prev_imu_time > self.done_stand_still_time

    def initialize_imu(self):  
        accel_accum = np.zeros((3,))
        gyro_accum = np.zeros((3,))
        num = self.imu_queue.qsize()
        print("imu include initialize:",num)
        for i in range(num):
            accel_accum = accel_accum + self.imu_queue.get().a
            gyro_accum = gyro_accum + self.imu_queue.get().omega
        print("start initialize!")
        accel_mean = accel_accum / num
        gyro_mean = gyro_accum / num
        self.init_imu_state.b_g = gyro_mean
        self.init_imu_state.g = np.array([0,0,-9.81])
        self.init_imu_state.q_IG = quaternion_from_two_vectors(-self.init_imu_state.g,accel_mean)
        self.init_imu_state.b_a = quaternion_rotate_vector(self.init_imu_state.q_IG,self.init_imu_state.g)+accel_mean
        print("Initial IMU State:","p_I_G:",self.init_imu_state.p_I_G)
        print("q_IG: ",quaternion2euler(self.init_imu_state.q_IG.elements))
        print("v_I_G ",self.init_imu_state.v_I_G)
        print("b_a ",self.init_imu_state.b_a)
        print("b_g ", self.init_imu_state.b_g)
        print("g ",self.init_imu_state.g)

    def load_parameters(self):
        pass

def main():
    print("start")
    rospy.init_node('msckf_mono_node', anonymous=True)
    rate = rospy.Rate(300)
    vio = Vio()
    rospy.spin()
    
if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass