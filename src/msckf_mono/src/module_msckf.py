from pyquaternion import Quaternion
import numpy as np
import cv2
import math
from queue import Queue

class Current_imu():
    def __init__(self):
        self.a = np.zeros((3,))
        self.omega = np.zeros((3,))
        self.dt = 0
        self.current_time = 0

class Imu_state():
    def __init__(self):
        self.b_g = 0
        self.g = 0
        self.b_a = 0
        self.q_IG = Quaternion(1, 0, 0, 0)
        self.q_IG_null = Quaternion(1, 0, 0, 0)
        self.p_I_G = np.zeros((3,))
        self.p_I_G_null = np.zeros((3,))
        self.v_I_G = np.zeros((3,))
        self.v_I_G_null = np.zeros((3,))

class NoiseParams ():
    def __init__(self):
        # 
        self.u_var_prime = 0
        self.v_var_prime = 0
        self.Q_imu = np.zeros((12,12))
        self.initial_imu_covar = np.zeros((15,15))

class Camera():
    def __init__(self):
        self.c_u = 0
        self.c_v = 0
        self.f_u = 0
        self.f_v = 0
        self.b = 0
        self.q_CI = Quaternion(1,0,0,0)
        self.p_C_I = np.zeros(3,)
    
class MSCKFParams():
    def __init__(self):
        self.max_gn_cost_norm = 0
        self.min_rcond = 0
        self.translation_threshold = 0
        self.redundancy_angle_thresh = 0
        self.redundancy_distance_thresh = 0
        self.min_track_length = 0
        self.max_track_length = 0
        self.max_cam_states = 0

class camState():
    def __init__(self):
        self.p_C_G = np.zeros((3,))
        self.q_CG = Quaternion(1,0,0,0)
        self.time = 0
        self.state_id = 0
        self.last_correlated_id = 0
        self.tracked_feature_ids = []


class CornerTracker():
    def __init__(self):
        self.window_size=51
        self.min_eigen_threshold=0.001
        self.max_level=5
        self.termcrit_max_iters=50
        self.termcirt_epsilon=0.01
    
    def configure(self,window_size,min_eigen_threshold,max_level,termcrit_max_iters,termcirt_epsilon):
        self.window_size = window_size
        self.min_eigen_threshold = min_eigen_threshold
        self.max_level = max_level
        

    def track_features(self,img_1, img_2, points1, points2, id1, id2):
        
        points2,status,err = cv2.calcOpticalFlowPyrLK(img_1, img_2, np.array(points1), 
                    (self.window_size,self.window_size),self.max_level, 
                    (cv2.TermCriteria_COUNT+cv2.TermCriteria_EPS, self.termcrit_max_iters, self.termcirt_epsilon),
                    cv2.OPTFLOW_USE_INITIAL_FLOW,self.min_eigen_threshold)
        h = img_1.shape[0]
        w = img_1.shape[1]
        index = []
        dist_vector = points2 - points1
        dist = np.sum(np.array(dist_vector)**2,axis=1)
        for i in range(status.shape()):
            if dist[i]>25.0 or status[i] == 0 or points2[i,0]<0 or points2[i,1]<0 or points2[i,0]>w or points2[i,1]>h:
                if points2[i,0]<0 or points2[i,1]<0 or points2[i,0]>w or points2[i,1]>h:
                    status[i] = 0
                index.append(i)
        points1 = np.delete(points1, index).tolist()
        points2 = np.delete(points2, index).tolist()
        id1 = np.delete(id1, index)
        id2 = np.delete(id2, index)

class CornerDetector():
    def __init__(self):
        self.n_rows=8
        self.n_cols=8
        self.grid_height = 0
        self.grid_width = 0
        self.detection_threshold=40.0
        self.grid_n_rows = 0
        self.grid_n_cols = 0

    def sub2ind(self,sub):
        return np.array(sub[0]/self.grid_width,sub[1]/self.grid_height*self.grid_n_cols)

    def get_n_rows(self):
        return self.grid_n_rows
    
    def get_n_cols(self):
        return self.grid_n_cols

    def set_grid_size(self, n_rows, n_cols):
        self.grid_n_rows = n_rows
        self.grid_n_cols = n_cols
        self.occupancy_grid = np.zeros((self.grid_n_rows,self.grid_n_cols))

    def set_grid_position(self,f):
        self.occupancy_grid[f[0],f[1]] = 1

    def shiTomasiScore(self,img,u,v):
        halfbox_size = 15
        box_size = 2*halfbox_size
        box_area = box_size*box_size
        x_min = u-halfbox_size
        x_max = u+halfbox_size
        y_min = v-halfbox_size
        y_max = v+halfbox_size
        if x_min < 1 or x_max >= img.cols-1 or y_min < 1 or y_max >= img.rows-1:
            return 0.0
        dx = img[x_min:x_max-1,y_min:y_max] - img[x_min+1:x_max,y_min:]
        dy = img[x_min:x_max,y_min:y_max-1] - img[x_min:x_max,y_min+1:y_max]
        dXX = np.sum(**dx)/(2*box_area)
        dYY = np.sum(**dy)/(2*box_area)
        dXY = np.sum(dx * dy)/(2*box_area)
        return 0.5 * (dXX + dYY - math.sqrt( (dXX + dYY) * (dXX + dYY) - 4 * (dXX * dYY - dXY * dXY) ))

    def detect_features(self,image):
        self.grid_height = (image.shape[0]/self.grid_n_rows)+1
        self.grid_width = (image.shape[1]/self.grid_n_cols)+1
        fast = cv2.FastFeatureDetector_create()
        kp = fast.detect(image,None)
        index = []
        score = 0
        feature = []
        for i in range(len(kp)):
            if kp[i].pt[0] < self.grid_n_cols*self.grid_width and kp[i].pt[1] < self.grid_n_rows*self.grid_height:
                k = self.sub2ind(kp[i].pt)
                if self.occupancy_grid[int(k[0]),int(k[1])]:
                    score = self.shiTomasiScore(image,kp[i].pt[0],kp[i].pt[1])
                    if score > self.detection_threshold:
                        feature.append(np.array([kp[i].pt[0]],kp[i].pt[1]))
        self.occupancy_grid = np.zeros((self.grid_n_rows,self.grid_n_cols))
        return feature  

class FeatureTrackToResidualize():
    def __init__(self):
        self.feature_id = 0
        self.observations = []
        self.cam_states = []
        self.cam_state_indices = []
        self.initialized = False
        self.p_f_G = np.zeros((3,))

class FeatureTrack():
    def __init__(self):
        self.feature_id = 0
        self.observations = []
        self.cam_state_indices = []
        self.initialized = False
        self.p_f_G = np.zeros((3,))

    
   