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
        self.lk_params = dict(winSize = (self.window_size,self.window_size),
        maxLevel = self.max_level)
    
    def configure(self,window_size,min_eigen_threshold,max_level,termcrit_max_iters,termcirt_epsilon):
        self.window_size = window_size
        self.min_eigen_threshold = min_eigen_threshold
        self.max_level = max_level
        

    def track_features(self,img_1, img_2, points1, points2, id1, id2):
        
        # points2,status,_ = cv2.calcOpticalFlowPyrLK(np.mat(img_1), np.mat(img_2), np.mat(points1[0]), np.mat(points2[0]),
        #             np.array([self.window_size,self.window_size]),self.max_level, 
        #             cv2.TermCriteria_COUNT+cv2.TermCriteria_EPS, self.termcrit_max_iters, self.termcirt_epsilon,
        #             cv2.OPTFLOW_USE_INITIAL_FLOW,self.min_eigen_threshold)
        points1 = np.array(points1,dtype = np.float32)
        # n = points1.shape[0]
        # points1 = points1.reshape(n,1,2)
        points2 = np.array(points2,dtype = np.float32)
        points2,status,err = cv2.calcOpticalFlowPyrLK(img_1,img_2,points1,None,**self.lk_params)

        h = img_1.shape[0]
        w = img_1.shape[1]
        index = []
        dist_vector = points2 - points1
        dist = np.sum(np.array(dist_vector)**2,axis=1)
        for i in range(status.shape[0]):
            if dist[i]>25.0 or status[i] == 0 or points2[i,0]<0 or points2[i,1]<0 or points2[i,0]>w or points2[i,1]>h:
                if points2[i,0]<0 or points2[i,1]<0 or points2[i,0]>w or points2[i,1]>h:
                    status[i] = 0
                index.append(i)
        if index:
            points1 = np.array(np.delete(np.array(points1), index,axis=0)).tolist()
            points2 = np.array(np.delete(np.array(points2), index,axis=0)).tolist()
            id1 = np.delete(id1, index).tolist()
            id2 = np.delete(id2, index).tolist()
        else:
            points1 = np.array(points1).tolist()
            points2 = np.array(points2).tolist()
            
        return points1,points2,id1,id2

class CornerDetector():
    def __init__(self):
        self.n_rows=8
        self.n_cols=8
        self.grid_height = 0
        self.grid_width = 0
        self.detection_threshold=-1
        self.grid_n_rows = 0
        self.grid_n_cols = 0

    def sub2ind(self,sub):
        return sub[1]/self.grid_width + sub[0]/self.grid_height*self.grid_n_cols

    def get_n_rows(self):
        return self.grid_n_rows
    
    def get_n_cols(self):
        return self.grid_n_cols

    def set_grid_size(self, n_rows, n_cols):
        self.grid_n_rows = n_rows
        self.grid_n_cols = n_cols
        self.occupancy_grid = np.zeros((self.grid_n_rows*self.grid_n_cols+self.grid_n_rows,1))

    def set_grid_position(self,f):
        self.occupancy_grid[int(self.sub2ind(f))] = 1

    def shiTomasiScore(self,img,u,v):
        halfbox_size = 15
        box_size = 2*halfbox_size
        box_area = box_size*box_size
        x_min = v-halfbox_size
        x_max = v+halfbox_size
        y_min = u-halfbox_size
        y_max = u+halfbox_size
        if x_min <= 1 or x_max >= img.shape[1]-1 or y_min <= 1 or y_max >= img.shape[0]-1:
            return 0.0
        # print(x_min,x_max,y_min,y_max)
        dx = img[int(y_min):int(y_max),int(x_min+1):int(x_max+1)] - img[int(y_min):int(y_max),int(x_min-1):int(x_max-1)]
        dy = img[int(y_min+1):int(y_max+1),int(x_min):int(x_max)] - img[int(y_min-1):int(y_max-1),int(x_min):int(x_max)]
        dXX = np.sum(dx*dx)/(2*box_area)
        dYY = np.sum(dy*dy)/(2*box_area)
        dXY = np.sum(dx*dy)/(2*box_area)
        return 0.5 * (dXX + dYY - math.sqrt( (dXX + dYY) * (dXX + dYY) - 4 * (dXX * dYY - dXY * dXY) ))

    def detect_features(self,image):
        self.grid_height = (image.shape[0]/self.grid_n_rows)+1
        self.grid_width = (image.shape[1]/self.grid_n_cols)+1
        fast = cv2.FastFeatureDetector_create()
        fast.setNonmaxSuppression(True)
        kp = fast.detect(image,None)
        score = 0
        feature = []
        scores = []
        for i in range(len(kp)):
            if kp[i].pt[0] < image.shape[0] and kp[i].pt[1] < image.shape[1]:
                pt = kp[i].pt
                # print(pt)
                k = self.sub2ind(pt)
                if not self.occupancy_grid[int(k)]:
                    score = self.shiTomasiScore(image,kp[i].pt[0],kp[i].pt[1])
                    scores.append(score)
                    if score > self.detection_threshold:
                        pt = kp[i].pt
                        feature.append(np.array([pt[0],pt[1]]))
        print(max(scores))
        self.occupancy_grid = np.zeros((self.grid_n_rows*self.grid_n_cols+self.grid_n_rows,1))
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

    
   