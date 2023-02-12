import numpy as np
from module_msckf import CornerDetector,CornerTracker
import cv2


    
class TrackHandle():
    def __init__(self, K, dist_coeffs, distortion_model):
        self.gyro_accum = np.zeros((3,))
        self.n_gyro_readings = 0
        self.next_feature_id = 0

        self.prev_time = 0
        self.cur_time = 0

        self.prev_features = []
        self.cur_features = []
        self.new_features = []
        
        self.prev_feature_ids = []
        self.cur_feature_ids = []
        self.new_feature_ids = []

        self.prev_img = []   #
        self.cur_img = []

        self.detector = CornerDetector()
        self.track = CornerTracker()

        self.use_gyro = False
        self.ransac_threshold = 0

        self.dR = np.eye(3)
        self.K = K
        self.dist_coeffs = dist_coeffs
        self.distortion_model = distortion_model

    def set_ransac_threshold(self, rt):
        self.ransac_threshold = rt

    def add_gyro_reading(self,gyro_reading):
        self.gyro_accum = self.gyro_accum + gyro_reading
        self.n_gyro_readings += 1
        print("finish add one gyro reading")
    
    def set_current_image(self, img, time):
        self.prev_time = self.cur_time
        self.prev_img = self.cur_img
        self.prev_features = self.cur_features
        self.prev_feature_ids = self.cur_feature_ids
        self.prev_feature_ids.append(self.new_feature_ids) # add at the last
        self.prev_features.append(self.new_features)
        rows = self.detector.get_n_rows()
        cols = self.detector.get_n_cols()
        oc_grid = np.zeros((rows*cols,))
        n = np.min(len(self.prev_features),len(self.prev_feature_ids))
        index = []
        for i in range(n):
            ind = self.detector.sub2ind(self.prev_features[i])
            if(oc_grid[ind]):
                index.append(ind)
            else:
                oc_grid[ind] = 1
        self.prev_features = np.delete(self.prev_features, index).tolist()
        self.prev_feature_ids = np.delete(self.prev_feature_ids, index).tolist()
        self.cur_time = time
        self.cur_img = img
        self.cur_features.clear()
        self.cur_feature_ids.clear()

        self.new_features.clear()
        self.new_feature_ids.clear()
    
    def integrate_gyro(self):
        dt = self.cur_time - self.prev_time
        if self.n_gyro_readings > 0:
            self.gyro_accum /= self.n_gyro_readings
            self.gyro_accum *= dt
            self.dR = cv2.Rodrigues(np.mat([[self.gyro_accum[0]],[self.gyro_accum[1]],[self.gyro_accum[2]]],dtype=float))
            self.gyro_accum = np.zeros((3,))
            self.n_gyro_readings = 0
        else:
            self.use_gyro = False
            
    
    def predict_features(self):
        self.cur_feature_ids.append(self.prev_feature_ids)
        if self.use_gyro:
            self.integrate_gyro()
            H = np.dot(np.dot(self.K , self.dR), np.linalg.inv(self.K))
            prev_feature = np.array(self.predict_features)
            pt_buf1 = np.hstack(prev_feature,np.ones((prev_feature.shape[0],1)))
            pt_buf1 = np.dot(H,pt_buf1)
            pt_buf1 = pt_buf1 / pt_buf1[:,2]
            self.cur_features.append(pt_buf1[:,:2].tolist())
        else:
            self.cur_features.append(self.prev_features)

    def undistortPoints(self,feature):
        if self.distortion_model == "radtan":
            feature = np.array(feature,dtype = float)       
            # feature = np.ndarray[int, np.dtype[np.generic]]
            feature = cv2.undistortPoints(feature,self.K,self.dist_coeffs)
            feature = np.squeeze(feature).tolist()
            return feature

    def tracked_features(self):
        prev_size = len(self.prev_features)
        if prev_size != 0:
            self.predict_features()
            # visualizer_.add_predicted(cur_features_, cur_feature_ids_);
            self.prev_features, self.cur_features,self.prev_feature_ids, self.cur_feature_ids = self.track.track_features(self.prev_img, self.cur_img,
                            self.prev_features, self.cur_features,
                            self.prev_feature_ids, self.cur_feature_ids)
            prev_features = self.undistortPoints(self.prev_features)   
            cur_features  = self.undistortPoints(self.cur_features)    
            return cur_features, self.cur_feature_ids
    
    def get_new_features(self):
        for f in self.cur_features:
            self.detector.set_grid_position(f)
        self.new_features = self.detector.detect_features(self.cur_img)
        for i in range(len(self.new_features)):
            self.new_feature_ids.append(self.next_feature_id)
            self.next_feature_id += 1        
        return self.undistortPoints(self.new_features),self.new_feature_ids
            
        




