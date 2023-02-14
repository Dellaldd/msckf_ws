from module_msckf import Imu_state,NoiseParams,Camera,MSCKFParams,camState,FeatureTrack,FeatureTrackToResidualize
import numpy as np
from pyquaternion import Quaternion
import math

def vectorToSkewSymmetric(v):
    M = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return M

def omegaMat(omega):
    # Compute the omega-matrix of a 3-d vector omega
    bigOmega = np.zeros((4,4))
    bigOmega[:3,:3] = -vectorToSkewSymmetric(omega)
    bigOmega[:3,3] = omega
    bigOmega[3,:3] = -omega.T
    return bigOmega

class Msckf():
    def __init__(self):
        self.imu_state = Imu_state()
        self.F = np.zeros((15,15))
        self.G = np.zeros((15,12))
        self.Phi = np.zeros((15,15))
        self.imu_cam_covar = np.zeros((15,6))# (15,dynamic)
        self.noise_params = NoiseParams()
        self.camera = Camera()
        self.msckf_params = MSCKFParams()
        self.map = np.zeros((3,))
        self.cam_states = []
        self.P = []
        self.cam_covar = []
        self.imu_covar = np.zeros((15,15))
        self.chi_squared_test_table = []

        self.feature_tracks_to_residualize = []
        self.tracks_to_remove = []
        self.tracked_feature_ids = []
        self.feature_tracks = []
        self.pruned_states = []
    
    def initialize(self,init_imu_state):
        q_var_init = 1e-5
        bg_var_init = 1e-2
        v_var_init = 1e-2
        ba_var_init = 1e-2
        p_var_init = 1e-12
        IMUCovar_vars = np.array([q_var_init, q_var_init, q_var_init,
                     bg_var_init,bg_var_init,bg_var_init,
                     v_var_init, v_var_init, v_var_init,
                     ba_var_init,ba_var_init,ba_var_init,
                     p_var_init, p_var_init, p_var_init])
        self.noise_params.initial_imu_covar = np.diag(IMUCovar_vars)
        
        w_var = 1e-5
        dbg_var = 3.6733e-5
        a_var = 1e-3
        dba_var = 7e-4
        Q_imu_vars = np.array([w_var, 	w_var, 	w_var,
                  dbg_var,dbg_var,dbg_var,
                  a_var,	a_var,	a_var,
                  dba_var,dba_var,dba_var])
        self.noise_params.Q_imu = np.diag(Q_imu_vars)

        feature_cov = 7
        intrinsics = np.array([457.587, 456.134, 379.999, 255.238])
        self.camera.f_u = intrinsics[0]
        self.camera.f_v = intrinsics[1]
        self.camera.c_u = intrinsics[2]
        self.camera.c_v = intrinsics[3]
        self.noise_params.u_var_prime = math.pow(feature_cov/self.camera.f_u,2)
        self.noise_params.v_var_prime = math.pow(feature_cov/self.camera.f_v,2)

        self.msckf_params.max_gn_cost_norm = 11
        self.msckf_params.max_gn_cost_norm = math.pow(self.msckf_params.max_gn_cost_norm/self.camera.f_u, 2)
        self.msckf_params.translation_threshold = 0.05
        self.msckf_params.min_rcond = 3e-12
        self.msckf_params.redundancy_angle_thresh = 0.005
        self.msckf_params.redundancy_distance_thresh = 0.05
        self.msckf_params.max_track_length =  1000
        self.msckf_params.min_track_length = 3
        self.msckf_params.max_cam_states = 20

        self.num_feature_tracks_residualized = 0
        self.imu_state = init_imu_state
        self.pos_init = self.imu_state.p_I_G
        self.imu_state.p_I_G_null = self.imu_state.p_I_G
        self.imu_state.v_I_G_null = self.imu_state.v_I_G
        self.imu_state.q_IG_null = self.imu_state.q_IG
        self.imu_covar = self.noise_params.initial_imu_covar
        self.last_feature_id = 0
        
        # // Initialize the chi squared test table with confidence
        # // level 0.95.
        # chi_squared_test_table.resize(99);
        # for (int i = 1; i < 100; ++i) {
        #   boost::math::chi_squared chi_squared_dist(i);
        #   chi_squared_test_table[i-1] = boost::math::quantile(chi_squared_dist, 0.05);
        # }
        

    def propagate(self,measurement):
        self.calcF(self.imu_state, measurement) 
        self.calcG(self.imu_state)
        imu_state_prop = self.propogateImuStateRK(self.imu_state, measurement)
        
        # F * dt
        self.F *= measurement.dt
        # Matrix exponential
        self.Phi = np.exp(self.F)
        
        # Apply observability constraints - enforce nullspace of Phi
        # Ref: Observability-constrained Vision-aided Inertial Navigation, Hesch J.
        # et al. Feb
        R_kk_1 = self.imu_state.q_IG_null.rotation_matrix
        self.Phi[:3,:3] = np.dot(imu_state_prop.q_IG.rotation_matrix, R_kk_1.T)
        u = np.dot(R_kk_1, self.imu_state.g)
        u = u.reshape(-1,1)
        u_u = np.dot(u,u.T)
        # print(u_u)
        # u_u = np.mat(u_u)
        u_u_inv = np.linalg.pinv(u_u)
        
        s = np.dot(u_u_inv,u)
        # s = np.mat(np.linalg.inv(np.dot(u,u.T))) * np.mat(u)
        # s = np.dot(np.linalg.inv(np.dot(u,u.T)), u)

        A1 = self.Phi[6:9, :3]
        tmp = self.imu_state.v_I_G_null - imu_state_prop.v_I_G
        w1 = np.dot(vectorToSkewSymmetric(tmp), self.imu_state.g)
        self.Phi[6:9, :3] = A1 - np.dot((np.dot(A1, u) - w1), s)

        A2 = self.Phi[12:15, :3]
        tmp = measurement.dt * self.imu_state.v_I_G_null + self.imu_state.p_I_G_null - imu_state_prop.p_I_G
        w2 = np.dot(vectorToSkewSymmetric(tmp), self.imu_state.g)
        self.Phi[12:15, :3] = A2 - np.dot((np.dot(A2, u) - w2), s)

        imu_covar_prop = np.mat(self.Phi) * (self.noise_params.initial_imu_covar + np.mat(self.G) * np.mat(self.noise_params.Q_imu) * np.mat(self.G.T) * measurement.dt) * np.mat(self.Phi.T)
        
        # Apply updates directly
        self.imu_state = imu_state_prop
        self.imu_state.q_IG_null = self.imu_state.q_IG
        self.imu_state.v_I_G_null = self.imu_state.v_I_G
        self.imu_state.p_I_G_null = self.imu_state.p_I_G

        self.noise_params.initial_imu_covar = (imu_covar_prop + imu_covar_prop.T) / 2.0
        self.imu_cam_covar = np.mat(self.Phi) * np.mat(self.imu_cam_covar) # problem
        print("finish one propagate")

    def propogateImuStateRK(self, imu_state_k, measurement_k):
        # Runge-Kutta
        imuStateProp = imu_state_k
        dt = measurement_k.dt

        omega_vec = measurement_k.omega - imu_state_k.b_g
        omega_psi = 0.5 * omegaMat(omega_vec)
        
        # Note: MSCKF Matlab code assumes quaternion form: -x,-y,-z,w
        # Eigen quaternion is of form: w,x,y,z
        # Following computation accounts for this change
        y0 = [-imu_state_k.q_IG.x, -imu_state_k.q_IG.y, -imu_state_k.q_IG.z, imu_state_k.q_IG.w]
        k0 = np.dot(omega_psi,y0)
        k1 = np.dot(omega_psi, (y0 + (k0 / 4.) * dt))
        k2 = np.dot(omega_psi, (y0 + (k0 / 8. + k1 / 8.) * dt))
        k3 = np.dot(omega_psi, (y0 + (-k1 / 2. + k2) * dt))
        k4 = np.dot(omega_psi, (y0 + (k0 * 3. / 16. + k3 * 9. / 16.) * dt))
        k5 = np.dot(omega_psi, (y0 +(-k0 * 3. / 7. + k1 * 2. / 7. + k2 * 12. / 7. - k3 * 12. / 7. + k4 * 8. / 7.) *
           dt))
        y_t = y0 + (7. * k0 + 32. * k2 + 12. * k3 + 32. * k4 + 7. * k5) * dt / 90.
        q = Quaternion(y_t[3], -y_t[0], -y_t[1], -y_t[2])
        q = q.normalised
        imuStateProp.q_IG = q
        delta_v_I_G = (np.dot(imu_state_k.q_IG.rotation_matrix.T,(measurement_k.a - imu_state_k.b_a)) + imu_state_k.g) * dt
        imuStateProp.v_I_G += delta_v_I_G
        imuStateProp.p_I_G = imu_state_k.p_I_G + imu_state_k.v_I_G * dt
        return imuStateProp

    def calcF(self, imu_state_k, measurement_k):
        # Multiplies the error state in the linearized continuous-time error state model 
        omegaHat = measurement_k.omega - imu_state_k.b_g
        aHat = measurement_k.a - imu_state_k.b_a
        C_IG = imu_state_k.q_IG.rotation_matrix
        self.F[:3,:3] = -vectorToSkewSymmetric(omegaHat)
        self.F[:3,3:6] = -np.identity(3)
        self.F[6:9,9:12] = -C_IG.T
        self.F[6:9,:3] = -np.dot(-C_IG.T,vectorToSkewSymmetric(aHat))
        self.F[12:15,6:9] = np.identity(3)

    def calcG(self, imu_state_k):
        C_IG = imu_state_k.q_IG.rotation_matrix
        self.G[:3,:3] = -np.identity(3)
        self.G[3:6,3:6] = np.identity(3)
        self.G[6:9,6:9] = -C_IG.T
        self.G[9:12,9:12] = np.identity(3)
    
    def augmentState(self,state_id, time):
        self.map = np.zeros((3,1))
        q_CG = self.camera.q_CI * self.imu_state.q_IG
        q_CG = q_CG.normalised
        camstate = camState()
        camstate.last_correlated_id = -1
        camstate.q_CG = q_CG

        camstate.p_C_G =self.imu_state.p_I_G + np.dot(self.imu_state.q_IG.inverse.rotation_matrix,self.camera.p_C_I) 

        camstate.time = time
        camstate.state_id = state_id

        n = len(self.cam_states)
        if n:
            self.P = np.zeros((15+self.cam_covar.shape[1],15+self.cam_covar.shape[1]))
            self.P[:15,:15] = self.imu_covar
            self.P[:15,15:15+self.cam_covar.shape[1]] = self.imu_cam_covar
            self.P[15:15+self.cam_covar.shape[1],:15] = self.imu_cam_covar.T
            self.P[15:15+self.cam_covar.shape[1],15:15+self.cam_covar.shape[1]] = self.cam_covar
        else:
            self.P = self.imu_covar
        J = np.zeros((6,15+6*n))
        J[:3,:3] = self.camera.q_CI.rotation_matrix
        J[3:6,:3] = vectorToSkewSymmetric(np.dot(self.imu_state.q_IG.inverse.rotation_matrix,self.camera.p_C_I))
        J[3:6,12:15] = np.identity(3)

        tempMat = np.identity(15 + 6*n)
        tempMat = np.vstack((tempMat,J))

        P_aug = np.dot(np.dot(tempMat, self.P), tempMat.transpose()) # Pk_k augment
        P_aug_sym = (P_aug + P_aug.transpose()) / 2.0
        P_aug = P_aug_sym

        self.cam_states.append(camstate)
        self.imu_covar = P_aug[:15,:15]

        self.cam_covar = P_aug[15:,15:]
        self.imu_cam_covar = P_aug[:15,15:]

        print("finish one augment!")
    def removeTrackedFeature(self,featureID):
        camStateIndices = []
        featCamStates = []
        for c_i in range(len(self.cam_states)):
            feature_iter = np.where(np.array(self.cam_states[c_i].tracked_feature_ids)==featureID)[0]
            if feature_iter is not None:
                self.cam_states[c_i].tracked_feature_ids.pop[feature_iter]
                camStateIndices.append(c_i)
                featCamStates.append(self.cam_states[c_i])
        return featCamStates, camStateIndices

    def update(self,measurements,feature_ids):
        self.feature_tracks_to_residualize = []
        self.tracks_to_remove = []
        id_iter = 0
        for feature_id in self.tracked_feature_ids:
            input_feature_ids_iter = feature_ids.index(feature_id)
            is_valid = input_feature_ids_iter != len(feature_ids)-1
            track = FeatureTrack()
            self.feature_tracks.append(track)
            if(is_valid):
                track.observations.append(measurements[input_feature_ids_iter])
                self.cam_states[-1].tracked_feature_ids.append(feature_id)
                track.cam_state_indices.append(self.cam_states.state_id)
            
            if (not is_valid) or len(track.observations) >= self.msckf_params.max_track_length:
                track_to_residualize = FeatureTrackToResidualize()
                track_to_residualize.cam_states,track_to_residualize.cam_state_indices=self.removeTrackedFeature(feature_id)

                if(len(track_to_residualize.cam_states)>= self.msckf_params.min_track_length):
                    track_to_residualize.feature_id = self.feature_tracks[id_iter].feature_id
                    track_to_residualize.observations = self.feature_tracks[id_iter].observations
                    track_to_residualize.initialized = self.feature_tracks[id_iter].initialized
                    if(self.feature_tracks[id_iter].initialized):
                        track_to_residualize.p_f_G = self.feature_tracks[id_iter].p_f_G
                    self.feature_tracks_to_residualize.append(track_to_residualize)
                
                self.tracks_to_remove.append(feature_id)

            id_iter += 1

    def addFeatures(self,features,feature_ids):
        for i in range(len(features)):
            id = feature_ids[i]
            if not np.where(np.array(self.tracked_feature_ids == id))[0]:
                track = FeatureTrack()
                track.feature_id = feature_ids[i]
                track.observations.append(features[i])
                self.cam_states[-1].tracked_feature_ids.append(feature_ids[i])
                track.cam_state_indices.append(self.cam_states[-1].state_id)

                self.feature_tracks.append(track)
                self.tracked_feature_ids.append(feature_ids[i])
            else:
                print("Error, added new feature that was already being tracked")

    def generateInitialGuess(self,T_c1_c2, z1,z2):
        m = np.dot(T_c1_c2[:3,:3],np.array([z1[0]],[z1[1]],1))
        A = np.zeros((2,))
        A[0] = m[0] - z2[0] * m[2]
        A[1] = m[1] - z2[1] * m[2]
        b = np.zeros((2,))
        b[0] = z2[0] * T_c1_c2[2,3] - T_c1_c2[0,3]
        b[1] = z2[1] * T_c1_c2[2,3] - T_c1_c2[1,3]

        depth = np.dot(np.dot(np.linalg.inv(np.dot(A.T,A)),A.T),b)
        p = np.zeros((3,))
        p[0] = z1[0] * depth
        p[1] = z1[1] * depth
        p[2] = depth
        return depth

    def cost(self, T_c0_ci,x, z):
        alpha = x[0]
        beta = x[1]
        rho = x[2]
        h = np.dot(T_c0_ci[:3,:3], np.array([alpha, beta, 1.0])) + rho * T_c0_ci[:3,3]
        z_hat = np.array([h[0]/h[2]],h[1]/h[2])
        e = np.linalg.norm(z_hat-z)  
        return e 


    def jacobian(self,T_c0_ci,x, z):
        alpha = x[0]
        beta = x[1]
        rho = x[2]

        h = np.dot(T_c0_ci[:3,:3], np.array([alpha, beta, 1.0])) + rho * T_c0_ci[:3,3]
        W= np.zeros((3,3))
        W[:,:2] = T_c0_ci[:3,:2]
        W[:,2] = T_c0_ci[:3,3]

        J = np.zeros((2,3))
        J[0,:] = 1/h[2]*W[0,:] - h[0]/(h[2]*h[2])*W[2,:]
        J[1,:] = 1/h[2]*W[1,:] - h[1]/(h[2]*h[2])*W[2,:]

        z_hat = np.array([h[0] / h[1], h[1] / h[2]])
        r = z_hat - z
        e = np.linalg.norm(r)
        huber_epsilon = 0.01
        if  e <= huber_epsilon:
            w = 1
        else:
            w = huber_epsilon/(2*e)
        return J,r,w

    def initializePosition(self,cam_states,measurements):
        cam_poses = []
        for cam in cam_states:
            cam0_pose = cam.q_CG.rotation_matrix.T
            cam0_pose = np.hstack(cam0_pose,cam.p_C_G)
            cam0_pose = np.vstack(cam0_pose,np.array([0,0,0,1]))
            cam_poses.append(cam0_pose)
        T_c0_w = cam_poses[0]
        for i in range(len(cam_poses)):
            cam_poses[i] = np.dot(np.linalg.inv(cam_poses[i]),T_c0_w)
        
        initial_position = self.generateInitialGuess(cam_poses[-1], measurements[0],
                             measurements[-1])
        solution = initial_position/initial_position[2]

        initial_damping = 1e-3
        lambda0 = initial_damping
        inner_loop_max_iteration = 10
        outer_loop_max_iteration = 10
        estimation_precision = 5e-7
        inner_loop_cntr = 0
        outer_loop_cntr = 0
        is_cost_reduced = False
        delta_norm = 0
        # Compute the initial cost.
        total_cost = 0.0

        for i in range(len(cam_poses)):
            this_cost = self.cost(cam_poses[i],solution,measurements[i])
            total_cost += this_cost
        
        while outer_loop_cntr<outer_loop_max_iteration and delta_norm > estimation_precision:
            A = np.zeros((3,3))
            b = np.zeros((3,))
            for i in range(len(cam_poses)):
                J,r,w = self.jacobian(cam_poses[i],solution,measurements[i])

                if w ==1:
                    A += np.dot(J.T, J)
                    b += np.dot(J.T, r)
                else:
                    w_square = w * w
                    A += np.dot(w_square * J.T, J)
                    b += np.dot(w_square * J.T, r)
            while (inner_loop_cntr < inner_loop_max_iteration and  not is_cost_reduced):
                damper = lambda0 * np.identity(3)
                delta = np.linalg.solve(A + damper, b)
                new_solution = solution - delta
                delta_norm = np.linalg.norm(delta)

                new_cost = 0.0
                for i in range(len(cam_poses)):
                    this_cost = 0.0
                    this_cost = self.cost(cam_poses[i], new_solution, measurements[i])
                    new_cost += this_cost

                if (new_cost < total_cost):
                    is_cost_reduced = True
                    solution = new_solution
                    total_cost = new_cost
                    if lambda0/10>1e-10:
                        lambda0 = lambda0 /10
                    else:
                        lambda0 = 1e-10
                else:
                    is_cost_reduced = False
                    if lambda0 *10 <1e12:
                        lambda0 = lambda0 *10
                    else:
                        lambda0 = 1e12
                inner_loop_cntr += 1   
            outer_loop_cntr += 1
        final_position = np.array([solution[0]/solution[2],solution[1]/solution[2],1/solution[2]])
        is_valid_solution = True
        for pose in cam_poses:
            position = np.dot(pose[:3,:3],final_position) + pose[:3,3]
            if position[2] <= 0:
                is_valid_solution = False
                break
        normalized_cost = total_cost / (2 * len(cam_poses) * len(cam_poses))
        cov_diag = np.diag(self.imu_covar)
        
        if (normalized_cost > self.msckf_params.max_gn_cost_norm):
            is_valid_solution = False
        p_f_G = T_c0_w[:3,:3] * final_position + T_c0_w[:3,3]
        return is_valid_solution,p_f_G

    def calcResidual(self,p_f_G,camstates,observations):
        iter = 0
        r_j = []
        for state_i in camstates:
            p_f_C = np.dot(state_i.q_CG.rotation_matrix, (p_f_G - state_i.p_C_G))
            zhat_i_j = p_f_C[:2] / p_f_C[2]
            r_j.append(observations[iter]-zhat_i_j)# list[i] = (x,y)
            iter += 1
        return r_j

    def calcMeasJacobian(self,p_f_G,camstateindices):
        H_f_j = np.zeros((2 * len(camstateindices), 3))
        H_x_j = np.zeros((2 * len(camstateindices), 15 + 6 * self.cam_states))
        for c_i in range(len(camstateindices)):
            index = camstateindices[c_i]
            p_f_C = self.cam_states[index].q_CG.rotation_matrix, (p_f_G - self.cam_states[index].p_C_G)

            X = p_f_C[0]
            Y = p_f_C[1]
            Z = p_f_C[2]

            J_i = np.array([[1, 0, -X / Z], [0, 1, -Y / Z]]) #2x3
            J_i *= 1 / Z
            
            A = np.hstack(np.dot(J_i, vectorToSkewSymmetric(p_f_C)),
            np.dot(-J_i, self.cam_states[index].q_CG.rotation_matrix))
            tmp = p_f_G - self.cam_states[index].p_C_G

            u = np.hstack(np.dot(self.cam_states[index].q_CG.rotation_matrix, self.imu_state.g),np.dot(vectorToSkewSymmetric(tmp), self.imu_state.g))

            H_x = A - np.dot(np.dot(A, u), np.dot(np.linalg.inv((np.dot(u.T * u)), u.T)))
            H_f = -H_x[:2,3:6]
            H_f_j[2*c_i:2*c_i+2,:3] = H_f

            H_x_j[2 * c_i:2*c_i+2, 15 + 6*index:15 + 6*index +6] = H_x

        jacobian_row_size = 2 * len(camstateindices)
        svd_helper,_,_ = np.linalg.svd(H_f_j)
        A_j = svd_helper[:,(-jacobian_row_size + 3):]
        H_o_j =np.dot(A_j.T, H_x_j) 
        return H_o_j, A_j


    def buildUpdateQuat(self,deltaTheta):
        deltaq = 0.5 * deltaTheta
        updateQuat = Quaternion(1,0,0,0)
        # // Replaced with squaredNorm() ***1x1 result so using sum instead of creating
        # // another variable and then referencing the 0th index value***
        checkSum = np.linalg.norm(deltaq)
        if (checkSum > 1) :
          updateQuat.w = 1
          updateQuat.x = -deltaq[0]
          updateQuat.y = -deltaq[1]
          updateQuat.z = -deltaq[2]
        else:
          updateQuat.w = math.sqrt(1 - checkSum)
          updateQuat.x = -deltaq[0]
          updateQuat.y = -deltaq[1]
          updateQuat.z = -deltaq[2]
        return updateQuat.normalised

    def measurementUpdate(self,H_o, r_o, R_o):
        if r_o.shape[0] != 0:
            P = np.zeros((15 + self.cam_covar.shape[0], 15 + self.cam_covar.shape[1]))
            P[:15, :15] = self.imu_covar
            if (self.cam_covar.shape[0] != 0):
                P[:15, 15:self.imu_cam_covar.shape[1]] = self.imu_cam_covar
                P[15:15+self.imu_cam_covar.shape[1], :15] = self.imu_cam_covar.T
                P[15:15+self.cam_covar.shape[0], 15:15+self.cam_covar.shape[1]] = self.cam_covar

            Q,R = np.linalg.qr(H_o)
            nonZeroRows = np.count_nonzero(R, axis=1)
            numNonZeroRows = np.count_nonzero(nonZeroRows)

            T_H = np.zeros((numNonZeroRows, R.cols()))
            Q_1 = np.zeros(Q.rows(), numNonZeroRows)

            counter = 0
            for r_ind in range(R.shape[0]):
                if (nonZeroRows(r_ind) == 1.0):
                    T_H[counter,:] = R[r_ind,:]
                    Q_1[:,counter] = Q[:,r_ind]
                    counter += 1
                    if (counter > numNonZeroRows):
                        print("More non zero rows than expected in QR decomp")
            
            r_n = np.dot(Q_1.T, r_o)
            R_n = np.dot(np.dot(Q_1.T , R_o), Q_1)

            # Calculate Kalman Gain
            temp = np.dot(np.dot(T_H, P), T_H.t()) + R_n
            K = np.dot(np.dot(P, T_H), np.linalg.inv(temp))

            # State Correction
            deltaX = np.dot(K, r_n)

            q_IG_up = np.dot(self.buildUpdateQuat(deltaX[:3]), self.imu_state.q_IG)

            self.imu_state.q_IG = q_IG_up

            self.imu_state.b_g += deltaX[3:6]
            self.imu_state.b_a += deltaX[9:12]
            self.imu_state.v_I_G += deltaX[6:9]
            self.imu_state.p_I_G += deltaX[12:15]

            # // Update Camera<_S> states
            for c_i in range(len(self.cam_states)):
                q_CG_up = np.dot(self.buildUpdateQuat(deltaX[15 + 6 * c_i: 15 + 6 * c_i+3]),self.cam_states[c_i].q_CG)
                self.cam_states[c_i].q_CG = q_CG_up.normalized
                self.cam_states[c_i].p_C_G += deltaX[18 + 6 * c_i:18 + 6 * c_i+3]

            # // Covariance correction
            tempMat = np.identity(15 + 6 * len(self.cam_states))-np.dot(K, T_H)

          
            P_corrected = np.dot(np.dot(tempMat, P), tempMat.T) + np.dot(np.dot(K, R_n), K.T)
            # // Enforce symmetry
            P_corrected_transpose = P_corrected.T
            P_corrected += P_corrected_transpose
            P_corrected /= 2

            if(P_corrected.shape[0]-15!=self.cam_covar.shape[0]):
                print(P_corrected.shape) 
                print(self.cam_covar.shape)

        #   TODO : Verify need for eig check on P_corrected here (doesn't seem tooimportant for now)
            self.imu_covar = P_corrected[:15,:15]

        #   TODO: Check here
            self.cam_covar = P_corrected[15:, 15:]
            self.imu_cam_covar = P_corrected[:15,15:]
            

    def marginalize(self):
        if len(self.feature_tracks_to_residualize):
            num_passed = 0
            num_rejected = 0
            num_ransac = 0
            max_length = -1
            min_length = 99
            max_norm = -1
            min_norm = 99
            total_nObs = 0
            total_nObs = 0
            valid_tracks = []
            p_f_G_vec = []
            for track in self.feature_tracks_to_residualize:
                if (self.num_feature_tracks_residualized > 3) and not (self.checkMotion(track.observations, track.cam_states)):
                    num_rejected +=1
                    valid_tracks.append(0)
                    continue
                isvalid,p_f_G = self.initializePosition(track.cam_states, track.observations)
                if isvalid:
                    track.initialized = True
                    track.p_f_G = p_f_G
                    self.map.push_back(p_f_G)
            p_f_G_vec.append(p_f_G)
            nObs = len(track.observations)

            p_f_C1 = np.dot(track.cam_states[0].q_CG.rotation_matrix,(p_f_G - track.cam_states[0].p_C_G))

            if not isvalid:
                num_rejected += 1
                valid_tracks.append(0)
            else:
                num_passed += 1
                valid_tracks.append(1)
                total_nObs += nObs
                if(nObs>max_length):
                    max_length = nObs
                if nObs < min_length:
                    min_length = nObs
                self.num_feature_tracks_residualized += 1
            
            if not num_passed:
                return 
            
            H_o = np.zeros((2 * total_nObs - 3 * num_passed,15 + 6 * len(self.cam_states)))
            R_o = np.zeros(2 * total_nObs - 3 * num_passed,2 * total_nObs - 3 * num_passed)
            r_o = np.zeros(((2 * total_nObs - 3 * num_passed),))
   
            rep = np.array([self.noise_params.u_var_prime, self.noise_params.v_var_prime])

            stack_counter = 0
            for iter in range(len(self.feature_tracks_to_residualize)):
                if not valid_tracks[iter]:
                    continue
                track = self.feature_tracks_to_residualize[iter]
                p_f_G = p_f_G_vec[iter]
                r_j = self.calcResidual(p_f_G, track.cam_states, track.observations)
                
                nObs = len(track.observations)
                R_j = np.tile(rep,(nObs,1))
                R_j = np.diag(R_j)

                H_o_j, A_j = self.calcMeasJacobian(p_f_G, track.cam_state_indices)
                r_o_j = np.dot(A_j.T, r_j)
                R_o_j = np.dot(np.dot(A_j.T, R_j), A_j)

                if (self.gatingTest(H_o_j, r_o_j, len(track.cam_states) - 1)):
                    r_o[stack_counter, stack_counter+len(r_o_j.size)] = r_o_j
                    H_o[stack_counter:stack_counter+H_o_j.shape[0], :H_o_j.shape[1]] = H_o_j
                    R_o[stack_counter:stack_counter+ R_o_j.shape[0],stack_counter:stack_counter+R_o_j.shape[1]]=R_o_j
                
                    stack_counter += H_o_j.shape[0]

            H_o.resize(stack_counter, H_o.shape[1])
            r_o.resize(stack_counter,1)
            R_o.resize(stack_counter, stack_counter)
            
            self.measurementUpdate(H_o, r_o, R_o)

    # Removes camera states that no longer contain any active observations
    def pruneEmptyStates(self):
        max_states = self.msckf_params.max_cam_states
        if (len(self.cam_states) < max_states):
            return
        deleteIdx = []
        num_states = len(self.cam_states)

        num_deleted = 0
        camstate_pos = 0
        num_cam_states = len(self.cam_states)

        last_to_remove = num_cam_states - max_states-1

        if(len(self.cam_states[0].tracked_feature_ids)):
            return
        
        for i in range(num_cam_states - max_states):
          if (len(self.cam_states[i].tracked_feature_ids)):
            last_to_remove = i - 1
            break
        
        index = []
        for i in range(last_to_remove): 
          deleteIdx.append(camstate_pos + num_deleted)
          self.pruned_states.append(self.cam_states[i])
          num_deleted += 1
        
        self.cam_states = np.delete(self.cam_states,index).tolist()
        
        if (len(deleteIdx) != 0):
            n_remove = 0
            n_keep = 0
            to_keep = np.zeros((num_states, 1))
            for IDx in range(num_states):
                find_index = np.where(deleteIdx==IDx)[0]
                if find_index is not None:
                    n_remove += 1
                else:
                    to_keep[IDx] = True
                    n_keep += 1

            remove_counter = 0
            keep_counter = 0
            keepCovarIdx = np.zeros((6 * n_keep,))
            removeCovarIdx = np.zeros((6 * n_remove,))
            for IDx in range(num_states):
                if not to_keep[IDx]:
                    removeCovarIdx[6 * remove_counter:6 * remove_counter+6] = np.linspace(6*IDx,6*(IDx+1)-1,6)
                    remove_counter += 1
                else:
                    keepCovarIdx[6 * keep_counter:6 * keep_counter+6] =np.linspace(6*IDx,6*(IDx+1)-1,6)
                    keep_counter += 1

            n = keepCovarIdx.shape[0]

            prunedCamCovar = self.cam_covar[:n,:n]
            self.cam_covar = prunedCamCovar

            prunedImuCamCovar = self.imu_cam_covar[:,:n]
            self.imu_cam_covar = prunedImuCamCovar