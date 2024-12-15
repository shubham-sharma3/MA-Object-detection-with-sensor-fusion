import numpy as np
import os
from config_loader import load_parameters

# Camera parameters
# Camera intrinsic parameters
params = load_parameters('camera_parameters.yaml')
fx = params['fx']
fy = params['fy']
cx = params['cx']
cy = params['cy']

K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

def generate_calib_txt(calib_path, TF_camera_array, TF_lidar_array):
    """
    Generate calibration file for each frame
    """

    if not os.path.exists(calib_path):
        os.makedirs(calib_path)
    

    for i in range(len(TF_camera_array)):
        calib_file = open(calib_path + str(i).zfill(6) + '.txt', 'w')
        calib_file.write('P0: 0 0 0 0 0 0 0 0 0 0 0 0\n')
        calib_file.write('P1: 0 0 0 0 0 0 0 0 0 0 0 0\n')
        R_T = TF_camera_array[i][:3, :]
        P_mat = np.dot(K, R_T)
        calib_file.write('P2: ' + ' '.join(map(str, P_mat.flatten())) + '\n')
        # calib_file.write(f'P2: {P_mat[0]} {P_mat[1]} {P_mat[2]} {P_mat[3]} {P_mat[4]} {P_mat[5]} {P_mat[6]} {P_mat[7]} {P_mat[8]} {P_mat[9]} {P_mat[10]} {P_mat[11]}\n')
        calib_file.write('P3: 0 0 0 0 0 0 0 0 0 0 0 0\n')
        calib_file.write('R0_rect: 1 0 0 0 1 0 0 0 1\n')
        W_t_V = TF_lidar_array[i].reshape(4,4)
        W_t_C = TF_camera_array[i].reshape(4,4)
        V_t_W = np.linalg.inv(W_t_V)
        V_t_C = V_t_W @ W_t_C
        # Flip the camera frame
        flip_tf = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
        V_t_C = np.dot(flip_tf,V_t_C)

        calib_file.write('Tr_velo_to_cam: ' + ' '.join(map(str, V_t_C.flatten())) + '\n')
        calib_file.write('Tr_imu_to_velo: 0 0 0 0 0 0 0\n')

        calib_file.close()

