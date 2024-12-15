import numpy as np
import cv2 as cv
import csv
import os
import pandas as pd
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import math



class KITTIDatasetWriter:
    def __init__(self, input_path,output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.gt_path = input_path + '\\gt_data'
        self.pointcloud_path = input_path + '\\velodyne2\\'
        self.fx = 9.597910e+02*1.5
        self.fy = 9.599251e+02*1.5
        self.cx = 6.960217e+02
        self.cy = 275
        self.K = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])
        # self.data = pd.read_csv(self.gt_path)
        self.dist_coeff = np.array([-3.691481e-01, 1.968681e-01, 1.353473e-03, 5.677587e-04, -6.770705e-02])
        self.data = None
        self.gt_data_folders = ['gt100.csv','gt101.csv','gt102.csv','gt103.csv','gt104.csv','gt105.csv','gt106.csv','gt107.csv','gt108.csv','gt109.csv','gt110.csv','gt111.csv','gt112.csv','gt113.csv','gt114.csv','gt115.csv','gt116.csv','gt117.csv','gt118.csv','gt119.csv','gt120.csv']
        
        

    def read_data(self,file_name):
        # gt_data_folders = ['gt100.csv','gt101.csv','gt102.csv','gt103.csv','gt104.csv','gt105.csv','gt106.csv','gt107.csv','gt108.csv','gt109.csv','gt110.csv','gt111.csv','gt112.csv','gt113.csv','gt114.csv','gt115.csv','gt116.csv','gt117.csv','gt118.csv','gt119.csv','gt120.csv']
        # for file_name in gt_data_folders:
        gt_filepath = os.path.join(self.gt_path, file_name)
        gt_data = pd.read_csv(gt_filepath)
        self.data = self.clean_data(gt_data)
        # self.data = pd.concat([self.data, gt_data], ignore_index=True)
        return self.data
    
    def clean_data(self,data):
        data['SimTime'] = np.round(data['SimTime'], 2)
        data.drop('WallTime', inplace=True, axis=1)
        data.drop('PropertyType', inplace=True, axis=1)
        data.drop('InstanceType', inplace=True, axis=1)
        data.drop('ModelInstanceIdPath', inplace=True, axis=1)
        index_to_keep = data[data['SimTime'] >= 6.0].index.min()

        # Drop rows before the index_to_keep
        if index_to_keep is not None:
            data = data.iloc[index_to_keep:]
        return data
    
    
        

    def data_array_from_column(self,df, column_name):
        vectors = df[column_name].apply(eval)
        vectors_df = pd.DataFrame(vectors.tolist())
        return np.asarray(vectors_df)

    def generate_camera_TF_array(self):
        tf_camera_data = self.data.loc[self.data['InstanceName']=='FrontCamera CCS']
        tf_camera_array = self.data_array_from_column(tf_camera_data,'Data').reshape(-1,4,4)
        return tf_camera_array
    
    def generate_lidar_TF_array(self):
        tf_lidar_data = self.data.loc[self.data['InstanceName']=='Velodyne HDL-64E']
        tf_lidar_array = self.data_array_from_column(tf_lidar_data,'Data').reshape(-1,4,4)
        return tf_lidar_array
    
    
       

    def generate_filtered_data(self):
        childsizes_all = self.data.loc[self.data['PropertyName']=='childsizes']
        childsizes_array = self.data_array_from_column(childsizes_all,'Data')
        childpositions_all = self.data.loc[self.data['PropertyName']=='childpositions']
        childpositions_array = self.data_array_from_column(childpositions_all,'Data')
        childorientations_all = self.data.loc[self.data['PropertyName']=='childorientations']
        childorientations_array = self.data_array_from_column(childorientations_all,'Data')

        childsizes_filtered = []
        childpositions_filtered = []
        childorientations_filtered = []

        for i in range(childsizes_array.shape[0]):
            childsizes_array_temp = [x for x in childsizes_array[i] if x is not None]
            childpositions_array_temp = [x for x in childpositions_array[i] if x is not None]   
            childorientations_array_temp = [x for x in childorientations_array[i] if x is not None]
            childpositions_filtered.append(childpositions_array_temp)
            childsizes_filtered.append(childsizes_array_temp) 
            childorientations_filtered.append(childorientations_array_temp)
        
        return childsizes_filtered, childpositions_filtered, childorientations_filtered
    
    def generate_box_dimensions(self, sizes, positions, orientations):
        

        BB_length = []
        BB_width = []
        BB_height = []
        BB_center = []
        BB_orientation = []

        for i in range(len(sizes)):
            for j in range(len(sizes[i])):
                
                length = (sizes[i][j][0])
                width = (sizes[i][j][1])
                height = (sizes[i][j][2])
                center = positions[i][j]
                theta = orientations[i][j][0]
                BB_length.append(length)
                BB_width.append(width)
                BB_height.append(height)
                BB_center.append(center)
                BB_orientation.append(theta)

        total_boxes = len(BB_center)
        self.BB_dimensions = {}
        frames = int(len(sizes)) # 
        objects = int(total_boxes/frames)

        
        
        self.BB_dimensions['length'] = np.reshape(BB_length, (frames, objects))
        self.BB_dimensions['width'] = np.reshape(BB_width, (frames, objects))
        self.BB_dimensions['height'] = np.reshape(BB_height, (frames, objects))
        self.BB_dimensions['center'] = np.reshape(BB_center, (frames, objects, 3))
        self.BB_dimensions['orientation'] = np.reshape(BB_orientation, (frames, objects))
        self.BB_dimensions['frames'] = frames
        self.BB_dimensions['objects'] = objects
        
    
    def generate_truncation(self,position_from_camera):
        truncation = []
        for i in range(self.BB_dimensions['frames']):
            for j in range(self.BB_dimensions['objects']):
                center = position_from_camera[i][j]
                l = self.BB_dimensions['length'][i][j]*0.8
                w = self.BB_dimensions['width'][i][j]*0.67
                h = self.BB_dimensions['height'][i][j]*0.5
                if self.BB_dimensions['orientation'][i][j] >-0.7853 and self.BB_dimensions['orientation'][i][j] < 0.7853:
                    car_left_end = np.array([center[0] + w/2, center[1], center[2]]) # sideways
                    car_right_end = np.array([center[0] - w/2, center[1], center[2]])
                else:
                    car_left_end = np.array([center[0] + l/2, center[1], center[2]])
                    car_right_end = np.array([center[0] - l/2, center[1], center[2]])
                car_bottom = np.array([center[0], center[1]-h/2, center[2]]) # infront
                car_roof = np.array([center[0], center[1]+h/2, center[2]])
                # car_right = [center[0] + w/2, center[1], center[2]]
                # car_left = [center[0] - w/2, center[1], center[2]]
                proj_pt = self.K@center.T
                proj_right_end = self.K@car_right_end.T
                proj_left_end = self.K@car_left_end.T
                proj_roof = self.K@car_roof.T
                proj_bottom = self.K@car_bottom.T
                # proj_right = K@car_right.T
                
                u,v = (proj_pt[:-1] / proj_pt[-1]).astype(int)
                p,_ = (proj_right_end[:-1] / proj_right_end[-1]).astype(int)
                q,_ = (proj_left_end[:-1] / proj_left_end[-1]).astype(int)
                _,n = (proj_bottom[:-1] / proj_bottom[-1]).astype(int)
                _,m = (proj_roof[:-1] / proj_roof[-1]).astype(int)
                proj_l = (p-u)*2
                # proj_w = (s-u)*2
                proj_h = (n -v)*2
                if p < 1392 and q > 0  and n < 512:
                    trunc = 0
                elif q >= 1392 or p <= 0 or m>=512 :
                    trunc = 1
                elif q<1392 and p>1392:
                    trunc = (p - 1392)/proj_l
                elif q < 0 and p > 0:
                    trunc = (0-q)/proj_l
                elif n > 512 and m < 512:
                    trunc = (n-512)/proj_h
                
                truncation.append(trunc)
        truncation = np.reshape(truncation,(self.BB_dimensions['frames'],self.BB_dimensions['objects']))       
            
        return truncation
    
    def generate_V2C_array(self,tf_lidar_array,tf_camera_array):
        V2C_array = []
        for i in range(len(tf_lidar_array)):
            inv_lidar_tf = np.linalg.inv(tf_lidar_array[i])
            V2C = np.dot(inv_lidar_tf,tf_camera_array[i])
            flip_tf = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]) #rotational correction
            V2C = np.dot(flip_tf,V2C)
            V2C_array.append(V2C)
        return V2C_array
    
    def generate_projection_matrix(self,camera_array):
        P = []
        for i in range(len(camera_array)):
            inv_Tf = np.linalg.inv(camera_array[i])
            Rot_cam = inv_Tf[:3,:3]
            Trans_cam = np.reshape(inv_Tf[:3,3],(3,1)) + np.reshape([0,0.5,1.0],(3,1)) # translation correction
            R_T = np.hstack((Rot_cam,Trans_cam))
            P_mat = np.dot(self.K, R_T)
            P.append(P_mat)
        return P

    
    def generate_calib_txt(self, V2C_array, P_array,calib_path):
        """
        Generate calibration file for each frame
        """
        # calib_path = self.input_path + 'calib/'
         
        
        if not os.path.exists(calib_path):
            os.makedirs(calib_path)
        

        for i in range(len(P_array)):
            calib_file = open(calib_path + str(i).zfill(6) + '.txt', 'w')
            calib_file.write('P0: 0 0 0 0 0 0 0 0 0 0 0 0\n')
            calib_file.write('P1: 0 0 0 0 0 0 0 0 0 0 0 0\n')
            
            P_mat = P_array[i]
            
            calib_file.write('P2: ' + ' '.join(map(str, P_mat.flatten())) + '\n')
            # calib_file.write(f'P2: {P_mat[0]} {P_mat[1]} {P_mat[2]} {P_mat[3]} {P_mat[4]} {P_mat[5]} {P_mat[6]} {P_mat[7]} {P_mat[8]} {P_mat[9]} {P_mat[10]} {P_mat[11]}\n')
            calib_file.write('P3: 0 0 0 0 0 0 0 0 0 0 0 0\n')
            calib_file.write('R0_rect: 1 0 0 0 1 0 0 0 1\n')
            
            V_t_C = V2C_array[i]
            calib_file.write('Tr_velo_to_cam: ' + ' '.join(map(str, V_t_C.flatten())) + '\n')
            calib_file.write('Tr_imu_to_velo: 0 0 0 0 0 0 0\n')
            calib_file.close()
        
    
    def generate_reference_position(self,tf_array,position,camera_flag):
        reference_position = []
        for i in range(self.BB_dimensions['frames']):
            for j in range(self.BB_dimensions['objects']):
                pos = np.reshape(position[i][j],(3,1))
                inv_TF = np.linalg.inv(tf_array[i].reshape(4,4))
                Rot = inv_TF[:3,:3]
                if camera_flag:
                    translation = np.reshape(inv_TF[:3,3],(3,1)) + np.reshape([0,0.5,1.0],(3,1))
                else:
                    translation = np.reshape(inv_TF[:3,3],(3,1))
                pos = np.dot(Rot,pos)
                pos = pos + translation
                
                reference_position.append(pos)
        reference_position = np.reshape(reference_position, (self.BB_dimensions['frames'], self.BB_dimensions['objects'], 3))
        return reference_position     
                
                
    def clipping_index_3d_boxes(self,position_from_lidar):
        filtered_bb_index = []
        Lidar_clipping_range = [5, -20, -3, 40.4, 20, 1]
        for i in range(self.BB_dimensions['frames']):
            temp_list = []
            for idx, j in enumerate(position_from_lidar[i]):
                if j[0] < Lidar_clipping_range[4] and j[0] > Lidar_clipping_range[1] and j[1] < Lidar_clipping_range[3] and j[1] > Lidar_clipping_range[0] and j[2] < Lidar_clipping_range[5] and j[2] > Lidar_clipping_range[2]:
                    temp_list.append(idx)
            filtered_bb_index.append(temp_list)
        return filtered_bb_index

    
    def generate_alpha(self,lidar_bbox, rotation_y):
        alpha = []
        for i in range((self.BB_dimensions['frames'])):
            for j in range((self.BB_dimensions['objects'])):
                yaw = np.arctan2(-lidar_bbox[i][j][1], lidar_bbox[i][j][0])
                alpha_diff = yaw - rotation_y[i][j]
                alpha.append(alpha_diff)
        alpha = np.reshape(alpha, (self.BB_dimensions['frames'], self.BB_dimensions['objects']))
        return alpha


    def calculate_rotation_y(self,tf_camera_array):
        rotation_y = []
        for i in range((self.BB_dimensions['frames'])):
            for j in range((self.BB_dimensions['objects'])):
                yaw_vehicle = np.rad2deg(self.BB_dimensions['orientation'][i][j])
                r = R.from_matrix(tf_camera_array[i][:3,:3])
                roll, pitch, yaw = (r.as_euler('xyz', degrees=True))
                angle_diff = np.deg2rad((yaw+90.0)-yaw_vehicle)
                rotation_y.append(angle_diff)
        rotation_y = np.reshape(rotation_y, (self.BB_dimensions['frames'], self.BB_dimensions['objects']))
        return rotation_y
    
       
    def generate_bbox(self,position, rotation_y, camera_flag):
        BB_coordinates = []
        # BB_coordinates_lidar = []
        for i in range((self.BB_dimensions['frames'])):
            for j in range((self.BB_dimensions['objects'])):
                    l, h, w = self.BB_dimensions['length'][i][j],self.BB_dimensions['height'][i][j], self.BB_dimensions['width'][i][j]

                    if camera_flag:
                        x_corners = [w / 2, w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2]
                        y_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
                        z_corners = [l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2]
                    else:
                        x_corners = [w / 2, w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2]
                        y_corners = [l / 2, l / 2, l / 2, l / 2, -l / 2, -l / 2, -l / 2, -l / 2]
                        z_corners = [h / 2, -h / 2, -h / 2, h / 2, h / 2, -h / 2, -h / 2, h / 2]
                    

                    R = np.array([[np.cos(rotation_y[i][j]), 0, np.sin(rotation_y[i][j])],
                                    [0, 1, 0],
                                    [-np.sin(rotation_y[i][j]), 0, np.cos(rotation_y[i][j])]])
                    corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
                    corners3d = np.dot(R, corners3d).T
                    corners3d = corners3d + position[i][j]
                    BB_coordinates.append(corners3d)
                    
        BB_coordinates = np.reshape( BB_coordinates,(self.BB_dimensions['frames'],self.BB_dimensions['objects'],8,3))
        return BB_coordinates
    
    def generate_image_3d_bbox(self,BB_coordinates_camera):
        image_bb_corners = []
        for i in range(self.BB_dimensions['frames']):
            for j in range(self.BB_dimensions['objects']):
                for k in range(len(BB_coordinates_camera[i][j])):
                    translated_point = BB_coordinates_camera[i][j][k]
                    proj_pt = self.K@translated_point.T
                    point_2d = (proj_pt[:-1] / proj_pt[-1]).astype(int)
                    image_bb_corners.append(point_2d)
        image_bb_corners = np.reshape(image_bb_corners,(self.BB_dimensions['frames'],self.BB_dimensions['objects'],8,2))
        return image_bb_corners
    
    def generate_2d_bbox(self,image_bb_corners):
        bb_2d = []
        for i in range(self.BB_dimensions['frames']):
            for j in range(self.BB_dimensions['objects']):
                x_min = np.min(image_bb_corners[i][j][:,0])
                x_max = np.max(image_bb_corners[i][j][:,0])
                y_min = np.min(image_bb_corners[i][j][:,1])
                y_max = np.max(image_bb_corners[i][j][:,1])
                bb_2d.append([x_min,y_min,x_max,y_max])
        bb_2d = np.reshape(bb_2d,(self.BB_dimensions['frames'],self.BB_dimensions['objects'],4))
        return bb_2d


    #Occlusion criteria 0:fully visible 1:partly occluded 2:largely occluded 3:unknown
    def get_object_info(self,postion_from_lidar,rotation_y):
        object_info = []
        for i in range(self.BB_dimensions['frames']):
            for j in range(self.BB_dimensions['objects']):
                length = self.BB_dimensions['length'][i][j]
                width = self.BB_dimensions['width'][i][j]
                height = self.BB_dimensions['height'][i][j]
                center = postion_from_lidar[i][j]
                orientation = rotation_y[i][j]
                object_data = np.array([center[0],center[1],center[2],length, width, height, orientation])
                object_info.append(object_data)

        # print(np.shape(object_info))
        object_info = np.reshape(object_info,(self.BB_dimensions['frames'],self.BB_dimensions['objects'],7))
        return object_info
    
    def get_points_in_box(self,points, object):
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        cx, cy, cz = object[0], object[1], object[2]
        dx, dy, dz, rz = object[3], object[4], object[5], object[6]
        shift_x, shift_y, shift_z = x - cx, y - cy, z - cz

        MARGIN = 1e-1
        cosa, sina = math.cos(-rz), math.sin(-rz)
        local_x = shift_x * cosa + shift_y * (-sina)
        local_y = shift_x * sina + shift_y * cosa

        mask = np.logical_and(abs(shift_z) <= dz / 2.0,
                            np.logical_and(abs(local_x) <= dx / 2.0 + MARGIN,
                                            abs(local_y) <= dy / 2.0 + MARGIN))

        points = points[mask]

        return points
    

    def generate_occlusion(self,gt_boxes):
        # Load pointcloud
        # total_pointclouds = np.linspace(0,12634,1)
        occlusion_array = []
        threshold_high = 100
        threshold_low = 50
        for i in range(self.BB_dimensions['frames']): 
            pointcloud_file = self.pointcloud_path + f"\\000000{i}"[-6:]+".npy"  # Format with 6 digits and ".npy" extension
            pointcloud = np.fromfile(pointcloud_file, dtype=np.float32).reshape(-1, 4)  # Range includes 0 up to 12634 (inclusive)
            for j in range(self.BB_dimensions['objects']):
                points_in_box = self.get_points_in_box(pointcloud, gt_boxes[i][j])
                if len(points_in_box) > threshold_high:
                    occlusion = 0
                elif len(points_in_box) > threshold_low and len(points_in_box) < threshold_high:
                    occlusion = 1
                else:
                    occlusion = 2
                occlusion_array.append(occlusion)
        occlusion_array = np.reshape(occlusion_array,(self.BB_dimensions['frames'],self.BB_dimensions['objects']))
        return occlusion_array
    
    def to_kitti_format(self):
        kitti_str = '%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' \
                    % (self.cls_type, self.truncation, int(self.occlusion), self.alpha, self.box2d[0], self.box2d[1],
                       self.box2d[2], self.box2d[3], self.h, self.w, self.l, self.loc[0], self.loc[1], self.loc[2],
                       self.ry)
        return kitti_str
    
    def write_dataset(self):
        """
        Write label file for each frame
        """
        for i,file_name in enumerate(self.gt_data_folders):
            label_path = self.output_path + f'/label_{i}/'
            calib_path = self.output_path + f'/calib_{i}/'
            # print(label_path)
            if not os.path.exists(label_path):
                os.makedirs(label_path)

            gt_data = self.read_data(file_name)
            # print(gt_data.shape)
            sizes, positions, orientations = self.generate_filtered_data()
            self.generate_box_dimensions(sizes, positions, orientations)
            tf_camera_array = self.generate_camera_TF_array()
            # print(tf_camera_array.shape)
            tf_lidar_array = self.generate_lidar_TF_array()
            V2C_array = self.generate_V2C_array(tf_lidar_array, tf_camera_array)
            P_array = self.generate_projection_matrix(tf_camera_array)
            self.generate_calib_txt(V2C_array, P_array,calib_path)
            position_from_lidar = self.generate_reference_position(tf_lidar_array, positions, False)
            position_from_camera = self.generate_reference_position(tf_camera_array, positions, True)
            filtered_bb_index = self.clipping_index_3d_boxes(position_from_lidar)
            rotation_y = self.calculate_rotation_y(tf_camera_array)
            alpha = self.generate_alpha(position_from_lidar, rotation_y)
            truncation = self.generate_truncation(position_from_camera)
            gt_boxes = self.get_object_info(position_from_lidar, rotation_y)
            occlusion = self.generate_occlusion(gt_boxes)
            BB_coordinates_camera = self.generate_bbox(position_from_camera, rotation_y, True)
            BB_coordinates_lidar = self.generate_bbox(position_from_lidar, rotation_y, False)
            image_bb_corners = self.generate_image_3d_bbox(BB_coordinates_camera)
            bb_2d = self.generate_2d_bbox(image_bb_corners)

            for i in range(self.BB_dimensions['frames']):
                label_file = open(label_path + str(i).zfill(6) + '.txt', 'w')
                for j in (filtered_bb_index[i]):
                    self.cls_type = 'Car'
                    self.truncation = truncation[i][j]
                    self.occlusion = occlusion[i][j]
                    self.alpha = alpha[i][j]
                    self.box2d = bb_2d[i][j]
                    self.h = gt_boxes[i][j][5]
                    self.w = gt_boxes[i][j][4]
                    self.l = gt_boxes[i][j][3]
                    self.loc = position_from_camera[i][j][:3]
                    self.ry = rotation_y[i][j]
                    label_file.write(self.to_kitti_format() + '\n')
                label_file.close()




        
        
       

  
    

    
        
    
