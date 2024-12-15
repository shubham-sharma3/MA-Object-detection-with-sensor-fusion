import open3d as o3d
import numpy as np
import os
from pyntcloud import PyntCloud as pt

def custom_linspace(start, num, gap):
    end = start + (num-1) * gap
    return np.linspace(start, end, num)

def process_lidar_data(input_folder, output_folder):
    # Create the output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)
    root = input_folder
    counter = 0
    for folder in os.listdir(root):
        total_files = len(os.listdir(os.path.join(root, folder)))
        timestep = custom_linspace(6000, total_files, 100).astype(int)
        for time in timestep:
            lidar_file = os.path.join(root, folder, f'output.pcd_{time}_{int(time/100)}.pcd')
            if os.path.exists(lidar_file):
                print(f'Processing {lidar_file}')
                # Load the point cloud from the binary file
                # pcd_read = o3d.io.read_point_cloud(lidar_file)
                pc = pt.from_file(lidar_file)
                # Create an Open3D point cloud object
                pcd_data = np.asarray(pc.points)
                # modified_points = np.hstack((pcd_data, np.ones((pcd_data.shape[0], 1))))
                # Save the point cloud as a .bin file
                output_filename = '{:06d}.bin'.format(counter)
                output_path = os.path.join(output_folder, output_filename)
                pcd_data.tofile(output_path)
                
                # Increment the counter for the next file
                counter += 1
                print(f'Saved {output_filename}')
            else:
                print(f'File {lidar_file} does not exist')

def main():
    # Define input and output folders
    input_root_folder = 'D:\\master_thesis\\pc_1'
    output_folder = 'D:\\master_thesis\\velo_train'
    
    # Process point clouds
    process_lidar_data(input_root_folder, output_folder)

if __name__ == '__main__':
    main()