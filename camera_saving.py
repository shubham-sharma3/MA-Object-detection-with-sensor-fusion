import os
import cv2
import numpy as np
from PIL import Image

# Define the range of folders and the destination folder
start_folder = 100
end_folder = 120
destination_folder = 'new_folder'
os.makedirs(destination_folder, exist_ok=True)

# Initialize the image counter
image_counter = 0

# Predefined camera parameters (replace these with your actual calibration values)
camera_matrix = np.array([[fx,  0, cx],
                          [ 0, fy, cy],
                          [ 0,  0,  1]], dtype=np.float32)

dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)

# Iterate through each folder in the specified range
for folder_num in range(start_folder, end_folder + 1):
    folder_path = f'{folder_num}'
    
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist.")
        continue
    
    # Iterate through each image in the folder
    for image_name in os.listdir(folder_path):
        # Construct full image path
        image_path = os.path.join(folder_path, image_name)
        
        # Check if the file is an image
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            # Read the image using OpenCV
            image = cv2.imread(image_path)
            
            # Check if the image is loaded
            if image is None:
                print(f"Failed to load image {image_path}")
                continue
            
            # Undistort the image
            undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)
            
            # Convert the undistorted image back to PIL format
            undistorted_image_pil = Image.fromarray(cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2RGB))
            
            # Define the new image name with zero-padding
            new_image_name = f"{image_counter:06}.png"
            new_image_path = os.path.join(destination_folder, new_image_name)
            
            # Save the undistorted image in the new folder
            undistorted_image_pil.save(new_image_path)
            
            # Increment the image counter
            image_counter += 1

print(f"Images have been undistorted and saved to '{destination_folder}'")
