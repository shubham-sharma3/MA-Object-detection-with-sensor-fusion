import os
import shutil

def gather_and_rename_files(source_dirs, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    file_counter = 0

    for source_dir in source_dirs:
        for root, _, files in os.walk(source_dir):
            for file in sorted(files):
                file_ext = os.path.splitext(file)[1]
                new_file_name = f"{file_counter:06d}{file_ext}"
                source_file_path = os.path.join(root, file)
                target_file_path = os.path.join(target_dir, new_file_name)

                shutil.copy2(source_file_path, target_file_path)
                file_counter += 1

# List of source directories
def main():
    source_path = "D:\\master_thesis\\kitti_dataset\\calibration"
    source_dirs = []
    for file in os.listdir(source_path):
        file_dir = os.path.join(source_path,file)
        source_dirs.append(file_dir)

    # Target directory to gather and rename files
    target_dir = 'D:\\master_thesis\\kitti_dataset\\calib'

    # Call the function
    gather_and_rename_files(source_dirs, target_dir)

if __name__ == "__main__":
    main()
