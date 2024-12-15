import os
import random
import shutil
import argparse

def split_dataset(data_size, val_split=0.2, test_split=0.1, seed=42):
    # Set random seed for reproducibility
    random.seed(seed)

    # Generate indices
    indices = list(range(data_size))
    random.shuffle(indices)

    # Determine split sizes
    val_size = int(data_size * val_split)
    test_size = int(data_size * test_split)
    train_size = data_size - val_size - test_size

    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    return train_indices, val_indices, test_indices

def move_files(indices, source_dirs, target_dir):
    for idx in indices:
        for dir_name in source_dirs.keys():
            src_file = os.path.join(source_dirs[dir_name], f"{idx:06d}.txt")
            if not os.path.exists(src_file):
                src_file = os.path.join(source_dirs[dir_name], f"{idx:06d}.bin")
                if not os.path.exists(src_file):
                    src_file = os.path.join(source_dirs[dir_name], f"{idx:06d}.png")
            target_file = os.path.join(target_dir, dir_name, f"{idx:06d}.txt")
            if src_file.endswith('.bin'):
                target_file = target_file.replace('.txt', '.bin')
            elif src_file.endswith('.png'):
                target_file = target_file.replace('.txt', '.png')
            shutil.copy2(src_file, target_file)

def create_imagesets_files(indices, file_name):
    with open(file_name, 'w') as f:
        for idx in sorted(indices):
            f.write(f"{idx:06d}\n")

def main(data_dir, val_split=0.2, test_split=0.1, seed=42):
    data_size = 12635
    train_indices, val_indices, test_indices = split_dataset(data_size, val_split, test_split, seed)

    # Define source directories
    source_dirs = {
        "image_2": os.path.join(data_dir, "image_2"),
        "label_2": os.path.join(data_dir, "label_2"),
        "calib": os.path.join(data_dir, "calib"),
        "velodyne": os.path.join(data_dir, "velodyne")
    }

    # Define target directories
    target_base_dir = os.path.dirname(data_dir)
    train_dir = os.path.join(target_base_dir, "training")
    val_dir = os.path.join(target_base_dir, "validation")
    test_dir = os.path.join(target_base_dir, "testing")

    # Create target directories if they do not exist
    for dir_name in source_dirs.keys():
        os.makedirs(os.path.join(train_dir, dir_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, dir_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, dir_name), exist_ok=True)

    # Move files to the corresponding directories
    move_files(train_indices, source_dirs, train_dir)
    move_files(val_indices, source_dirs, val_dir)
    move_files(test_indices, source_dirs, test_dir)

    # Create ImageSets directory and files
    imagesets_dir = os.path.join(target_base_dir, "ImageSets")
    os.makedirs(imagesets_dir, exist_ok=True)
    create_imagesets_files(train_indices, os.path.join(imagesets_dir, "train.txt"))
    create_imagesets_files(val_indices, os.path.join(imagesets_dir, "val.txt"))
    create_imagesets_files(test_indices, os.path.join(imagesets_dir, "test.txt"))
    create_imagesets_files(train_indices + val_indices, os.path.join(imagesets_dir, "trainval.txt"))

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Split dataset into training, validation, and testing sets in KITTI format.')
    # parser.add_argument('data_dir', type=str, help='Directory containing the data')
    data_dir = "D:\\master_thesis\\kitti_dataset"
    main(data_dir)
