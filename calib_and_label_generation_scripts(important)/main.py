from scripts.KITTI_dataset_writer import KITTIDatasetWriter
from pathlib import Path

def main():
    # Define the path to the source directory
    input_path = "F:\\master_thesis"
    output_path = "F:\\master_thesis\\simulated_dataset"
   

    # Create an instance of the KITTIDatasetWriter class
    writer = KITTIDatasetWriter(input_path,output_path)

    # Write the dataset to the output directory
    writer.write_dataset()

if __name__ == "__main__":
    main()