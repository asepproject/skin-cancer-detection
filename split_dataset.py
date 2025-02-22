import os
import shutil
import random

def split_dataset_into_test(original_data_dir, test_dir, test_split=0.1):
    """
    Moves a portion of images from the original dataset to the test directory.

    Args:
        original_data_dir (str): Path to the train dataset directory.
        test_dir (str): Path to the test dataset directory.
        test_split (float): Proportion of images to allocate to the test set.
    """
    # Ensure the test directory exists
    os.makedirs(test_dir, exist_ok=True)

    # Get all class names (subfolders in the train data directory)
    classes = os.listdir(original_data_dir)

    for class_name in classes:
        class_dir = os.path.join(original_data_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)

        # Ensure class subdirectory exists in the test directory
        os.makedirs(test_class_dir, exist_ok=True)

        # Get all images in the class
        images = os.listdir(class_dir)
        random.shuffle(images)

        # Move test_split% of images to the test directory
        test_size = int(test_split * len(images))
        for image in images[:test_size]:
            src = os.path.join(class_dir, image)
            dest = os.path.join(test_class_dir, image)
            shutil.move(src, dest)

    print(f"Test set created successfully at '{test_dir}'.")

# Example usage
if __name__ == "__main__":  # Fixed __name__ and __main__
    # Define paths
    original_data_dir = "organized_dataset/train"  # Path to the train dataset
    test_dir = "organized_dataset/test"           # Path to the test dataset

    # Call the function to create the test set
    split_dataset_into_test(original_data_dir, test_dir, test_split=0.1)
