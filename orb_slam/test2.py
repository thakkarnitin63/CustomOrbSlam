import numpy as np
import cv2
import matplotlib.pyplot as plt

from orb_slam.image_loader import ImageLoader
from orb_slam.feature_extractor import FeatureExtractor


def load_ground_truth_poses(file_path):
    """
    Load ground truth poses from a KITTI pose file.
    
    Args:
        file_path (str): Path to the poses.txt file.
        
    Returns:
        list: List of 4x4 numpy arrays representing the poses.
    """
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            pose = np.array(values).reshape(3, 4)
            pose = np.vstack((pose, [0, 0, 0, 1]))  # Convert to homogeneous matrix
            poses.append(pose)
    return poses

def main():
    # Paths (adjust these paths as needed)
    sequence_path = '/home/nitin/NitinWs/CustomOrbSlam/data/dataset/sequences/00'
    ground_truth_file = '/home/nitin/NitinWs/CustomOrbSlam/data/dataset/poses/00.txt'

    # Initialize system components
    image_loader = ImageLoader(sequence_path)
    sample_image =image_loader.load_image(99)
    feature_extractor = FeatureExtractor(sample_image)

    keypoints, descriptors = feature_extractor.extract(sample_image)

    print(len(keypoints))

    img_keypoints = cv2.drawKeypoints(sample_image, keypoints, None, color=(0, 255, 0))
    cv2.imshow("Keypoints", img_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()