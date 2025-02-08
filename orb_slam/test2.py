import numpy as np
import cv2
import matplotlib.pyplot as plt

from orb_slam.image_loader import ImageLoader
from orb_slam.feature_extractor import FeatureExtractor
from orb_slam.feature_matcher import FeatureMatcher


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
    image1 = image_loader.load_image(0)
    image2 = image_loader.load_image(1)
    feature_extractor = FeatureExtractor(image1)
    matcher = FeatureMatcher()

    keypoints1, descriptors1 = feature_extractor.extract(image1)
    keypoints2, descriptors2 = feature_extractor.extract(image2)

    # img_with_kp = cv2.drawKeypoints(image1, keypoints1, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # print(f"Keypoints {len(keypoints1)} Descriptors {len(descriptors1)}")
    # keypoints2, descriptors2 = feature_extractor.extract(image1)
    # cv2.imshow("Image", img_with_kp)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print(f"Number of keypoints in Image 1: {len(keypoints1)}")
    # print(f"Number of keypoints in Image 2: {len(keypoints2)}")
    
    matches = matcher.match(descriptors1, descriptors2)
    filtered_matches = matcher.filter_matches(matches, keypoints1, keypoints2)

    print(f"Number of matches: {len(matches)}")
    print(f"Filtered Matches: {len(filtered_matches)}")
    matcher.visualize_matches(image1, image2, keypoints1, keypoints2, filtered_matches)


if __name__ == "__main__":
    main()