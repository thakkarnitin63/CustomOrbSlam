import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from orb_slam.image_loader import ImageLoader
from orb_slam.feature_extractor import FeatureExtractor
from orb_slam.feature_matcher import FeatureMatcher
from orb_slam.pose_estimator import PoseEstimator

def load_ground_truth_poses(file_path):
    """
    Loads ground truth poses from a text file.
    Each line in the file should represent a 3x4 transformation matrix.
    """
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            T = np.fromstring(line, sep=' ').reshape(3, 4)
            pose = np.eye(4)
            pose[:3, :4] = T
            poses.append(pose)
    return np.array(poses)

def main():
    # Define the paths to your dataset sequence and ground truth poses
    sequence_path = '/home/nitin/NitinWs/CustomOrbSlam/data/dataset/sequences/00'
    gt_poses_path = '/home/nitin/NitinWs/CustomOrbSlam/data/dataset/poses/00.txt'

    # Load ground truth poses
    ground_truth_poses = load_ground_truth_poses(gt_poses_path)

    # Initialize the ImageLoader, FeatureExtractor, FeatureMatcher, and PoseEstimator
    image_loader = ImageLoader(sequence_path)
    feature_extractor = FeatureExtractor(nfeatures=500)
    feature_matcher = FeatureMatcher(ratio_thresh=0.75)

    # Load calibration parameters
    K = image_loader.calibration['P0']  # Extract the 3x3 intrinsic matrix

    # Initialize PoseEstimator with the intrinsic camera matrix
    pose_estimator = PoseEstimator(camera_matrix=K)

    # Initialize variables to store the trajectory
    trajectory = [np.eye(4)]

    # Process a sequence of images
    num_frames = min(len(ground_truth_poses), 10)  # Ensure we don't exceed available ground truth data
    for i in range(1, num_frames):
        # Load consecutive images
        image1 = image_loader.load_image(i - 1)
        image2 = image_loader.load_image(i)

        # Extract features from both images
        keypoints1, descriptors1 = feature_extractor.extract(image1)
        keypoints2, descriptors2 = feature_extractor.extract(image2)

        # Match features between the two images
        matches = feature_matcher.match(descriptors1, descriptors2)

        # Estimate the pose between the two frames
        R, t = pose_estimator.estimate_pose(keypoints1, keypoints2, matches)

        # Construct the transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()

        # Update the current pose
        current_pose = trajectory[-1] @ np.linalg.inv(T)
        trajectory.append(current_pose)

    # Convert trajectory to numpy array for plotting
    trajectory = np.array(trajectory)

    # Extract X and Z coordinates for top-down view
    est_x = trajectory[:, 0, 3]
    est_z = trajectory[:, 2, 3]
    gt_x = ground_truth_poses[:num_frames, 0, 3]
    gt_z = ground_truth_poses[:num_frames, 2, 3]

    # Plot the top-down trajectory
    plt.figure()
    plt.plot(est_x, est_z, marker='o', label='Estimated Trajectory')
    plt.plot(gt_x, gt_z, marker='x', label='Ground Truth Trajectory')
    plt.xlabel('X Position')
    plt.ylabel('Z Position')
    plt.title('Top-Down View of Trajectories')
    plt.legend()
    plt.grid()
    plt.axis('equal')
    plt.savefig('top_down_trajectory_comparison.png')
    plt.show()

if __name__ == '__main__':
    main()
