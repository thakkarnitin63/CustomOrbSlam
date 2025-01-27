import numpy as np
from orb_slam.image_loader import ImageLoader
from orb_slam.feature_extractor import FeatureExtractor
from orb_slam.feature_matcher import FeatureMatcher
from orb_slam.pose_estimator import PoseEstimator
from orb_slam.visualizer import Visualizer
import matplotlib.pyplot as plt


def plot_translation_deviations(estimated_poses, ground_truth_poses):
    estimated_positions = np.array([pose[:3, 3] for pose in estimated_poses])
    ground_truth_positions = np.array([pose[:3, 3] for pose in ground_truth_poses])

    deviations = np.abs(estimated_positions - ground_truth_positions)

    plt.figure(figsize=(10, 6))
    plt.plot(deviations[:, 0], label='X Deviation')
    plt.plot(deviations[:, 1], label='Y Deviation')
    plt.plot(deviations[:, 2], label='Z Deviation')
    plt.xlabel('Frame Index')
    plt.ylabel('Deviation (meters)')
    plt.title('Translation Deviations Between Estimated and Ground Truth Poses')
    plt.legend()
    plt.grid()
    plt.show()

# Call this after RMSE computation



def load_ground_truth_poses(file_path):
    """
    Load ground truth poses from a KITTI pose file.
    :param file_path: Path to the poses.txt file.
    :return: List of 4x4 numpy arrays representing the poses.
    """
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            pose = np.array(values).reshape(3, 4)
            pose = np.vstack((pose, [0, 0, 0, 1]))  # Convert to homogeneous matrix
            poses.append(pose)
    return poses


def compute_rmse_per_frame(estimated_poses, ground_truth_poses):
    """
    Compute the RMSE between estimated and ground truth poses for each frame.
    :param estimated_poses: List of 4x4 numpy arrays of estimated poses.
    :param ground_truth_poses: List of 4x4 numpy arrays of ground truth poses.
    :return: List of RMSE values for each frame.
    """
    assert len(estimated_poses) == len(ground_truth_poses), "Pose lists must have the same length."
    rmse_values = []
    for est_pose, gt_pose in zip(estimated_poses, ground_truth_poses):
        translation_diff = est_pose[:3, 3] - gt_pose[:3, 3]
        rmse = np.sqrt(np.mean(translation_diff**2))
        rmse_values.append(rmse)
    return rmse_values





def initialize_first_pose(image_loader, feature_extractor, feature_matcher, pose_estimator):
    """
    Estimate the initial pose using the first two frames.
    :return: Initial rotation matrix (R), translation vector (t), pose (4x4 matrix), and descriptors for frame 2.
    """
    image1 = image_loader.load_image(0)
    image2 = image_loader.load_image(1)

    # Extract features
    keypoints1, descriptors1 = feature_extractor.extract(image1)
    keypoints2, descriptors2 = feature_extractor.extract(image2)

    # Match features
    matches = feature_matcher.match(descriptors1, descriptors2)
    filtered_matches = feature_matcher.filter_matches(matches, keypoints1, keypoints2)

    # Estimate relative pose
    R, t, _ = pose_estimator.estimate_pose(keypoints1, keypoints2, filtered_matches)

    # Construct the pose matrix for the second frame
    initial_pose = np.eye(4)
    initial_pose[:3, :3] = R
    initial_pose[:3, 3] = t.flatten()

    return R, t, initial_pose, keypoints2, descriptors2


def main():
    sequence_path = '/home/nitin/NitinWs/CustomOrbSlam/data/dataset/sequences/00'
    ground_truth_file = '/home/nitin/NitinWs/CustomOrbSlam/data/dataset/poses/00.txt'

    # Initialize components
    image_loader = ImageLoader(sequence_path)
    feature_extractor = FeatureExtractor()
    feature_matcher = FeatureMatcher(ratio_thresh=0.75)
    calibration_matrices = image_loader._load_calibration()
    K = calibration_matrices.get('P0')
    pose_estimator = PoseEstimator(K)
    visualizer = Visualizer()

    # Load ground truth poses
    ground_truth_poses = load_ground_truth_poses(ground_truth_file)

    # Initialize the first pose
    prev_R, prev_t, initial_pose, prev_keypoints, prev_descriptors = initialize_first_pose(
        image_loader, feature_extractor, feature_matcher, pose_estimator
    )
    estimated_poses = [np.eye(4), initial_pose]

    num_frames_to_process = 1000
    for frame_id in range(2, num_frames_to_process):  # Start from the third frame
        current_image = image_loader.load_image(frame_id)

        # Extract features
        keypoints, descriptors = feature_extractor.extract(current_image)

        # Match features
        matches = feature_matcher.match(prev_descriptors, descriptors)
        filtered_matches = feature_matcher.filter_matches(matches, prev_keypoints, keypoints)

        try:
            R, t, _ = pose_estimator.estimate_pose(prev_keypoints, keypoints, filtered_matches)
            current_pose = np.eye(4)
            current_pose[:3, :3] = np.dot(R, prev_R)
            current_pose[:3, 3] = prev_R.dot(t).flatten() + prev_t.flatten()

            estimated_poses.append(current_pose)
            prev_R, prev_t = current_pose[:3, :3], current_pose[:3, 3].reshape(3, 1)
        except ValueError as e:
            print(f"Pose estimation failed at frame {frame_id}: {e}")
            continue

        # Update for the next iteration
        prev_keypoints, prev_descriptors = keypoints, descriptors

    # Adjust estimated poses and ground truth lengths
    min_length = min(len(estimated_poses), len(ground_truth_poses[:num_frames_to_process]))
    estimated_poses = estimated_poses[:min_length]
    ground_truth_poses = ground_truth_poses[:min_length]

    # Compute RMSE
    rmse_values = compute_rmse_per_frame(estimated_poses, ground_truth_poses)
    for frame_id, rmse in enumerate(rmse_values):
        print(f"Frame {frame_id}: RMSE = {rmse:.4f}")

    # Visualize trajectory comparison
    visualizer.plot_trajectory_xz(estimated_poses)
    visualizer.plot_trajectory_xz(ground_truth_poses)
    plot_translation_deviations(estimated_poses, ground_truth_poses)


if __name__ == "__main__":
    main()