import numpy as np
from orb_slam.image_loader import ImageLoader
from orb_slam.feature_extractor import FeatureExtractor
from orb_slam.feature_matcher import FeatureMatcher
from orb_slam.pose_estimator import PoseEstimator
from orb_slam.visualizer import Visualizer
from orb_slam.keyframe_manager import KeyframeManager
from orb_slam.mapping import SparseMapping, MapPoint, KeyFrame
import matplotlib.pyplot as plt



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



def main():
    # Paths
    sequence_path = '/home/nitin/NitinWs/CustomOrbSlam/data/dataset/sequences/00'
    ground_truth_file = '/home/nitin/NitinWs/CustomOrbSlam/data/dataset/poses/00.txt'

    # Initialize system components
    image_loader = ImageLoader(sequence_path)
    feature_extractor = FeatureExtractor()
    feature_matcher = FeatureMatcher(ratio_thresh=0.75)
    calibration_matrices = image_loader._load_calibration()
    K = calibration_matrices.get('P0')
    pose_estimator = PoseEstimator(K)
    keyframe_manager = KeyframeManager()
    visualizer = Visualizer()

    # Load ground truth poses
    ground_truth_poses = load_ground_truth_poses(ground_truth_file)

    # Initialize tracking variables
    prev_keypoints, prev_descriptors = None, None
    prev_pose = np.eye(4)  # Identity pose
    estimated_poses = [prev_pose]

    # Add the initial pose as a keyframe
    initial_keyframe = KeyFrame(id=0, pose=prev_pose, keypoints=None, descriptors=None)
    keyframe_manager.add_keyframe(initial_keyframe)
    print(f"Added initial keyframe at frame 0.")

    # Process frames
    num_frames_to_process = 1000
    for frame_id in range(1, num_frames_to_process):
        current_image = image_loader.load_image(frame_id)

        # Extract features
        keypoints, descriptors = feature_extractor.extract(current_image)

        # Match features
        if prev_descriptors is not None:
            matches = feature_matcher.match(prev_descriptors, descriptors)
            filtered_matches = feature_matcher.filter_matches(matches, prev_keypoints, keypoints)

            # Estimate pose
            try:
                R, t, _ = pose_estimator.estimate_pose(prev_keypoints, keypoints, filtered_matches)
                current_pose = np.eye(4)
                current_pose[:3, :3] = np.dot(R, prev_pose[:3, :3])
                current_pose[:3, 3] = prev_pose[:3, :3].dot(t).flatten() + prev_pose[:3, 3]

                estimated_poses.append(current_pose)

                # Check if the current frame should be a keyframe
                last_keyframe = keyframe_manager.get_last_keyframe()
                if last_keyframe is not None:
                    if keyframe_manager.is_new_keyframe(current_pose, last_keyframe.pose, len(filtered_matches)):
                        new_keyframe = KeyFrame(id=frame_id, pose=current_pose, keypoints=keypoints, descriptors=descriptors)
                        keyframe_manager.add_keyframe(new_keyframe)
                        print(f"Added keyframe at frame {frame_id}.")

                prev_pose = current_pose
            except ValueError as e:
                print(f"Pose estimation failed at frame {frame_id}: {e}")

        # Update previous frame data
        prev_keypoints, prev_descriptors = keypoints, descriptors

    # Visualize the trajectory
    visualizer.plot_trajectory_xz(estimated_poses)
    print(f"Number of keyframes: {len(keyframe_manager.keyframes)}")


if __name__ == "__main__":
    main()