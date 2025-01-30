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
def compute_rmse(estimated_keyframes, ground_truth_poses):
    """
    Compute the RMSE between the estimated keyframe poses and the ground truth poses.
    :param estimated_keyframes: List of KeyFrame objects.
    :param ground_truth_poses: List of 4x4 numpy arrays representing the ground truth poses.
    :return: RMSE value.
    """
    errors = []
    for keyframe in estimated_keyframes:
        # Use `id` as the index for ground truth
        gt_pose = ground_truth_poses[keyframe.id]
        est_translation = keyframe.pose[:3, 3]
        gt_translation = gt_pose[:3, 3]
        translation_error = np.linalg.norm(est_translation - gt_translation)
        errors.append(translation_error)
    rmse = np.sqrt(np.mean(np.array(errors) ** 2))
    return rmse


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
    sparse_mapping = SparseMapping(K)
    visualizer = Visualizer()

    # Load ground truth poses
    ground_truth_poses = load_ground_truth_poses(ground_truth_file)

    # Initialize tracking variables
    keypoints1, descriptors1 = feature_extractor.extract(image_loader.load_image(0))
    prev_keypoints, prev_descriptors = keypoints1, descriptors1
    prev_pose = np.eye(4)  # Identity pose
    estimated_poses = [prev_pose]


    # Add the initial pose as a keyframe
    initial_keyframe = KeyFrame(id=0, pose=prev_pose, keypoints=keypoints1, descriptors=descriptors1)
    keyframe_manager.add_keyframe(initial_keyframe)
    sparse_mapping.add_keyframe(initial_keyframe)
    print(f"Added initial keyframe at frame 0.")

    # Process frames
    num_frames_to_process = 25
    for frame_id in range(1, num_frames_to_process):
        current_image = image_loader.load_image(frame_id)

        # Extract features
        keypoints, descriptors = feature_extractor.extract(current_image)

        # Match features
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
                    sparse_mapping.add_keyframe(new_keyframe)

                    # Triangulate points between last two keyframes
                    if len(keyframe_manager.keyframes) > 1:
                        keyframe1 = keyframe_manager.keyframes[-2]
                        keyframe2 = keyframe_manager.keyframes[-1]
                        new_map_points = sparse_mapping.triangulate_points(
                            keyframe1, keyframe2, filtered_matches, R, t
                        )
                        # print(f"Triangulated {len(new_map_points)} map points.")

            prev_pose = current_pose
            # Print tracking stability details
            print(f"Frame {frame_id}: Matches={len(filtered_matches)}, KeyFrames={len(keyframe_manager.keyframes)}, MapPoints={len(sparse_mapping.map_points)}")

        except ValueError as e:
                print(f"Pose estimation failed at frame {frame_id}: {e}")

        # Update previous frame data
        prev_keypoints, prev_descriptors = keypoints, descriptors

    # Visualize the trajectory
    visualizer.plot_trajectory_xz(estimated_poses)
    # print(f"Number of keyframes: {len(keyframe_manager.keyframes)}")
    # print(f"Total number of 3D map points stored: {len(sparse_mapping.map_points)}")
    # for keyframe in keyframe_manager.keyframes:
        # print(f"KeyFrame {keyframe.id} observes {keyframe.num_observations()} map points.")
    rmse = compute_rmse(keyframe_manager.keyframes, ground_truth_poses)
    print(f"RMSE of the trajectory: {rmse:.4f}")


if __name__ == "__main__":
    main()