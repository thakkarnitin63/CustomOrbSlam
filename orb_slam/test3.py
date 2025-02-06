import numpy as np
import cv2
import matplotlib.pyplot as plt

from orb_slam.image_loader import ImageLoader
from orb_slam.feature_extractor import FeatureExtractor
from orb_slam.feature_matcher import FeatureMatcher
from orb_slam.pose_estimator import PoseEstimator
from orb_slam.visualizer import Visualizer
from orb_slam.keyframe_manager import KeyframeManager
from orb_slam.mapping import SparseMapping, KeyFrame  # MapPoint is used internally in SparseMapping

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
    feature_extractor = FeatureExtractor()
    feature_matcher = FeatureMatcher(ratio_thresh=0.75)
    calibration_matrices = image_loader._load_calibration()
    K = calibration_matrices.get('P0')
    pose_estimator = PoseEstimator(K)
    keyframe_manager = KeyframeManager()
    sparse_mapping = SparseMapping(K)
    visualizer = Visualizer()

    # Load ground truth poses (for trajectory visualization)
    ground_truth_poses = load_ground_truth_poses(ground_truth_file)

    # --- Initialization (Frame 0) ---
    first_image = image_loader.load_image(0)
    keypoints1, descriptors1 = feature_extractor.extract(first_image)
    prev_keypoints, prev_descriptors = keypoints1, descriptors1
    prev_pose = np.eye(4)
    estimated_poses = [prev_pose]

    # Create and add the initial keyframe.
    initial_kf = KeyFrame(id=0, pose=prev_pose, keypoints=keypoints1, descriptors=descriptors1)
    keyframe_manager.add_keyframe(initial_kf)
    sparse_mapping.add_keyframe(initial_kf)
    print("Added initial keyframe (frame 0).")

    num_frames = 500
    for frame_id in range(1, num_frames):
        current_image = image_loader.load_image(frame_id)
        keypoints, descriptors = feature_extractor.extract(current_image)

        # Feature matching between previous and current frame.
        matches = feature_matcher.match(prev_descriptors, descriptors)
        filtered_matches = feature_matcher.filter_matches(matches, prev_keypoints, keypoints)

        try:
            # --- Pose Estimation (Tracking) ---
            # Always use essential matrix–based pose estimation here.
            R_em, t_em, _ = pose_estimator.estimate_pose(prev_keypoints, keypoints, filtered_matches)
            T_rel = np.eye(4)
            T_rel[:3, :3] = R_em
            T_rel[:3, 3] = t_em.flatten()
            # According to our convention, update the global pose using inverse composition.
            current_pose = prev_pose @ np.linalg.inv(T_rel)
            estimated_poses.append(current_pose)
            print(f"Frame {frame_id}: Pose estimated using essential matrix.")

            # --- Keyframe Insertion and Mapping ---
            last_kf = keyframe_manager.get_last_keyframe()
            if last_kf is not None and keyframe_manager.is_new_keyframe(current_pose, last_kf.pose, len(filtered_matches)):
                # Create and add a new keyframe.
                new_kf = KeyFrame(id=frame_id, pose=current_pose, keypoints=keypoints, descriptors=descriptors)
                keyframe_manager.add_keyframe(new_kf)
                sparse_mapping.add_keyframe(new_kf)
                print(f"Frame {frame_id}: New keyframe inserted.")

                # --- Triangulation ---
                # Re-match features between the last keyframe and the new keyframe.
                kf_matches = feature_matcher.match(last_kf.descriptors, new_kf.descriptors)
                kf_filtered_matches = feature_matcher.filter_matches(kf_matches, last_kf.keypoints, new_kf.keypoints)
                # Compute the relative transformation between keyframes.
                T_rel_kf = np.linalg.inv(last_kf.pose) @ new_kf.pose
                R_rel = T_rel_kf[:3, :3]
                t_rel = T_rel_kf[:3, 3].reshape(3, 1)
                # Triangulate new map points.
                new_map_points = sparse_mapping.triangulate_points(last_kf, new_kf, kf_filtered_matches, R_rel, t_rel)
                print(f"Frame {frame_id}: Triangulated {len(new_map_points)} new map points.")

            prev_pose = current_pose

        except ValueError as e:
            print(f"Frame {frame_id}: Pose estimation failed: {e}")

        # Update previous frame data.
        prev_keypoints, prev_descriptors = keypoints, descriptors

    visualizer.visualize_trajectory(estimated_poses, ground_truth_poses[:num_frames])

if __name__ == "__main__":
    main()
