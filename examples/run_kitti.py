import numpy as np
import cv2
import open3d as o3d
import os
import matplotlib.pyplot as plt
import g2o

# Import modules from our package structure
from orbslam.utils.image_loader import ImageLoader
from orbslam.frontend.feature_extractor import FeatureExtractor
from orbslam.frontend.feature_matcher import FeatureMatcher
from orbslam.frontend.initializer import MapInitializer
from orbslam.core.bundle_adjustment import BundleAdjustment
from orbslam.core.map import Map
from orbslam.core.map_point import MapPoint
from orbslam.core.keyframe import KeyFrame


def se3_to_transformation(R, t):
    """Convert rotation matrix and translation vector to 4x4 transformation matrix."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T


def visualize_initialization(points3d_before, points3d_after, R, t, frame_size=2, sphere_radius=0.2):
    """
    Visualize point clouds before and after bundle adjustment.
    
    Shows both point clouds in the same view with camera positions.
    """
    if points3d_before is None or points3d_before.shape[1] == 0:
        print("No valid 3D points found before BA.")
        return

    if points3d_after is None or points3d_after.shape[1] == 0:
        print("No valid 3D points found after BA.")
        return

    # Convert to Open3D format (N, 3)
    pts3d_before = points3d_before.T
    pts3d_after = points3d_after.T

    # Filter points with positive depth
    valid_depths_before = pts3d_before[:, 2] > 0
    valid_pts3d_before = pts3d_before[valid_depths_before, :]

    valid_depths_after = pts3d_after[:, 2] > 0
    valid_pts3d_after = pts3d_after[valid_depths_after, :]

    if valid_pts3d_before.shape[0] == 0 or valid_pts3d_after.shape[0] == 0:
        print("No valid points with positive depth.")
        return

    # Print depth statistics
    min_depth = np.min(valid_pts3d_before[:, 2])  
    max_depth = np.max(valid_pts3d_before[:, 2])
    print(f"Depth stats (Pre-BA): min = {min_depth:.2f}, max = {max_depth:.2f}")

    min_depth_after = np.min(valid_pts3d_after[:, 2])  
    max_depth_after = np.max(valid_pts3d_after[:, 2])
    print(f"Depth stats (Post-BA): min = {min_depth_after:.2f}, max = {max_depth_after:.2f}")

    # Create point clouds
    pcd_before = o3d.geometry.PointCloud()
    pcd_before.points = o3d.utility.Vector3dVector(valid_pts3d_before)
    pcd_before.paint_uniform_color([1, 0, 0])  # Red for pre-BA

    pcd_after = o3d.geometry.PointCloud()
    pcd_after.points = o3d.utility.Vector3dVector(valid_pts3d_after)
    pcd_after.paint_uniform_color([0, 1, 0])  # Green for post-BA

    # Set appropriate visualization size
    bbox = pcd_after.get_axis_aligned_bounding_box()
    extent = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
    frame_size = extent * 0.1

    # Create coordinate frames
    cam_frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size, origin=[0, 0, 0])
    
    # Position second camera frame
    T2 = se3_to_transformation(R, t)
    cam_frame2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
    cam_frame2.transform(T2)

    # Add camera position markers
    sphere1 = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
    sphere1.paint_uniform_color([1, 0, 0])
    sphere1.translate(np.array([0, 0, 0]))

    sphere2 = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
    sphere2.paint_uniform_color([0, 1, 0])
    sphere2.translate(t.flatten())

    # Show the visualization
    o3d.visualization.draw_geometries(
        [pcd_before, pcd_after, cam_frame1, cam_frame2, sphere1, sphere2],
        window_name="3D Points Before (Red) & After (Green) BA",
        width=800,
        height=600,
        left=50,
        top=50
    )


def visualize_post_ba(points3d, R, t, frame_size=2, sphere_radius=0.2):
    """Visualize point cloud after bundle adjustment with camera poses."""
    if points3d is None or points3d.shape[1] == 0:
        print("No valid 3D points found after BA.")
        return

    # Convert to Open3D format
    pts3d = points3d.T  

    # Filter points with positive depth
    valid_depths = pts3d[:, 2] > 0
    valid_pts3d = pts3d[valid_depths, :]

    if valid_pts3d.shape[0] == 0:
        print("No valid points with positive depth after BA.")
        return

    # Print depth statistics
    min_depth = np.min(valid_pts3d[:, 2])
    max_depth = np.max(valid_pts3d[:, 2])
    print(f"Depth stats (Post-BA): min = {min_depth:.2f}, max = {max_depth:.2f}")

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_pts3d)
    pcd.paint_uniform_color([0, 1, 0])  # Green for post-BA

    # Set appropriate visualization size
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
    frame_size = extent * 0.1

    # Create coordinate frames
    cam_frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
    cam_frame2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
    cam_frame2.transform(se3_to_transformation(R, t))  

    # Add camera position markers
    sphere1 = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
    sphere1.paint_uniform_color([1, 0, 0])
    
    sphere2 = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
    sphere2.paint_uniform_color([0, 1, 0])
    sphere2.translate(t.flatten())

    # Show the visualization
    o3d.visualization.draw_geometries(
        [pcd, cam_frame1, cam_frame2, sphere1, sphere2],
        window_name="3D Points After Bundle Adjustment",
        width=800,
        height=600,
        left=50,
        top=50
    )


def filter_depth_outliers(points3d, min_depth=3, max_depth=100):
    """Remove outlier points based on depth thresholds."""
    valid_mask = (points3d[2, :] > min_depth) & (points3d[2, :] < max_depth)
    return points3d[:, valid_mask]


def load_ground_truth_poses(file_path):
    """Load ground truth poses from KITTI format file."""
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            pose = np.array(values).reshape(3, 4)
            pose = np.vstack((pose, [0, 0, 0, 1]))  # Convert to homogeneous matrix
            poses.append(pose)
    return poses


def compute_rmse(estimated_poses, ground_truth_poses):
    """
    Calculate RMSE between estimated and ground truth camera poses.
    
    Uses translation component for error calculation.
    """
    errors = []
    for est in estimated_poses:
        frame_id = est['id']
        if frame_id >= len(ground_truth_poses):
            print(f"Warning: Frame {frame_id} not in ground truth, skipping.")
            continue
        
        gt_pose = ground_truth_poses[frame_id]
        est_translation = est['pose'][:3, 3]
        gt_translation = gt_pose[:3, 3]
        translation_error = np.linalg.norm(est_translation - gt_translation)
        errors.append(translation_error)

    if len(errors) == 0:
        print("No valid poses for RMSE computation.")
        return float('inf')
    
    return np.sqrt(np.mean(np.array(errors) ** 2))


def main():
    # Set file paths
    sequence_path = '/home/nitin/NitinWs/CustomOrbSlam/data/dataset/sequences/00'
    ground_truth_file = '/home/nitin/NitinWs/CustomOrbSlam/data/dataset/poses/00.txt'

    # Load ground truth
    ground_truth_poses = load_ground_truth_poses(ground_truth_file)

    # Initialize system components
    image_loader = ImageLoader(sequence_path)
    image1 = image_loader.load_image(0)
    image2 = image_loader.load_image(1)

    calibration_matrices = image_loader._load_calibration()
    K = calibration_matrices['P0']

    feature_extractor = FeatureExtractor()
    matcher = FeatureMatcher()
    global_map = Map()

    # Initialize map from first two frames
    map_initializer = MapInitializer(K, feature_extractor, matcher, global_map)
    success = map_initializer.initialize_map(image1, image2)
    if not success:
        print("Map initialization failed.")
        return
    
    # Get keyframes and print statistics
    keyframe1 = global_map.get_keyframe(0)
    keyframe2 = global_map.get_keyframe(1)

    print(f"Keypoints in KeyFrame1: {len(keyframe1.keypoints)}")
    print(f"Keypoints in KeyFrame2: {len(keyframe2.keypoints)}")
    print(f"Descriptors Shape KeyFrame1: {keyframe1.descriptors.shape}")
    print(f"Descriptors Shape KeyFrame2: {keyframe2.descriptors.shape}")
    print(f"Total MapPoints: {len(global_map.map_points)}")

    # Print sample map point info
    if len(global_map.map_points) > 0:
        example_point = next(iter(global_map.map_points.values()))
        print(f"Sample MapPoint ID: {example_point.id}")
        print(f"Sample MapPoint Position: {example_point.position}")
        print(f"Sample MapPoint Descriptor Shape: {example_point.descriptor.shape}")

    # Check for invalid map points
    for mp in global_map.map_points.values():
        if not isinstance(mp.position, np.ndarray):
            print(f"ERROR: MapPoint {mp.id} has invalid position type: {type(mp.position)}")
        elif mp.position.shape != (3,):
            print(f"ERROR: MapPoint {mp.id} has invalid shape: {mp.position.shape}")
        elif np.isnan(mp.position).any() or np.isinf(mp.position).any():
            print(f"ERROR: MapPoint {mp.id} contains NaN/Inf values: {mp.position}")

    # Save pre-BA state
    points3d_before = np.array([mp.position for mp in global_map.map_points.values()]).T
    print(f"Pre-BA 3D points: {points3d_before.shape}")
    
    before_ba_pose = keyframe2.pose.copy()
    before_ba_R = before_ba_pose[:3, :3]
    before_ba_t = before_ba_pose[:3, 3]
    
    # Perform Bundle Adjustment
    ba = BundleAdjustment(K, iterations=15)
    print("\nPerforming Bundle Adjustment...")
    ba.optimize_full(global_map)

    # Extract post-BA state
    refined_pose = global_map.get_keyframe(1).pose
    after_ba_R = refined_pose[:3, :3]
    after_ba_t = refined_pose[:3, 3]

    # Print pose information
    print(f"Pose Before BA:\n{before_ba_pose}")
    print(f"Pose After BA:\n{refined_pose}")

    # Calculate and print RMSE
    pre_ba_rmse = compute_rmse(
        [{'id': 1, 'pose': before_ba_pose}], ground_truth_poses
    )
    print(f"RMSE of Translation Vector Before BA: {pre_ba_rmse:.4f}")

    post_ba_rmse = compute_rmse(
        [{'id': 1, 'pose': refined_pose}], ground_truth_poses
    )
    print(f"RMSE After Translation Vector BA: {post_ba_rmse:.4f}")

    # Get post-BA points
    points3d_after = np.array([mp.position for mp in global_map.map_points.values()]).T
    print(f"Post-BA 3D points: {points3d_after.shape}")

    # Print visualization guide
    print("\nVISUALIZATION LEGEND:")
    print("- RED points: Before Bundle Adjustment")
    print("- GREEN points: After Bundle Adjustment")
    print("- RED sphere: Camera 1 position")
    print("- GREEN sphere: Camera 2 position")
    
    # Show visualization
    print("\nOpening 3D visualization...")
    try:
        visualize_initialization(points3d_before, points3d_after, after_ba_R, after_ba_t)
    except Exception as e:
        print(f"Visualization error: {e}")
        print("Try running in a different environment with display support.")


if __name__ == "__main__":
    main()