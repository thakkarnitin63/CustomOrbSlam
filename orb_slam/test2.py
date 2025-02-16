import numpy as np
import cv2
import open3d as o3d
import os

import matplotlib.pyplot as plt
import g2o

from orb_slam.image_loader import ImageLoader
from orb_slam.feature_extractor import FeatureExtractor
from orb_slam.feature_matcher import FeatureMatcher
from orb_slam.global_init_map import MapInitializer
from orb_slam.bundle_adjustment import BundleAdjuster

def se3_to_transformation(R, t):
    """
    Converts a rotation matrix R and translation vector t into a 4x4 homogeneous transformation.
    """
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()  # Ensure t is 1D
    return T

def visualize_initialization(points3d_before, points3d_after, R, t, frame_size=2, sphere_radius=0.2):
    """
    Visualize the 3D point cloud before and after BA, along with camera poses using Open3D.

    Parameters:
      - points3d_before: (3 x N) numpy array of triangulated 3D points before BA.
      - points3d_after: (3 x N) numpy array of triangulated 3D points after BA.
      - R: 3x3 rotation matrix (Keyframe 2 relative to Keyframe 1).
      - t: 3x1 translation vector (Keyframe 2 relative to Keyframe 1).
      - frame_size: Size of coordinate frame axes.
      - sphere_radius: Radius of spheres to mark camera centers.
    """
    if points3d_before is None or points3d_before.shape[1] == 0:
        print("No valid 3D points found before BA.")
        return

    if points3d_after is None or points3d_after.shape[1] == 0:
        print("No valid 3D points found after BA.")
        return

    # Convert to (N, 3) format for Open3D
    pts3d_before = points3d_before.T
    pts3d_after = points3d_after.T

    # Ensure valid depth (positive Z values)
    valid_depths_before = pts3d_before[:, 2] > 0
    valid_pts3d_before = pts3d_before[valid_depths_before, :]

    valid_depths_after = pts3d_after[:, 2] > 0
    valid_pts3d_after = pts3d_after[valid_depths_after, :]

    if valid_pts3d_before.shape[0] == 0:
        print("No valid 3D points with positive depth before BA.")
        return

    if valid_pts3d_after.shape[0] == 0:
        print("No valid 3D points with positive depth after BA.")
        return

    # Compute depth statistics
    min_depth = np.min(valid_pts3d_after[:, 2])
    max_depth = np.max(valid_pts3d_after[:, 2])
    print(f"Depth stats after BA: min = {min_depth:.2f}, max = {max_depth:.2f}")

    # Create Open3D point clouds
    pcd_before = o3d.geometry.PointCloud()
    pcd_before.points = o3d.utility.Vector3dVector(valid_pts3d_before)
    pcd_before.paint_uniform_color([1, 0, 0])  # Red before BA

    pcd_after = o3d.geometry.PointCloud()
    pcd_after.points = o3d.utility.Vector3dVector(valid_pts3d_after)
    pcd_after.paint_uniform_color([0, 1, 0])  # Green after BA

    # Compute a suitable frame size
    bbox = pcd_after.get_axis_aligned_bounding_box()
    extent = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
    frame_size = extent * 0.1  # Adjust as needed

    # Create coordinate frame for Keyframe 1 (origin)
    cam_frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size, origin=[0, 0, 0])

    # Create coordinate frame for Keyframe 2 using recovered pose
    T2 = se3_to_transformation(R, t)
    cam_frame2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
    cam_frame2.transform(T2)  # Apply transformation to move frame to second camera

    # Spheres at camera centers
    sphere1 = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
    sphere1.paint_uniform_color([1, 0, 0])  # Red for camera 1
    sphere1.translate(np.array([0, 0, 0]))

    sphere2 = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
    sphere2.paint_uniform_color([0, 1, 0])  # Green for camera 2
    sphere2.translate(t.flatten())

    # Visualize before and after BA
    o3d.visualization.draw_geometries(
        [pcd_before, pcd_after, cam_frame1, cam_frame2, sphere1, sphere2],
        window_name="3D Points Before (Red) & After (Green) BA",
        width=800,
        height=600,
        left=50,
        top=50
    )

def visualize_post_ba(points3d, R, t, frame_size=2, sphere_radius=0.2):
    """
    Visualize the 3D point cloud **after** Bundle Adjustment using Open3D.

    Parameters:
      - points3d: (3 x N) numpy array of 3D points after BA.
      - R: 3x3 rotation matrix (Keyframe 2 relative to Keyframe 1).
      - t: 3x1 translation vector (Keyframe 2 relative to Keyframe 1).
      - frame_size: Size of coordinate frame axes.
      - sphere_radius: Radius of spheres to mark camera centers.
    """
    if points3d is None or points3d.shape[1] == 0:
        print("No valid 3D points found after BA.")
        return

    # Convert to Open3D format
    pts3d = points3d.T  

    # Ensure valid depth (positive Z values)
    valid_depths = pts3d[:, 2] > 0
    valid_pts3d = pts3d[valid_depths, :]

    if valid_pts3d.shape[0] == 0:
        print("No valid 3D points with positive depth after BA.")
        return

    # Compute depth statistics
    min_depth = np.min(valid_pts3d[:, 2])
    max_depth = np.max(valid_pts3d[:, 2])
    print(f"📌 Depth stats (Post-BA): min = {min_depth:.2f}, max = {max_depth:.2f}")

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_pts3d)
    pcd.paint_uniform_color([0, 1, 0])  # Green color for post-BA points

    # Compute a suitable frame size
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
    frame_size = extent * 0.1  # Adjust as needed

    # Create coordinate frame for Keyframe 1 (origin)
    cam_frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size, origin=[0, 0, 0])

    # Create coordinate frame for Keyframe 2 using recovered pose
    T2 = se3_to_transformation(R, t)
    cam_frame2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
    cam_frame2.transform(T2)  # Apply transformation to move frame to second camera

    # Spheres at camera centers
    sphere1 = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
    sphere1.paint_uniform_color([1, 0, 0])  # Red for camera 1
    sphere1.translate(np.array([0, 0, 0]))

    sphere2 = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
    sphere2.paint_uniform_color([0, 1, 0])  # Green for camera 2
    sphere2.translate(t.flatten())

    # Visualize post-BA points only
    o3d.visualization.draw_geometries(
        [pcd, cam_frame1, cam_frame2, sphere1, sphere2],
        window_name="3D Points After Bundle Adjustment (BA)",
        width=800,
        height=600,
        left=50,
        top=50
    )
def main():
    # Paths (adjust these paths as needed)
    sequence_path = '/home/nitin/NitinWs/CustomOrbSlam/data/dataset/sequences/00'
    ground_truth_file = '/home/nitin/NitinWs/CustomOrbSlam/data/dataset/poses/00.txt'

    # Initialize system components
    image_loader = ImageLoader(sequence_path)
    image1 = image_loader.load_image(0)
    image2 = image_loader.load_image(1)

    calibration_matrices = image_loader._load_calibration()
    K = calibration_matrices['P0']

    feature_extractor = FeatureExtractor()
    matcher = FeatureMatcher()

    map_initializer = MapInitializer(K, feature_extractor, matcher)
    ba = BundleAdjuster(K, max_iterations=100)

    # Attempt map initialization.
    init_results = map_initializer.initialize_map(image1, image2)

    if init_results is not None:
        print(f"📌 Pre-BA 3D points: {init_results['points3d'].shape}")
        min_depth = np.min(init_results['points3d'][:, 2])
        max_depth = np.max(init_results['points3d'][:, 2])
        print(f"�� Depth stats (Pre-BA): min = {min_depth:.2f}, max = {max_depth:.2f}")

        # Run Bundle Adjustment
        # pose_ba0, pose_ba1, points3d_ba = ba.optimize(init_results)

        # print(f"📌 Post-BA 3D points: {points3d_ba.shape}")

        # Visualize Before vs After Bundle Adjustment
        # visualize_initialization(init_results['points3d'], points3d_ba, pose_ba1.rotation().matrix(), pose_ba1.translation())
        # visualize_post_ba(points3d_ba, pose_ba1.rotation().matrix(), pose_ba1.translation())


if __name__ == "__main__":
    main()
