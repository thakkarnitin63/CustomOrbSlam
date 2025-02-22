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
from orb_slam.bundle_adjustment import BundleAdjustment
from orb_slam.map import Map
from orb_slam.map_point import MapPoint
from orb_slam.keyframe import KeyFrame

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
    pts3d_before = points3d_before.T  # Shape (N, 3)
    pts3d_after = points3d_after.T    # Shape (N, 3)

    # ✅ Ensure valid depth (positive Z values)
    valid_depths_before = pts3d_before[:, 2] > 0  # ✅ Select Z-column
    valid_pts3d_before = pts3d_before[valid_depths_before, :]  # ✅ Apply filter

    valid_depths_after = pts3d_after[:, 2] > 0  # ✅ Select Z-column
    valid_pts3d_after = pts3d_after[valid_depths_after, :]  # ✅ Apply filter

    if valid_pts3d_before.shape[0] == 0:
        print("No valid 3D points with positive depth before BA.")
        return

    if valid_pts3d_after.shape[0] == 0:
        print("No valid 3D points with positive depth after BA.")
        return

    # ✅ Compute depth statistics (Fix: Use [:, 2] instead of [2, :])
    min_depth = np.min(valid_pts3d_before[:, 2])  
    max_depth = np.max(valid_pts3d_before[:, 2])
    print(f"📌 Depth stats (Pre-BA): min = {min_depth:.2f}, max = {max_depth:.2f}")

    min_depth_after = np.min(valid_pts3d_after[:, 2])  
    max_depth_after = np.max(valid_pts3d_after[:, 2])
    print(f"📌 Depth stats (Post-BA): min = {min_depth_after:.2f}, max = {max_depth_after:.2f}")

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
      - points3d: (3 x N) numpy array of optimized 3D points after BA.
      - R: 3x3 rotation matrix of Keyframe 2.
      - t: 3x1 translation vector of Keyframe 2.
    """
    if points3d is None or points3d.shape[1] == 0:
        print("⚠️ No valid 3D points found after BA.")
        return

    # Convert shape to (N, 3)
    pts3d = points3d.T  

    # ✅ Fix: Ensure depth filtering works correctly
    valid_depths = pts3d[:, 2] > 0  # ✅ Correct slicing
    valid_pts3d = pts3d[valid_depths, :]

    if valid_pts3d.shape[0] == 0:
        print("⚠️ No valid 3D points with positive depth after BA.")
        return

    # ✅ Compute depth statistics after BA
    min_depth = np.min(valid_pts3d[:, 2])
    max_depth = np.max(valid_pts3d[:, 2])
    print(f"📌 Depth stats (Post-BA): min = {min_depth:.2f}, max = {max_depth:.2f}")

    # ✅ Create Open3D point cloud for post-BA points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_pts3d)
    pcd.paint_uniform_color([0, 1, 0])  # Green color for BA points

    # ✅ Compute a suitable frame size
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
    frame_size = extent * 0.1  # Adjust as needed

    # ✅ Create coordinate frames
    cam_frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
    cam_frame2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
    cam_frame2.transform(se3_to_transformation(R, t))  

    # ✅ Camera Position Markers
    sphere1 = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
    sphere1.paint_uniform_color([1, 0, 0])
    
    sphere2 = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
    sphere2.paint_uniform_color([0, 1, 0])
    sphere2.translate(t.flatten())

    # ✅ Visualize post-BA points
    o3d.visualization.draw_geometries(
        [pcd, cam_frame1, cam_frame2, sphere1, sphere2],
        window_name="3D Points After Bundle Adjustment (BA)",
        width=800,
        height=600,
        left=50,
        top=50
    )

def filter_depth_outliers(points3d, min_depth=3, max_depth=100):
    """ Remove 3D points that are too far away (outliers). """
    valid_mask = (points3d[2, :] > min_depth) & (points3d[2, :] < max_depth)
    return points3d[:, valid_mask]

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


def compute_rmse(estimated_poses, ground_truth_poses):
    """
    Compute the RMSE between the estimated keyframe poses and the ground truth poses.
    :param estimated_poses: List of dictionaries with {'id': frame_index, 'pose': 4x4 numpy array}.
    :param ground_truth_poses: List of 4x4 numpy arrays representing the ground truth poses.
    :return: RMSE value.
    """
    errors = []
    for est in estimated_poses:
        frame_id = est['id']  # Extract frame index
        if frame_id >= len(ground_truth_poses):
            print(f"⚠️ Warning: Frame {frame_id} not in ground truth, skipping.")
            continue
        
        gt_pose = ground_truth_poses[frame_id]
        est_translation = est['pose'][:3, 3]
        gt_translation = gt_pose[:3, 3]
        translation_error = np.linalg.norm(est_translation - gt_translation)
        errors.append(translation_error)

    if len(errors) == 0:
        print("⚠️ No valid RMSE computation due to missing data.")
        return float('inf')  # Return a large error if no valid poses exist
    
    return np.sqrt(np.mean(np.array(errors) ** 2))



def main():
    # Paths (adjust these paths as needed)
    sequence_path = '/home/nitin/NitinWs/CustomOrbSlam/data/dataset/sequences/00'
    ground_truth_file = '/home/nitin/NitinWs/CustomOrbSlam/data/dataset/poses/00.txt'

    # ✅ Load ground truth trajectory
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

    map_initializer = MapInitializer(K, feature_extractor, matcher, global_map)

    # ✅ Perform Initial Map Construction
    success = map_initializer.initialize_map(image1, image2)
    if not success:
        print("⚠️ Map initialization failed.")
        return
    
        # ✅ Verify Extracted Keypoints & Descriptors
    keyframe1 = global_map.get_keyframe(0)
    keyframe2 = global_map.get_keyframe(1)

    print(f"🔹 Keypoints in KeyFrame1: {len(keyframe1.keypoints)}")
    print(f"🔹 Keypoints in KeyFrame2: {len(keyframe2.keypoints)}")
    
    print(f"🔹 Descriptors Shape KeyFrame1: {keyframe1.descriptors.shape}")
    print(f"🔹 Descriptors Shape KeyFrame2: {keyframe2.descriptors.shape}")

    # ✅ Verify Number of 3D Map Points Created
    print(f"🔹 Total MapPoints: {len(global_map.map_points)}")

    # ✅ Verify One Example MapPoint
    if len(global_map.map_points) > 0:
        example_point = next(iter(global_map.map_points.values()))
        print(f"🔹 Sample MapPoint ID: {example_point.id}")
        print(f"🔹 Sample MapPoint Position: {example_point.position}")
        print(f"🔹 Sample MapPoint Descriptor Shape: {example_point.descriptor.shape}")

    before_ba =keyframe2.pose
    # ✅ Perform Full Bundle Adjustment (BA)
    ba = BundleAdjustment(K, iterations=15)
    print("\n🛠 Performing Bundle Adjustment (BA)...")

    # Optimize entire global map with BA
    ba.optimize_full(global_map)

    # ✅ Extract Optimized Keyframe Poses
    refined_pose1 = global_map.get_keyframe(1).pose  # Updated KeyFrame 2 pose

    print(f"📌 Pose Before BA:\n{keyframe2.pose}")
    print(f"📌 Pose After BA:\n{refined_pose1}")

    # ✅ Compute RMSE BEFORE BA
    pre_ba_rmse = compute_rmse(
        [{'id': 1, 'pose': before_ba}], ground_truth_poses
    )
    print(f"📏 RMSE Before BA: {pre_ba_rmse:.4f}")

    # ✅ Compute RMSE AFTER BA
    post_ba_rmse = compute_rmse(
        [{'id': 1, 'pose': refined_pose1}], ground_truth_poses
    )
    print(f"📏 RMSE After BA: {post_ba_rmse:.4f}")

    # ✅ Optional: Compare Map Points Before & After BA
    points3d_ba = np.array([mp.position for mp in global_map.map_points.values()]).T
    print(f"📌 Post-BA 3D points: {points3d_ba.shape}")


    # # ✅ Visualization (Optional)
    # visualize_initialization(init_results['points3d'], points3d_ba, R_ba, t_ba)

if __name__ == "__main__":
    main()
