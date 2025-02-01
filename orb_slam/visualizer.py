import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class Visualizer:
    def __init__(self):
        pass

    def visualize_3d_points(self, points):
        """
        Visualizes 3D points using Matplotlib.
        :param points: Nx3 array of 3D points.
        """
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='green', s=5, label='3D Points')
        ax.set_title("3D Points Visualization")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        plt.show()

    def visualize_trajectory(self, trajectory):
        """
        Visualizes the camera trajectory using Matplotlib.
        :param trajectory: Nx3 array of camera positions.
        """
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        trajectory = np.array(trajectory)
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], c='red', label='Trajectory')
        ax.scatter(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], c='blue', s=20)
        ax.set_title("Camera Trajectory Visualization")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        plt.show()

    def visualize_combined(self, trajectory, points):
        """
        Visualizes the trajectory and 3D points together.
        :param trajectory: Nx3 array of camera positions.
        :param points: Nx3 array of 3D points.
        """
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot 3D points
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='green', s=5, label='3D Points')

        # Plot trajectory
        trajectory = np.array(trajectory)
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], c='red', label='Trajectory')
        ax.scatter(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], c='blue', s=20)

        ax.set_title("Trajectory and 3D Points")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        plt.show()


    def visualize_trajectory(self, estimated_poses, ground_truth_poses):
        """
        Visualizes the estimated and ground truth trajectories in the XZ plane.
        
        Args:
            estimated_poses (list): List of 4x4 pose matrices (estimated).
            ground_truth_poses (list): List of 4x4 pose matrices (ground truth).
        """
        est_x = [pose[0, 3] for pose in estimated_poses]
        est_z = [pose[2, 3] for pose in estimated_poses]
        gt_x = [pose[0, 3] for pose in ground_truth_poses]
        gt_z = [pose[2, 3] for pose in ground_truth_poses]

        plt.figure(figsize=(10, 8))
        plt.plot(gt_x, gt_z, label="Ground Truth", color='blue', linestyle='--')
        plt.plot(est_x, est_z, label="Estimated", color='red')
        plt.title("Trajectory Comparison (X-Z Plane)")
        plt.xlabel("X Position")
        plt.ylabel("Z Position")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()

    def plot_trajectory_xz(self, poses):
        """
        Plot the trajectory in the XZ plane (top-down view).
        :param poses: List of 4x4 numpy arrays representing the poses.
        """
        x_coords = [pose[0, 3] for pose in poses]
        z_coords = [pose[2, 3] for pose in poses]

        plt.figure(figsize=(10, 6))
        plt.plot(x_coords, z_coords, marker='o', linestyle='-', color='b')
        plt.title('Ground Truth Trajectory (Top-Down View)')
        plt.xlabel('X Position')
        plt.ylabel('Z Position')
        plt.grid()
        plt.axis('equal')
        plt.show()

