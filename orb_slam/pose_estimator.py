import numpy as np
import cv2
from sklearn.cluster import DBSCAN

class PoseEstimator:
    def __init__(self, camera_matrix, dist_coeffs=None):
        """
        Initializes the PoseEstimator with the camera's intrinsic parameters.
        :param camera_matrix: The intrinsic camera matrix.
        :param dist_coeffs: The distortion coefficients of the camera (optional).
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros((4, 1))

    def estimate_pose(self, keypoints1, keypoints2, matches):
        """
        Estimates the relative pose between two frames given matched keypoints.
        Handles pure rotation cases using the homography matrix.
        """
        # Extract the matched keypoints' coordinates
        points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
        points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

        # Compute the Essential matrix using RANSAC
        E, mask_E = cv2.findEssentialMat(
            points1, points2, self.camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0
        )

        # Recover the relative pose from the Essential matrix
        num_inliers, R, t, mask_pose = cv2.recoverPose(E, points1, points2, self.camera_matrix)

        # Check for pure rotation (translation norm is near zero)
        if np.linalg.norm(t) < 1e-6:
            print("Detected pure rotation frame. Using Homography for pose estimation.")

            # Compute the homography matrix
            H, mask_H = cv2.findHomography(points1, points2, method=cv2.RANSAC, ransacReprojThreshold=4.0)

            # Decompose the homography to extract possible rotations and translations
            retval, Rs, ts, normals = cv2.decomposeHomographyMat(H, self.camera_matrix)

            # Initialize variables to store the best solution
            best_R = None
            best_t = None
            max_positive_depths = -1

            # Evaluate each decomposition
            for i in range(retval):
                R_candidate = Rs[i]
                t_candidate = ts[i]

                # Ensure the rotation matrix has a determinant of +1
                if np.linalg.det(R_candidate) < 0:
                    R_candidate = -R_candidate
                    t_candidate = -t_candidate

                # Perform cheirality check (count points with positive depth)
                P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
                P2 = np.hstack((R_candidate, t_candidate.reshape(3, 1)))
                P1 = self.camera_matrix @ P1
                P2 = self.camera_matrix @ P2

                # Triangulate points
                points_4d_hom = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
                points_4d = points_4d_hom[:3, :] / points_4d_hom[3, :]

                # Count positive depths
                positive_depths = np.sum(points_4d[2, :] > 0)

                # Update the best solution if this one has more points with positive depth
                if positive_depths > max_positive_depths:
                    max_positive_depths = positive_depths
                    best_R = R_candidate
                    best_t = t_candidate

            # Use the best decomposition
            R = best_R
            t = best_t

        # Validate rotation matrix determinant
        if not np.isclose(np.linalg.det(R), 1.0, atol=1e-6):
            raise ValueError("Invalid rotation matrix detected. Check the pose computation.")

        # Ensure parallax and valid matches
        valid_matches = np.sum(mask_E)
        if valid_matches / len(matches) < 0.5:
            raise ValueError("Insufficient parallax for reliable pose estimation.")

        return R, t, mask_E

    

    def triangulate_points(self, keypoints1, keypoints2, matches, R, t, dbscan_eps=7, dbscan_min_samples=5):
        """
        Triangulates 3D points from matched keypoints and relative pose.
        Applies DBSCAN clustering to remove outliers.
        returns filtered 2D points after clustering to remove outliers.
        :param keypoints1: Keypoints from the first image.
        :param keypoints2: Keypoints from the second image.
        :param matches: List of matched keypoints.
        :param R: Rotation matrix from the first to the second camera.
        :param t: Translation vector from the first to the second camera.
        :param dbscan_eps: Maximum distance between samples for them to be considered as in the same neighborhood.
        :param dbscan_min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
        :return: Filtered 3D points. corresponding 2d points of both images.
        """
        # Extract the matched keypoints' coordinates
        points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
        points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

        # Compute projection matrices
        P1 = np.dot(self.camera_matrix, np.hstack((np.eye(3), np.zeros((3, 1)))))
        P2 = np.dot(self.camera_matrix, np.hstack((R, t)))

        # Triangulate points
        points_4d = cv2.triangulatePoints(P1, P2, points1.T, points2.T)

        # Convert to inhomogeneous coordinates
        points_3d = points_4d[:3, :] / points_4d[3, :]
        points_3d = points_3d.T  # Transpose for easier access

        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(points_3d)
        labels = clustering.labels_

        # Filter out noise points (label == -1)
        valid_indices = np.where(labels != -1)[0]
        valid_points = points_3d[valid_indices]

        # Retrieve corresponding 2D points for valid 3D points
        valid_2d_points1 = points1[valid_indices]
        valid_2d_points2 = points2[valid_indices]

        return valid_points, valid_2d_points1, valid_2d_points2
    
    
