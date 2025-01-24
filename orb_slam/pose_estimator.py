import numpy as np
import cv2

class PoseEstimator:
    def __init__(self, camera_matrix, dist_coeffs=None):
        """
        Initializes the PoseEstimator with the camera's intrinsic parameters.
        :param camera_matrix: The intrinsic camera matrix.
        # :param dist_coeffs: The distortion coefficients of the camera.
        """
        self.camera_matrix = camera_matrix
        # self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros((4, 1))
        

    def estimate_pose(self, keypoints1, keypoints2, matches):
            """
            Estimates the relative pose between two frames given matched keypoints.
            :param keypoints1: Keypoints from the first image.
            :param keypoints2: Keypoints from the second image.
            :param matches: List of matched keypoints.
            :return: Rotation and translation vectors (R, t) representing the pose transformation.
            """
            # Extract the matched keypoints' coordinates
            points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
            points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

            # Estimate the Essential Matrix
            E, mask = cv2.findEssentialMat(points1, points2, self.camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)

            # Recover the relative pose from the Essential Matrix
            _, R, t, mask = cv2.recoverPose(E, points1, points2, self.camera_matrix)

            return R, t