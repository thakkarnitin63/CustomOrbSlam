import numpy as np
import cv2
from sklearn.cluster import DBSCAN

class PoseEstimator:
    def __init__(self, camera_matrix, dist_coeffs=None):
        """
        Initializes the pose estimator.
        
        Args:
            camera_matrix (np.ndarray): Camera intrinsic matrix.
            dist_coeffs (np.ndarray, optional): Distortion coefficients.
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros((4, 1))

    def estimate_pose(self, keypoints1, keypoints2, matches):
        """
        Estimates relative pose using the essential matrix (with homography fallback for pure rotation).
        
        Args:
            keypoints1: Keypoints from the first frame.
            keypoints2: Keypoints from the second frame.
            matches: List of matched keypoints.
            
        Returns:
            tuple: (R, t, mask) where R is a 3x3 rotation matrix, t is a 3x1 translation vector.
        """
        points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
        points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

        E, mask_E = cv2.findEssentialMat(points1, points2, self.camera_matrix,
                                           method=cv2.RANSAC, prob=0.999, threshold=1.0)
        num_inliers, R, t, mask_pose = cv2.recoverPose(E, points1, points2, self.camera_matrix)

        if np.linalg.norm(t) < 1e-6:
            print("Pure rotation detected; using homography for pose estimation.")
            H, _ = cv2.findHomography(points1, points2, method=cv2.RANSAC, ransacReprojThreshold=4.0)
            retval, Rs, ts, normals = cv2.decomposeHomographyMat(H, self.camera_matrix)
            best_R, best_t = None, None
            max_positive_depths = -1
            for i in range(retval):
                R_candidate = Rs[i]
                t_candidate = ts[i]
                if np.linalg.det(R_candidate) < 0:
                    R_candidate = -R_candidate
                    t_candidate = -t_candidate
                P1 = self.camera_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
                P2 = self.camera_matrix @ np.hstack((R_candidate, t_candidate.reshape(3, 1)))
                points_4d = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
                points_3d = points_4d[:3, :] / points_4d[3, :]
                positive_depths = np.sum(points_3d[2, :] > 0)
                if positive_depths > max_positive_depths:
                    max_positive_depths = positive_depths
                    best_R = R_candidate
                    best_t = t_candidate
            R, t = best_R, best_t

        if not np.isclose(np.linalg.det(R), 1.0, atol=1e-6):
            raise ValueError("Invalid rotation matrix computed.")

        if np.sum(mask_E) / len(matches) < 0.5:
            raise ValueError("Insufficient parallax for reliable pose estimation.")

        return R, t, mask_E
    

    def estimate_pose_pnp(self, map_points, current_keyframe, matcher=None, min_inliers=10):
        object_points = []
        image_points = []
        
        map_descriptors = []
        valid_map_points = []
        
        for mp in map_points:
            if mp.descriptors is not None:
                map_descriptors.append(mp.descriptors)
                valid_map_points.append(mp)
        
        if len(map_descriptors) == 0:
            raise ValueError("No map point descriptors available for PnP.")

        map_descriptors = np.array(map_descriptors)

        # Use KNN Matching with Ratio Test
        if matcher is None:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        knn_matches = matcher.knnMatch(map_descriptors, current_keyframe.descriptors, k=2)

        # Apply Lowe's Ratio Test
        good_matches = []
        ratio_thresh = 0.75
        for m, n in knn_matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

        # Collect correspondences.
        for m in good_matches:
            object_points.append(valid_map_points[m.queryIdx].position)
            image_points.append(current_keyframe.keypoints[m.trainIdx].pt)

        print(f"[PnP Debug] Frame {current_keyframe.id}: Found {len(object_points)} 3D-2D correspondences.")

        # ** Debugging Print Statements **
        print(f"3D Points (Object Points) Sample: {np.array(object_points[:5])}")  # Print first 5 3D points
        print(f"2D Points (Image Points) Sample: {np.array(image_points[:5])}")  # Print first 5 2D points

        if len(object_points) < min_inliers:
            raise ValueError(f"PnP failed: Not enough correspondences ({len(object_points)})")

        object_points = np.array(object_points, dtype=np.float32)
        image_points = np.array(image_points, dtype=np.float32)

        # Solve PnP with a looser RANSAC threshold
        success, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image_points,
                                                   self.camera_matrix, self.dist_coeffs,
                                                   reprojectionError=8.0,  # Loosen threshold
                                                   flags=cv2.SOLVEPNP_AP3P)  # More robust method
        if not success or inliers is None or len(inliers) < min_inliers:
            print(f"[PnP Debug] Frame {current_keyframe.id}: Only {len(inliers) if inliers is not None else 0} inliers found.")
            print(f"Reprojection Threshold: 8.0 (Consider adjusting further)")
            raise ValueError("PnP failed: No inliers found.")

        R, _ = cv2.Rodrigues(rvec)
        return R, tvec

    

    
