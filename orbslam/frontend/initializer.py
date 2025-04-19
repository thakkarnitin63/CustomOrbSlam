import cv2
import numpy as np
import matplotlib.pyplot as plt
from orbslam.core.map import Map
from orbslam.core.map_point import MapPoint
from orbslam.core.keyframe import KeyFrame

class MapInitializer:
    def __init__(self, K, feature_extractor, feature_matcher, map_instance):
        """
        params:
            -K: Camera calibration matrix.
            - feature_extractor : An instance that provides an .extract(image) method.
            - matcher: An instance that provides a .match(des1, des2) method.
            - map_instance: Global Map instance.
        """
        self.K = K
        self.extractor = feature_extractor
        self.matcher = feature_matcher
        self.map = map_instance  # Global Map instance

        # Set outlier thresholds (based on X^2 test at 95% for 1 pixel std)
        self.T_H = 5.99 # Homography Mat. threshold
        self.T_F = 3.84 # Fundamental Mat. threshold

        # Gamma is set equal to the threshold.
        self.Gamma_H = self.T_H
        self.Gamma_F = self.T_F

    def compute_symmetric_transfer_error_homography(self, H, pts_ref, pts_cur):
        """
        Computes symmetric transfer errors from homography model.
        """
        # Convert to homogeneous coodinates.
        pts_ref_h = np.hstack([pts_ref, np.ones((pts_ref.shape[0], 1))]).T # shape: 3 x N (2d points)
        pts_cur_h = np.hstack([pts_cur, np.ones((pts_cur.shape[0], 1))]).T  # shape: 3 x N (2d points)

        # Map points from the reference image to current image.
        proj_pts_cur = H @ pts_ref_h
        proj_pts_cur /= proj_pts_cur[2,:] #normalize homogenous coordinates
        d2_cr = np.sum((pts_cur.T - proj_pts_cur[:2,:])**2, axis=0)

        # Map points from current image back to the reference image.
        H_inv = np.linalg.inv(H)
        proj_pts_ref = H_inv @ pts_cur_h
        proj_pts_ref /= proj_pts_ref[2,:]
        d2_rc = np.sum((pts_ref.T - proj_pts_ref[:2, :])**2, axis=0)

        return d2_cr, d2_rc
    
    def compute_symmetric_transfer_error_fundamental(self, F, pts_ref, pts_cur):
        """
        Computes symmetric epipolar transfer errors for the fundamental matrix model.
        """
        # Convert to homogeneous coordinates.
        pts_ref_h = np.hstack([pts_ref, np.ones((pts_ref.shape[0], 1))])
        pts_cur_h = np.hstack([pts_cur, np.ones((pts_cur.shape[0], 1))])

        # Epipolar lines in the current image for points from the reference image.
        l_cur = (F @ pts_ref_h.T).T  # shape: N x 3 Computing epipolar line l_cur=F*pref each element rep as line ax+by+c=0
        num = np.abs(np.sum(l_cur * pts_cur_h, axis=1)) # Compute the Distance from Current Points to Their Epipolar Lines
        den = np.sqrt(l_cur[:, 0]**2 + l_cur[:, 1]**2) + 1e-6 #normalising factor d=(ax+by+c)/(sqrt(a^2+b^2)) 
        d_cr = num / den
        d2_cr = d_cr**2

        # Epipolar lines in the reference image for points from the current image.
        l_ref = (F.T @ pts_cur_h.T).T  # shape: N x 3
        num2 = np.abs(np.sum(l_ref * pts_ref_h, axis=1))
        den2 = np.sqrt(l_ref[:, 0]**2 + l_ref[:, 1]**2) + 1e-6
        d_rc = num2 / den2
        d2_rc = d_rc**2

        return d2_cr, d2_rc
    
    def compute_score(self, errors, T, Gamma):
        """
        Computes the score contribution from a set of squared errors.
        For each error d^2, if d^2< T, add (gamma -d^2); otherwise add 0.
        """
        return np.sum(np.where(errors < T, Gamma - errors,0))

    def initialize_map(self, img_ref, img_cur):
        """
        Attempts to initialize the map from a reference image and a current image.
        
        Returns:
            - True if initialization was successful
            - False if it failed
        If initialization fails, returns None.
        """
        # -----------------------------
        # Step 1: Extract Features
        # -----------------------------
        kp_ref, des_ref = self.extractor.extract(img_ref)
        kp_cur, des_cur = self.extractor.extract(img_cur)
        if des_ref is None or des_cur is None:
            print("Feature extraction failed.")
            return False
        
        # -----------------------------
        # Step 2: Match Features
        # -----------------------------
        matches = self.matcher.match(des_ref, des_cur, kp_ref, kp_cur)
        if len(matches)<8 :
            print("Not enough matches for initialization.")
            return False
        
        
        # Build arrays of matched points. #Filtering the good matches from keypoints.
        pts_ref = np.array([kp_ref[m.queryIdx].pt for m in matches])
        pts_cur = np.array([kp_cur[m.trainIdx].pt for m in matches])

        kp_ref_filter = np.array([kp_ref[m.queryIdx] for m in matches])
        kp_cur_filter = np.array([kp_cur[m.trainIdx] for m in matches])
    
        des_ref_filter = np.array([des_ref[m.queryIdx] for m in matches])
        des_cur_filter = np.array([des_cur[m.trainIdx] for m in matches])

        # -----------------------------
        # Step 3: Compute Geometrical Models (RANSAC)
        # -----------------------------
        H, mask_H = cv2.findHomography(pts_ref, pts_cur, cv2.RANSAC, 3.0)
        F, mask_F = cv2.findFundamentalMat(pts_ref, pts_cur, cv2.FM_RANSAC, 3.0, 0.99)
        if H is None or F is None:
            print("Model estimation failed.")
            return False   


        # -----------------------------
        # Step 4: Score Both Models
        # -----------------------------
        d2_cr_H, d2_rc_H = self.compute_symmetric_transfer_error_homography(H, pts_ref, pts_cur)
        S_H = (self.compute_score(d2_cr_H, self.T_H, self.Gamma_H) +
               self.compute_score(d2_rc_H, self.T_H, self.Gamma_H))

        d2_cr_F, d2_rc_F = self.compute_symmetric_transfer_error_fundamental(F, pts_ref, pts_cur)
        S_F = (self.compute_score(d2_cr_F, self.T_F, self.Gamma_F) +
               self.compute_score(d2_rc_F, self.T_F, self.Gamma_F))


        # -----------------------------
        # Step 5: Model Selection Heuristic
        # -----------------------------
        R_H_ratio = S_H / (S_H + S_F + 1e-6)
        if R_H_ratio > 0.45:
            selected_model = 'homography'
        else:
            selected_model = 'fundamental'
        print(f"Selected model: {selected_model} (R_H: {R_H_ratio:.2f})")


        # -----------------------------
        # Step 6: Recover Motion and Triangulate
        # -----------------------------
        # Step 6: Recover Motion and Triangulate
        if selected_model == 'homography':
            retval, rotations, translations, normals = cv2.decomposeHomographyMat(H, self.K)

            best_inliers = 0
            best_pose = None
            best_reprojection_error = float("inf")

            for R, t in zip(rotations, translations):
                P1 = self.K @ np.hstack((np.eye(3), np.zeros((3, 1))))
                P2 = self.K @ np.hstack((R, t.reshape(3, 1)))

                pts4d = cv2.triangulatePoints(P1, P2, pts_ref.T, pts_cur.T)
                pts4d /= pts4d[3, :]
                pts3d = pts4d[:3, :]

                # Cheirality Check
                pts3d_cam2 = R @ pts3d + t.reshape(3, 1)
                valid = (pts3d[2, :] > 0) & (pts3d_cam2[2, :] > 0)
                num_valid = np.sum(valid)

                # Compute Reprojection Error
                projected_pts = (self.K @ P2) @ pts4d
                projected_pts /= projected_pts[2, :]
                error = np.linalg.norm(pts_cur.T - projected_pts[:2, :], axis=0).mean()

                # Pick best pose
                if num_valid > best_inliers and error < best_reprojection_error:
                    best_inliers = num_valid
                    best_pose = (R, t)
                    best_reprojection_error = error

            if best_pose is None:
                print("No valid pose found from homography decomposition.")
                return False
            R, t = best_pose

        else:
            # Fundamental Matrix Case
            U, S, Vt = np.linalg.svd(F)
            S[2] = 0  # Enforce rank-2 constraint
            F = U @ np.diag(S) @ Vt

            E = self.K.T @ F @ self.K
            retval, R, t, mask_pose = cv2.recoverPose(E, pts_ref, pts_cur, self.K)
            t /= np.linalg.norm(t)  # Normalize translation
        # -----------------------------
        # Final Triangulation and Robust Filtering
        # -----------------------------
        P1 = self.K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = self.K @ np.hstack((R, t.reshape(3, 1)))

        # ✅ More numerically stable triangulation
        pts4d = cv2.triangulatePoints(P1, P2, pts_ref.T, pts_cur.T)
        pts4d /= (pts4d[3, :] + 1e-8)  # Avoid division by zero
        pts3d = pts4d[:3, :]

        # ✅ Stronger cheirality check (ensures valid points in front of both cameras)
        pts3d_cam2 = R @ pts3d + t.reshape(3, 1)
        valid1 = pts3d[2, :] > 0  # Points should be in front of the first camera
        valid2 = pts3d_cam2[2, :] > 0  # Points should be in front of the second camera
        valid = np.logical_and(valid1, valid2)

        # ✅ Remove outliers using depth filtering
        depths = pts3d[2, valid]
        if depths.size > 0:
            max_depth_threshold = np.percentile(depths, 95)  # Remove extreme outliers
            valid = valid & (pts3d[2, :] < max_depth_threshold)

        # ✅ Apply valid mask

        pts3d_filtered = pts3d[:, valid]
        kps_ref_filtered = [kp_ref_filter[i] for i in range(len(kp_ref_filter)) if valid[i]]        
        des_ref_filtered = np.array([des_ref_filter[i] for i in range(len(des_ref_filter)) if valid[i]])

        kps_cur_filtered = [kp_cur_filter[i] for i in range(len(kp_cur_filter)) if valid[i]]
        des_cur_filtered = np.array([des_cur_filter[i] for i in range(len(des_cur_filter)) if valid[i]])




        keyframe1 = KeyFrame(0, np.eye(4), self.K, kps_ref_filtered, des_ref_filtered)
        keyframe2_pose = np.eye(4)
        keyframe2_pose[:3, :3] = R
        keyframe2_pose[:3, 3] = t.flatten()
        keyframe2 = KeyFrame(1, keyframe2_pose, self.K, kps_cur_filtered, des_cur_filtered)

        self.map.add_keyframe(keyframe1)
        self.map.add_keyframe(keyframe2)

        for keypoint_idx, points3d in enumerate(pts3d_filtered.T):

            mappoint = MapPoint(points3d, des_cur_filtered[keypoint_idx])
            map_point_id = self.map.add_map_point(mappoint)

            if map_point_id is None:
                print(f"❌ ERROR: Failed to add MapPoint {keypoint_idx}")  # Debugging line

            keyframe1.add_map_point(keypoint_idx, map_point_id)
            keyframe2.add_map_point(keypoint_idx, map_point_id)

        print(f"Map initialized successfully with 2 keyframes and {len(self.map.map_points)} map points.")
        return True
    


