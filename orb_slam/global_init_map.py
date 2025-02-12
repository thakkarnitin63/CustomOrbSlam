import cv2
import numpy as np

class MapInitializer:
    def __init__(self, K, feature_extractor, feature_matcher):
        """
        params:
            -K: Camera calibration matrix.
            - feature_extractor : An instance that provides an .extract(image) method.
            - matcher: An instance that provides a .match(des1, des2) method.
        """
        self.K = K
        self.extractor = feature_extractor
        self.matcher = feature_matcher

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
        return np.sum(np.where(errors<T, Gamma-errors,0))

    def initialize_map(self, img_ref, img_cur):
        """
        Attempts to initialize the map from a reference image and a current image.
        
        Returns a dictionary containing:
         - 'R': Recovered rotation matrix.
         - 't': Recovered translation vector.
         - 'points3d': Triangulated 3D map points (3 x N array).
         - 'matches': The list of matches used.
         - 'selected_model': Which model was selected ('homography' or 'fundamental').
        If initialization fails, returns None.
        """
        # -----------------------------
        # Step 1: Extract Features
        # -----------------------------
        kp_ref, des_ref = self.extractor.extract(img_ref)
        kp_cur, des_cur = self.extractor.extract(img_cur)
        if des_ref is None or des_cur is None:
            print("Feature extraction failed.")
            return None
        
        # -----------------------------
        # Step 2: Match Features
        # -----------------------------
        matches = self.matcher.match(des_ref, des_cur, kp_ref, kp_cur)
        if len(matches)<8 :
            print("Not enough matches for initialization.")
            return None
        
        # Build arrays of matched points. #Filtering the good matches from keypoints.
        pts_ref = np.array([kp_ref[m.queryIdx].pt for m in matches])
        pts_cur = np.array([kp_cur[m.trainIdx].pt for m in matches])

        # -----------------------------
        # Step 3: Compute Geometrical Models (RANSAC)
        # -----------------------------
        H, mask_H = cv2.findHomography(pts_ref, pts_cur, cv2.RANSAC, 3.0)
        F, mask_F = cv2.findFundamentalMat(pts_ref, pts_cur, cv2.FM_RANSAC, 3.0, 0.99)
        if H is None or F is None:
            print("Model estimation failed.")
            return None   


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
        if selected_model == 'homography':
            # Decompose homography to get possible motion hypotheses.
            retval, rotations, translations, normals = cv2.decomposeHomographyMat(H, self.K)
            best_inliers = 0
            best_pose = None
            for R, t in zip(rotations, translations):
                # Create projection matrices for the two views.
                P1 = self.K @ np.hstack((np.eye(3), np.zeros((3, 1))))
                P2 = self.K @ np.hstack((R, t.reshape(3, 1)))
                pts4d = cv2.triangulatePoints(P1, P2, pts_ref.T, pts_cur.T)
                pts3d = pts4d[:3] / pts4d[3]
                # Check the cheirality condition (points must be in front of both cameras).
                pts3d_cam2 = R @ pts3d + t.reshape(3, 1)
                valid = np.sum((pts3d[2, :] > 0) & (pts3d_cam2[2, :] > 0))
                if valid > best_inliers:
                    best_inliers = valid
                    best_pose = (R, t)
            if best_pose is None:
                print("No valid pose found from homography decomposition.")
                return None
            R, t = best_pose
        else:
            # For the fundamental matrix, convert to the essential matrix.
            E = self.K.T @ F @ self.K
            retval, R, t, mask_pose = cv2.recoverPose(E, pts_ref, pts_cur, self.K)
            if retval < 10:
                print("Pose recovery from fundamental matrix failed.")
                return None

        # -----------------------------
        # Final Triangulation of 3D Points
        # -----------------------------
        P1 = self.K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = self.K @ np.hstack((R, t.reshape(3, 1)))
        pts4d = cv2.triangulatePoints(P1, P2, pts_ref.T, pts_cur.T)
        pts3d = pts4d[:3] / pts4d[3]

        return {
            'R': R,
            't': t,
            'points3d': pts3d,
            'matches': matches,
            'selected_model': selected_model
        }
       
        

    


