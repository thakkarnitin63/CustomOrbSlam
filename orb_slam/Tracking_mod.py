import numpy as np
import cv2
from orb_slam.feature_extractor import FeatureExtractor
from orb_slam.feature_matcher import FeatureMatcher
from orb_slam.bundle_adjustment import BundleAdjustment
from orb_slam.bow_database import BoWDatabase
from orb_slam.map import Map
from orb_slam.map_point import MapPoint
from orb_slam.keyframe import KeyFrame

class Tracking:
    def __init__(self, camera_intrinsics, map_instance, bow_database):
        """
        ORB-SLAM Tracking Module
        
        Assumes the map is already initialized by a separate module.
        
        :param camera_intrinsics: Intrinsic matrix K (3x3 numpy array)
        :param map_instance: Already-initialized instance of the Map class that contains keyframes and map points.
        :param bow_database: Instance of BoWDatabase for relocalization and loop closure.
        """
        # Camera parameters
        self.K = camera_intrinsics.copy()  # Camera intrinsic matrix
        
        # Map and database (map is assumed to be already initialized)
        self.map = map_instance              # Global Map
        self.bow_database = bow_database     # Bag of Words Database
        
        # Core modules: feature extraction, matching, and bundle adjustment.
        self.feature_extractor = FeatureExtractor()
        self.feature_matcher = FeatureMatcher()
        self.bundle_adjustment = BundleAdjustment(self.K)
        
        # Motion model:
        # We expect the map to have been initialized with at least two keyframes.
        # These keyframes are used to compute the relative motion (velocity).
        self.motion_model = {
            "last_keyframe": self.map.get_keyframe(1),      # Assumes keyframe 1 exists.
            "second_last_keyframe": self.map.get_keyframe(0)  # Assumes keyframe 0 exists.
        }
        self.current_keypoints = None
        self.current_descriptors = None
        
        # Current frame state:
        self.current_frame = None
        self.current_pose = None             # 4x4 transformation matrix (world-to-camera)
        self.tracked_map_points = {}         # Mapping: current frame keypoint index -> MapPoint ID
        # After global initialization is complete
        if self.map.get_keyframe(1) is not None:
            self.last_frame = self.map.get_keyframe(1)
        else:
            self.last_frame = None  # fallback, though this case shouldn't happen if initialization succeeded               
                
        # Additional state variables for robust tracking:
        self.mVelocity = np.eye(4, dtype=np.float32)  # Initial velocity (identity)
        self.mState = "WORKING"                       # Tracking state (could be "NOT_INITIALIZED", "LOST", etc.)
        self.mMinFrames = 0                           # Minimum frames to wait before inserting a new keyframe
        self.mMaxFrames = 20                          # Maximum frames allowed before considering keyframe insertion
        
        print("Tracking module initialized:")
        print(f" - Camera intrinsics:\n{self.K}")
        print(f" - Map has {len(self.map.keyframes)} keyframes")
        print(f" - Initial tracking state: {self.mState}")

    def process_frame(self, frame, frame_id):
        """
        Process an incoming frame for tracking.
        
        Steps:
        - Extract features from the input grayscale image.
        - Attempt tracking using a cascade of methods:
                1. track_with_motion_model
                2. If that fails, track_with_map_points
                3. If that fails, global_relocalization
        - If tracking succeeds, refine the pose using local map tracking and check keyframe insertion.
        - If tracking is successful, create a new KeyFrame from the current frame and update the motion model.
        - Update self.last_frame with the newly created KeyFrame (not the raw image).
        
        :param frame: Input grayscale image.
        :param frame_id: ID of the frame in the sequence.
        """
        # (Optionally) store the raw image in current_frame for visualization, etc.
        self.current_frame = frame
        
        # Extract features from the current frame.
        keypoints, descriptors = self.feature_extractor.extract(frame)
        
        self.current_keypoints = keypoints
        self.current_descriptors = descriptors
        # Attempt to track using the cascade of methods.
        success = self.track_with_motion_model(keypoints, descriptors)
        if not success:
            print("Motion model tracking failed; trying wider map point search.")
            success = self.track_with_map_points(keypoints, descriptors)
        if not success:
            print("Wider map point search failed; trying global relocalization.")
            success = self.global_relocalization(keypoints, descriptors)
        
        if success:
            # Refine pose using local map tracking and check new keyframe conditions.
            self.track_local_map()
            self.check_new_keyframe(frame_id, keypoints, descriptors)
            self.mState = "WORKING"
        else:
            print("Tracking lost!")
            self.mState = "LOST"
        
        # Update motion model and map only if tracking was successful.
        if self.current_pose is not None and self.mState == "WORKING":
            # Create a new keyframe from the current frame's extracted features.
            new_keyframe = KeyFrame(frame_id, self.current_pose, self.K, keypoints, descriptors)
            self.map.add_keyframe(new_keyframe)
            # Update the motion model by shifting keyframes.
            self.motion_model["second_last_keyframe"] = self.motion_model["last_keyframe"]
            self.motion_model["last_keyframe"] = new_keyframe
            # IMPORTANT: Set last_frame to the new keyframe (not the raw image)
            self.last_frame = new_keyframe
        else:
            print("Not updating keyframes because tracking is lost.")
            # Optionally, you might choose to keep the previous last_frame if tracking is lost.

        
    def _projection_match_search(self, keypoints, descriptors, windowSize, TH_HIGH, MIN_MATCHES):
        """
        Helper function that performs batch projection of map points from the last keyframe,
        builds a KD-tree for current frame keypoints, and searches for correspondences.
        
        :param keypoints: List of keypoints from the current frame.
        :param descriptors: Corresponding descriptors from the current frame.
        :param windowSize: Search window size (in pixels).
        :param TH_HIGH: Maximum acceptable Hamming distance.
        :param MIN_MATCHES: Minimum number of matches required.
        :return: A dictionary mapping keypoint indices (from last keyframe) to current frame keypoint indices,
                or None if not enough matches are found.
        """
        # Convert current pose to rotation vector for projection.
        R = self.current_pose[:3, :3]
        t = self.current_pose[:3, 3]
        rvec, _ = cv2.Rodrigues(R)
        
        # Get map point indices from the last keyframe's stored associations.
        mp_keys = list(self.motion_model["last_keyframe"].map_points.keys())
        if not mp_keys:
            return None

        mp_positions = []
        mp_descs = []
        mp_map_idx = []  # These indices correspond to keypoints in the last keyframe.
        for idx in mp_keys:
            map_point_id = self.motion_model["last_keyframe"].map_points[idx]
            mp = self.map.get_map_point(map_point_id)
            if mp is None:
                continue
            mp_positions.append(mp.position)
            mp_descs.append(mp.descriptor)
            mp_map_idx.append(idx)
        if not mp_positions:
            return None

        mp_positions = np.array(mp_positions, dtype=np.float32)
        mp_descs = np.array(mp_descs)

        # Batch-project all 3D map point positions into the image.
        imagePoints, _ = cv2.projectPoints(mp_positions, rvec, t, self.K, None)
        projected_pts = imagePoints.reshape(-1, 2)

        # Build a KD-tree for current frame keypoints.
        current_kp_positions = np.array([kp.pt for kp in keypoints], dtype=np.float32) # New image pts 
        from scipy.spatial import cKDTree
        kd_tree = cKDTree(current_kp_positions)

        # Perform matching for each projected map point.
        projected_matches = {}
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        h_img, w_img = self.current_frame.shape  # Assumes grayscale image.
        mfNNratio = 0.9

        for i, proj in enumerate(projected_pts):
            x, y = proj
            if x < 0 or x >= w_img or y < 0 or y >= h_img:
                continue
            candidate_indices = kd_tree.query_ball_point(proj, windowSize) # query on new image tree
            if not candidate_indices:
                continue
            candidate_descs = np.array([descriptors[idx] for idx in candidate_indices]) # getting new image's descrip
            map_desc = np.array([mp_descs[i]])
            matches = bf.knnMatch(map_desc, candidate_descs, k=2)
            if not matches or len(matches[0]) == 0:
                continue
            best_matches = matches[0]
            if len(best_matches) < 2:
                if best_matches[0].distance < TH_HIGH:
                    current_idx = candidate_indices[best_matches[0].trainIdx] # just a idea to check for queryIdx
                    map_point_id = self.motion_model["last_keyframe"].map_points[mp_map_idx[i]]
                    projected_matches[current_idx] =  map_point_id
            else:
                if best_matches[0].distance < mfNNratio * best_matches[1].distance and best_matches[0].distance < TH_HIGH:
                    current_idx = candidate_indices[best_matches[0].trainIdx]
                    map_point_id = self.motion_model["last_keyframe"].map_points[mp_map_idx[i]]
                    projected_matches[current_idx] = map_point_id


        if len(projected_matches) < MIN_MATCHES:
            print(f"Projection match search: Not enough matches ({len(projected_matches)})")
            return None
        return projected_matches

    def track_with_motion_model(self, keypoints, descriptors):
        """
        Attempts to track using a narrow, guided search based on the motion model.
        
        Uses the last frame's pose (updated with precomputed velocity) as the initial guess,
        then calls _projection_match_search with a narrow window (15 pixels, TH_HIGH=100, MIN_MATCHES=20).
        
        Returns True if tracking is successful, False otherwise.
        """
        # --- 1. Pose Prediction using last frame's pose and precomputed velocity ---
        if self.last_frame is None or not hasattr(self.last_frame, 'pose'):
            return False
        predicted_pose = np.dot(self.mVelocity, self.last_frame.pose)
        self.current_pose = predicted_pose

        # Narrow search parameters:
        windowSize = 15
        TH_HIGH = 100
        MIN_MATCHES = 20

        matches = self._projection_match_search(keypoints, descriptors, windowSize, TH_HIGH, MIN_MATCHES)
        if matches is None:
            return False

        # --- 5. Refine Pose ---
        temp_keyframe = KeyFrame(-1, self.current_pose, self.K, keypoints, descriptors)
        temp_keyframe.map_points = matches
        self.bundle_adjustment.optimize_pose(temp_keyframe, self.map)
        self.current_pose = temp_keyframe.pose
        self.tracked_map_points = matches

        return True

    def track_with_map_points(self, keypoints, descriptors):
        """
        Fallback method: If the motion model narrow search fails,
        perform a wider search using looser constraints.
        
        Uses the last keyframe's pose as the reference, then calls
        _projection_match_search with a wider window (200 pixels, TH_HIGH=100, MIN_MATCHES=10).
        
        Returns True if tracking is successful, False otherwise.
        """
        if self.motion_model["last_keyframe"] is None:
            return False
        reference_pose = self.motion_model["last_keyframe"].pose.copy()
        self.current_pose = reference_pose

        # Wider search parameters:
        windowSize = 200
        TH_HIGH = 100
        MIN_MATCHES = 10

        matches = self._projection_match_search(keypoints, descriptors, windowSize, TH_HIGH, MIN_MATCHES)
        if matches is None:
            return False

        temp_keyframe = KeyFrame(-1, self.current_pose, self.K, keypoints, descriptors)
        temp_keyframe.map_points = matches
        self.bundle_adjustment.optimize_pose(temp_keyframe, self.map)
        self.current_pose = temp_keyframe.pose
        self.tracked_map_points = matches

        return True

    
    def global_relocalization(self, keypoints, descriptors):
        """
        Perform global relocalization when tracking is lost.
        
        Steps:
        1. Convert the current frame's descriptors into visual words.
        2. Query the BoW database for candidate keyframes.
        3. For each candidate keyframe:
                a. Use the feature matcher to match candidate keyframe descriptors with current frame descriptors.
                b. For each match, if the candidate keyframe has an associated MapPoint,
                    add the corresponding 3D position (from the map) and 2D point (from the current frame) to lists.
                c. Use cv2.solvePnPRansac to estimate the camera pose from these correspondences.
        4. If a candidate yields a pose with enough inliers, refine the pose using motion-only bundle adjustment,
            update self.current_pose, and return True.
        5. Otherwise, return False.
        
        Returns:
        True if a valid pose is recovered and refined, False otherwise.
        """
        # 1. Convert current frame descriptors to visual words.
        visual_words = self.bow_database.get_visual_word(descriptors)
        
        # 2. Query the BoW database for candidate keyframes.
        top_candidates = self.bow_database.query(visual_words, top_k=5)
        if not top_candidates:
            print("Global Relocalization: No candidate keyframes found.")
            return False

        best_pose = None
        best_inliers_count = 0
        best_candidate_id = None
        best_matches = None

        # 3. Process each candidate keyframe.
        for candidate_id, score in top_candidates:
            candidate_kf = self.map.get_keyframe(candidate_id)
            if candidate_kf is None:
                continue
            
            # 3a. Match candidate keyframe descriptors to current frame descriptors.
            # (We assume self.feature_matcher.match returns a list of match objects.)
            matches = self.feature_matcher.match(candidate_kf.descriptors, descriptors, candidate_kf.keypoints, keypoints)
            if matches is None or len(matches) < 15:
                continue
            
            pts_3d = []
            pts_2d = []
            valid_matches = [] # Track with matches have valid map points

            # 3b. Build correspondences.
            for m in matches:
                cand_idx = m.queryIdx  # Index in candidate keyframe.
                curr_idx = m.trainIdx   # Index in current frame.

                # Check if candidate keyframe has an associated map point for this keypoint.
                if cand_idx not in candidate_kf.map_points:
                    continue
        
                map_point_id = candidate_kf.map_points[cand_idx]
                mp = self.map.get_map_point(map_point_id)
                if mp is None:
                        continue
                pts_3d.append(mp.position)
                pts_2d.append(keypoints[curr_idx].pt)
                valid_matches.append(m) # track the valid matches
            if len(pts_3d) < 6:
                continue

            pts_3d = np.array(pts_3d, dtype=np.float32)
            pts_2d = np.array(pts_2d, dtype=np.float32)
            
            # 3c. Estimate pose using PnP with RANSAC.
            retval, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts_3d, pts_2d, self.K, None,
                flags=cv2.SOLVEPNP_ITERATIVE,
                reprojectionError=8.0, confidence=0.99, iterationsCount=100)
            
            if not retval or inliers is None or len(inliers) < 10:
                continue
            
            # Convert estimated pose into 4x4 matrix.
            R_est, _ = cv2.Rodrigues(rvec)
            pose_est = np.eye(4, dtype=np.float32)
            pose_est[:3, :3] = R_est
            pose_est[:3, 3] = tvec.flatten()
            
            if len(inliers) > best_inliers_count:
                best_inliers_count = len(inliers)
                best_pose = pose_est
                best_candidate_id = candidate_id
                best_matches = valid_matches

        if best_pose is None:
            print("Global Relocalization: Failed to recover pose.")
            return False

        # 4. Refine the pose using motion-only bundle adjustment.
        self.current_pose = best_pose

        # Update tracked map points from the best candidate
        best_candidate_kf = self.map.get_keyframe(best_candidate_id)
        for m in best_matches:
            cand_idx = m.queryIdx
            curr_idx = m.trainIdx

            if cand_idx in best_candidate_kf.map_points:
                map_point_id = best_candidate_kf.map_points[cand_idx]
                if self.map.get_map_point(map_point_id) is not None:
                    self.tracked_map_points[curr_idx] = map_point_id
                    
        temp_keyframe = KeyFrame(-1, self.current_pose, self.K, keypoints, descriptors)
        # (Optionally, you could re-associate map points using a guided search here.)
        self.bundle_adjustment.optimize_pose(temp_keyframe, self.map)
        self.current_pose = temp_keyframe.pose

        print(f"Global Relocalization: Successful with {best_inliers_count} inliers.")
        return True
    
    def track_local_map(self, keypoints = None, descriptors = None):
        """
        Track map points in the local map and refine camera pose.
    
        Steps:
        1. Find local map (keyframes K1 that share points with current frame and their neighbors K2)
        2. Project all map points visible in these keyframes into the current frame
        3. Apply filtering criteria (viewing angle, scale, etc.)
        4. Match remaining projections with still-unmatched ORB features
        5. Optimize the camera pose with all found correspondences
        
        Returns: True if tracking is successful with sufficient matches, False otherwise
        """
        if self.current_pose is None or not self.tracked_map_points:
            print("Cannot track local map: no initial pose or tracked points")
            return False
        
        # Use the keypoints and descriptors from the current frame
        if keypoints is None:
            keypoints = self.current_keypoints
        if descriptors is None:
            descriptors = self.current_descriptors
        
        
        if keypoints is None or descriptors is None:
            print("No features available in current frame")
            return False
        
        # Create set to track already matched keypoints in current frame
        matched_indices = set(self.tracked_map_points.keys()) # 2d point indices of current frame with map points
        
        # --- 1. Indentify the local map ---

        # Find K1: keyframes sharing map points with current frame
        K1 = set()
        point_observations = {} # Keyframe ID -> count of shared map points

        # For each map point seen in current frame, find all keyframes that observe it
        for keypoint_idx, map_point_id in self.tracked_map_points.items():
            map_point = self.map.get_map_point(map_point_id)
            if map_point is None:
                continue

            # Add all keyframes that observe this map points 
            if hasattr(map_point, 'keyframes_observed'):
                for kf_id in map_point.keyframes_observed:
                    K1.add(kf_id)

                    # Count observations for finding reference keyframe
                    if kf_id not in point_observations:
                        point_observations[kf_id] = 0
                    point_observations[kf_id] += 1

        # Find reference keyframe (Kref): keyframe in K1 with most shared map points
        reference_keyframe_id = None
        max_observations = 0
        for kf_id, count in point_observations.items():
            if count>max_observations:
                max_observations = count
                reference_keyframe_id = kf_id

        if reference_keyframe_id is None:
            print("No reference keyframe found for local map.")
            return False

        # Get K2: neighbors to K1 in covisibility graph
        K2 = set()
        for kf_id in K1:
            kf = self.map.get_keyframe(kf_id)
            if kf is None:
                continue

            # Add neighbors with strong covisibility connections
            K2.update(kf.get_best_covisibility_keyframes(min_shared_points=15))

        # Combine K1 and K2 to form the local map keyframes
        local_map_keyframe_ids = K1.union(K2)

        # --- 2. Find visible map points in the local map ---

        # Collect all map points observed by the local map keyframes
        local_map_points = set()
        for kf_id in local_map_keyframe_ids:
            kf = self.map.get_keyframe(kf_id)
            if kf is None:
                continue

            # Add all map points observed by this keyframe
            for keypoint_idx, map_point_id in kf.map_points.items():
                if map_point_id not in self.tracked_map_points.values():  # Only add points not already tracked
                    local_map_points.add(map_point_id)

        # --- 3. Project and filter local map points ---

        # Get current camera information for projection
        K = self.K                      # Camera intrinsic matrix
        R = self.current_pose[:3, :3]   # Rotation matrix
        t = self.current_pose[:3, 3]    # Translation vector
        camera_center = -R.T @ t        # Camera center in world coordinates
        rvec, _ = cv2.Rodrigues(R)      # Convert rotation matrix to rotation vector

        # Get image dimensions
        h, w = self.current_frame.shape

        new_matches ={} # will store new keypoints_idx -> map_points_id matches

        # Process each map point in the local map
        for map_point_id in local_map_points:
            map_point = self.map.get_map_point(map_point_id)
            if map_point is None:
                continue

            # 1. Project the map point into current frame
            pt3d = map_point.position.reshape(1,3)
            image_pts, _ = cv2.projectPoints(pt3d, rvec, t, K, None)
            x, y = image_pts[0, 0]

            # Discard if projection is outside image bounds
            if x < 0 or x >= w or y < 0 or y >= h:
                continue

            # 2. Check viewing angle
            viewing_ray = map_point.position - camera_center # Vector from camera to that 3d point
            viewing_ray = viewing_ray/ np.linalg.norm(viewing_ray) # Normalize

            # Get mean viewing direction (if available)

            mean_viewing_dir = None 
            if hasattr(map_point, 'compute_mean_viewing_direction'):
                mean_viewing_dir =map_point.compute_mean_viewing_direction()
            elif hasattr(map_point, 'viewing_directions') and map_point.viewing_directions:
                # Compute mean direction directly if function not available
                dirs = np.array(map_point.viewing_directions)
                mean_viewing_dir = np.mean(dirs, axis=0)
                mean_viewing_dir = mean_viewing_dir / np.linalg.norm(mean_viewing_dir)

            if mean_viewing_dir is None:
                continue

            # Discard if viewing angle is too large (>60 degrees)
            cos_angle = np.dot(viewing_ray, mean_viewing_dir)
            if cos_angle < np.cos(np.radians(60)):
                continue

            # 3. Check scale invariance
            dist = np.linalg.norm(map_point.position - camera_center)

            # Discard if point is too close or too far based on scale invariance
            d_min  = getattr(map_point, 'd_min' , None)
            d_max = getattr(map_point, 'd_max' , None)
            if d_min is not None and d_max is not None:
                if dist < d_min or dist > d_max:
                    continue
            # 4. Compute the scale in the frame
            scale = 1.0
            if d_min is not None:
                scale = dist/d_min

            # 5. Find the best match among unmatched keypoints
            best_match_idx = None
            best_match_dist = float('inf')
            max_search_radius = 15 # Search radius in pixels

            # Search for keypoints near the projected position at the right scale
            for i, kp in enumerate(keypoints):
                # Skip already matched keypoints
                if i in matched_indices:
                    continue

                # Check if keypoint is close to the projected position
                dx = kp.pt[0] - x
                dy = kp.pt[1] - y
                if dx**2 + dy**2 > max_search_radius**2:
                    continue

                # Check if scale are compatible 
                if abs(np.log2(scale / (1 << kp.octave))) >1.0: # More than 1 octave difference
                    continue

                # Compare descriptors
                desc_dist = cv2.norm(descriptors[i], map_point.descriptor, cv2.NORM_HAMMING)
                if desc_dist < best_match_dist:
                    best_match_dist = desc_dist
                    best_match_idx = i

            if best_match_idx is not None and best_match_idx < 50:  # Threshold for descriptor distance
                new_matches[best_match_idx] = map_point_id
                matched_indices.add(best_match_idx)

 




    
    def check_new_keyframe(self, frame_id, keypoints, descriptors):
        """
        Decide if the current frame should be inserted as a new keyframe.
        """
        # To be implemented in the next step
        pass
