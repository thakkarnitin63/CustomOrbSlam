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
        mp_map_idx = []  # These indices correspond to map points in map for this keyframe
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
    
    def track_local_map(self):
        """
        Track map points in the local map to refine camera pose.
    
        1. Identifies the local map(keyframes sharing points with current frame and their neighbors)
        2. Projects all map points from these keyframes into the current frame
        3. Matches projected points with unmatched features using filtering criteria
        4. Refines the camera pose with all matches using motion-only bundle adjustment

        Returns:
            bool: True if tracking is successful, False otherwise
        """
        # Skip if we don't have initial pose or tracked points
        if self.current_pose is None or not self.tracked_map_points:
            print("Cannot track local map: no initial pose or tracked points")
            return False
        
        
        keypoints = self.current_keypoints
        descriptors = self.current_descriptors

        if keypoints is None or descriptors is None:
            print("No features available in current frame")
            return False
        
        # Keep track of already matched keypoints
        matched_indices = set(self.tracked_map_points.keys())

        # ===== 1. Identify Local Map ======

        # Find K1: keyframe sharing map points with current frame
        K1 = set()
        point_observations = {} # Keyframe ID -> count of shared points

        # for each map points in current frame, find keyframes that observe it 
        for point_id in self.tracked_map_points.values(): # getting 3d point id which we saw from previous frame to new frame
            map_point = self.map.get_map_points(point_id)
            if map_point is None:
                continue

            for kf in self.map.keyframe.values(): # checking all keyframe instance in global map
                # Check if this keyframe observes the map point
                for kp_idx, mp_id in kf.map_points.items():
                    if mp_id == point_id:
                        K1.add(kf.id)

                        # Count for reference keyframe selection
                        if kf.id not in point_observations:
                            point_observations[kf.id]=0
                        point_observations[kf.id] += 1
                        break
        if not K1:
            print("No keyframe share points with current frame")
            return False
        
        # Find reference keyframe (most shared points)
        K_ref_id = max(point_observations.items(), key=lambda x: x[1])[0]
        K_ref = self.map.get_keyframe(K_ref_id)

        # Find K2: Neighbor keyframe in covisibility graph
        K2 = set()
        for kf_id in K1:
            kf = self.map.get_keyframe(kf_id)
            if kf is None:
                continue

            # Add neighbors with strong connections (threshold: 15 shared points)
            neighbors = kf.get_best_covisibility_keyframes(min_shared_points=15)
            K2.update(neighbors)


        # Combine to get all local keyframes
        local_keyframe_ids = K1.union(K2)

        #====== 2. Project Map points ==========

        # Get all map points in local keyframes
        local_map_points = {} # Mappoint ID -> MapPoint

        for kf_id in local_keyframe_ids:
            kf = self.map.get_keyframe(kf_id)
            if kf is None:
                continue

            for _, point_id in kf.map_points.items():
                # Skip points already tracked in current frame
                if point_id in self.tracked_map_points.values():
                    continue

                map_point = self.map.get_map_point(point_id)
                if map_point is not None:
                    local_map_points[point_id] = map_point

        # Setup for projection
        img_h, img_w = self.current_frame.shape
        R = self.current_pose[:3, :3]
        t = self.current_pose[:3, 3]
        rvec, _ = cv2.Rodrigues(R)
        camera_center = -np.dot(R.T, t) # Camera center in world coordinates
        # X_camera = R·X_world + t
        # For the camera center, we have X_camera = [0,0,0] by definition. So:
        # 0 = R·X_center + t
        # R·X_center = -t
        # X_center = -R^T·t

        # Project and match points
        new_matches = {} # Keypoints idx -> MapPoint ID

        for point_id, map_point in local_map_points.items():
            # 1. Project point to image
            position = map_point.position.reshape(1,3) # getting 3d point position
            img_points, _ = cv2.projectPoints(position,rvec,t, self.K, None)
            x,y =img_points[0,0]

            # Skip if outside image bounds
            if x < 0 or x>= img_w or y < 0 or y >= img_h:
                continue

            # 2. Check viewing angle
            view_ray = map_point.position - camera_center
            view_ray = view_ray / np.linalg.norm(view_ray)

            # Get mean viewing direction of map point
            mean_dir = map_point.compute_mean_viewing_direction() if hasattr(map_point, 'compute_mean_viewing_direction') else None
            if mean_dir is None and hasattr(map_point, 'viewing_directions') and map_point.viewing_directions:
                dirs = np.array(map_point.viewing_directions)
                mean_dir = np.mean(dirs, axis=0)
                mean_dir = mean_dir / np.linalg.norm(mean_dir)

            if mean_dir is None:
                continue

            #Skip if viewing angle too large (>60 degree)
            cos_angle = np.dot(view_ray, mean_dir)
            if cos_angle < np.cos(np.radians(60)):
                continue  

            # 3. Check distance (scale invariance)
            dist = np.linalg.norm(map_point.position - camera_center)

            if hasattr(map_point, 'd_min') and hasattr(map_point, 'd_max'):
                if map_point.d_min is not None and map_point.d_max is not None:
                    if dist < map_point.d_min or dist > map_point.d_max:
                        continue 

            # 4. Find best match among unmatched keypoints
            best_idx = None
            best_dist = float('inf')
            search_radius = 15 # Pixels

            for i, kp in enumerate(keypoints):
                # Skip already matched keypoints
                if i in matched_indices:
                    continue

                # Check distance to projection
                dx = kp.pt[0] - x
                dy = kp.pt[1] - y
                if dx**2 + dy**2 > search_radius**2:
                    continue

                #Check scale compatibility
                scale_factor = dist/map_point.d_min if hasattr(map_point, 'd_min') and map_point.d_min else 1.0
                if abs(np.log2(scale_factor / (1 << kp.octave))) > 1.0: # More than 1 octave difference
                    continue

                # Compare descriptors
                dist = cv2.norm(descriptors[i], map_point.descriptor, cv2.NORM_HAMMING)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

            # Add match if good enough
            if best_idx is not None and best_idx < 50: # Threshold for descriptor distance
                new_matches[best_idx] = point_id
                matched_indices.add(best_idx)     
    

        # ====== 3. Optimize pose with all matches =======
        
        # Combine initial and new matches
        all_matches = {** self.track_with_map_points, **new_matches}

        if len(all_matches) < 20: # Not enough matches for reliable pose estimation
            print(f"Not enough matches for local map tracking: {len(all_matches)}")
            return False
        
        # create temporary keyframe with all matches
        temp_keyframe = KeyFrame(-1, self.current_pose, self.K, keypoints, descriptors)
        temp_keyframe.map_points = all_matches

        # Optimize pose
        self.bundle_adjustment.optimize_pose(temp_keyframe, self.map)

        # Update current pose and tracked points
        self.current_pose = temp_keyframe.pose
        self.tracked_map_points = all_matches

        print(f"Local map tracking: {len(self.track_with_map_points)} points tracked")
        return True
    
        


    
    def check_new_keyframe(self, frame_id, keypoints, descriptors):
        """
        Decide if the current frame should be inserted as a new keyframe.

        According to the ORB-SLAM paper, a new keyframe is inserted when all these conditions are met:
        1. More than 20 frames have passed from last global relocalization
        2. Local mapping is idle, or more than 20 frames have passed from last keyframe insertion
        3. Current frame tracks at least 50 map points
        4. Current frame tracks less than 90% of points compared to reference keyframe
        
        Returns:
            bool: True if a new keyframe was inserted, False otherwise
        """
        # Condition 1: check if enough frames since last relocalization
        if self.mMinFrames < 20:
            # Not enough frames since relocalization
            self.mMinFrames +=1
            return False
        
        # Condition 2: Check if mapping is idle or enough frames since last keyframe
        # Note: In a complete implementation, we would need a way to check if mapping is idle
        # As a simplification, we will just use frame counter
        if self.mMaxFrames < 20:
            # Not enough frames since last keyframe insertion
            self.mMaxFrames += 1
            return False
        
        if len(self.tracked_map_points) <50:
            # Not enough points tracked
            return False
        
        # Condition 4: Check visual changes (compare with reference keyframe)
        reference_keyframe = self._get_reference_keyframe()
        if reference_keyframe is None:
            # No valid reference, use conservative approach and insert keyframe
            print("No reference keyframe found, inserting new keyframe")
        else:
            # Count points in reference keyframe
            ref_point_count = len(reference_keyframe.map_points)
            if ref_point_count ==0:
                # No points in reference, use conservative approach
                print("Reference keyframe has no points, inserting new keyframe")
            else:
                # Calculate ratio of tracked points
                current_tracked_ratio = len(self.tracked_map_points) / ref_point_count
                if current_tracked_ratio >= 0.9:
                    # Current frame tracks more than 90% of reference points,
                    # not enough visual change to warrant a new keyframe
                    return False
                
        #All conditions met, insert a new keyframe
        print(f"Inserting new keyframe  {frame_id} with {len(self.tracked_map_points)} tracked points")

        # Create a new keyframe
        new_keyframe = KeyFrame(frame_id, self.current_pose, self.K, keypoints, descriptors)


        # Associate current tracked map points with the new keyframe
        for keypoint_idx, map_point_id in self.tracked_map_points.items():
            new_keyframe.add_map_point(keypoint_idx,map_point_id)

        # Add to the map
        self.map.add_keyframe(new_keyframe)

        # Update motion model
        self.motion_model["second_last_keyframe"] =self.motion_model["last_keyframe"]
        self.motion_model["last_keyframe"] = new_keyframe

        # Reset Frame counters
        self.mMaxFrames = 0 # Reset counter for frames since last keyframe

        # Signal to the local mapping thread (in a full implementation)
        # Here we would notify the local mapping thread that a new keyframe is available

        return True
    
    def _get_reference_keyframe(self):
        """
        Find the reference keyframe that shares most map points with current frame.
        
        Returns:
            KeyFrame: The reference keyframe, or None if not found
        """
        # Count observations per keyframe
        observations = {}
        
        # For each mapped point in current frame
        for map_point_id in self.tracked_map_points.values():
            # Find all keyframes observing this point
            for keyframe in self.map.keyframes.values():
                # Check if this keyframe observes the map point
                if any(mp_id == map_point_id for mp_id in keyframe.map_points.values()):
                    if keyframe.id not in observations:
                        observations[keyframe.id] = 0
                    observations[keyframe.id] += 1
        
        # Find keyframe with maximum shared points
        if not observations:
            return None
        
        max_observations = 0
        ref_id = None
        
        for kf_id, count in observations.items():
            if count > max_observations:
                max_observations = count
                ref_id = kf_id
        
        return self.map.get_keyframe(ref_id)




        
