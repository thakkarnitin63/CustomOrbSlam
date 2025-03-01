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

        
    def track_with_motion_model(self, keypoints, descriptors):
        """
        Robust tracking using a constant velocity motion model, batch projection,
        spatial indexing (KD‑tree), and BFMatcher with a ratio test.
        
        Steps:
        1. Predict the current pose using the last frame’s pose (with precomputed velocity).
        2. Batch-project all MapPoints from the last keyframe into the current image using cv2.projectPoints.
        3. Build a KD‑tree on current frame keypoint locations.
        4. For each projected MapPoint:
            a. Query the KD‑tree for candidate keypoints within a narrow window (~15 pixels).
            b. Use BFMatcher (with k=2 and a ratio test) to match the stored MapPoint descriptor against candidate descriptors.
        5. If a sufficient number of matches is found, refine the current pose using motion‑only bundle adjustment.
        
        Returns:
        True if tracking is successful (enough matches found and pose refined), False otherwise.
        """
        # --- 1. Pose Prediction using last frame's pose and precomputed velocity ---
        if self.last_frame is None or not hasattr(self.last_frame, 'pose'):
            return False
        # Use precomputed velocity (self.mVelocity) applied to the last frame's pose.
        predicted_pose = np.dot(self.mVelocity, self.last_frame.pose)
        self.current_pose = predicted_pose

        # Convert the rotation matrix from the current pose to a rotation vector (required for cv2.projectPoints).
        R = self.current_pose[:3, :3]
        t = self.current_pose[:3, 3]
        rvec, _ = cv2.Rodrigues(R)

        # --- 2. Batch Project All MapPoints from Last Keyframe ---
        # Retrieve map point indices from the last keyframe's stored associations.
        mp_keys = list(self.motion_model["last_keyframe"].map_points.keys())
        if len(mp_keys) == 0:
            return False

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
        if len(mp_positions) == 0:
            return False

        mp_positions = np.array(mp_positions, dtype=np.float32)  # Shape: (N, 3)
        mp_descs = np.array(mp_descs)  # Shape: (N, descriptor_length)

        # Project all 3D map point positions into the image.
        imagePoints, _ = cv2.projectPoints(mp_positions, rvec, t, self.K, None)
        projected_pts = imagePoints.reshape(-1, 2)  # (N, 2)

        # --- 3. Build a KD-Tree for Current Frame Keypoints ---
        current_kp_positions = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        from scipy.spatial import cKDTree
        kd_tree = cKDTree(current_kp_positions)

        # --- 4. For Each Projected MapPoint, Find Candidate Matches within a narrow window ---
        windowSize = 15  # Narrow window (in pixels) for guided search.
        mfNNratio = 0.9  # Ratio test threshold.
        TH_HIGH = 100    # Maximum acceptable Hamming distance.
        projected_matches = {}  # Mapping: last keyframe's keypoint index -> current frame keypoint index.
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        h_img, w_img = self.current_frame.shape  # Assumes a grayscale image.

        for i, proj in enumerate(projected_pts):
            x, y = proj
            # Discard projected points outside the image.
            if x < 0 or x >= w_img or y < 0 or y >= h_img:
                continue

            # Query the KD-Tree for candidate keypoints within the narrow window.
            candidate_indices = kd_tree.query_ball_point(proj, windowSize)
            if len(candidate_indices) == 0:
                continue

            # Gather candidate descriptors.
            candidate_descs = [descriptors[idx] for idx in candidate_indices]
            candidate_descs = np.array(candidate_descs)

            # Use BFMatcher with knnMatch (k=2) to compare the stored MapPoint descriptor with candidates.
            map_desc = np.array([mp_descs[i]])
            matches = bf.knnMatch(map_desc, candidate_descs, k=2)
            if matches is None or len(matches) == 0:
                continue

            best_matches = matches[0]
            if len(best_matches) < 2:
                if best_matches[0].distance < TH_HIGH:
                    current_idx = candidate_indices[best_matches[0].trainIdx]
                    projected_matches[mp_map_idx[i]] = current_idx
            else:
                if best_matches[0].distance < mfNNratio * best_matches[1].distance and best_matches[0].distance < TH_HIGH:
                    current_idx = candidate_indices[best_matches[0].trainIdx]
                    projected_matches[mp_map_idx[i]] = current_idx

        # --- 5. Validate and Refine Pose ---
        MIN_MATCHES = 20
        if len(projected_matches) < MIN_MATCHES:
            print(f"TrackWithMotionModel: Not enough matches ({len(projected_matches)})")
            return False

        # Create a temporary keyframe to encapsulate the matched correspondences.
        temp_keyframe = KeyFrame(-1, self.current_pose, self.K, keypoints, descriptors)
        temp_keyframe.map_points = projected_matches

        # Refine the current pose using motion-only bundle adjustment.
        self.bundle_adjustment.optimize_pose(temp_keyframe, self.map)
        self.current_pose = temp_keyframe.pose
        self.tracked_map_points = projected_matches

        return True
    
    def track_with_map_points(self, keypoints, descriptors):
        """
        Use map points to estimate the camera pose if motion model fails.
        """
        # To be implemented in the next step
        pass
    
    def global_relocalization(self, keypoints, descriptors):
        """
        Perform relocalization if tracking is lost.
        """
        # Convert descriptors into visual words
        visual_words = self.bow_database.get_visual_word(descriptors)
        
        # Query the database for the best matching keyframes
        top_matches = self.bow_database.query(visual_words, top_k=5)
        
        # To be implemented: Use top matches for pose estimation
        pass
    
    def track_local_map(self):
        """
        Track map points in the local map and refine pose.
        """
        # To be implemented in the next step
        pass
    
    def check_new_keyframe(self, frame_id, keypoints, descriptors):
        """
        Decide if the current frame should be inserted as a new keyframe.
        """
        # To be implemented in the next step
        pass
