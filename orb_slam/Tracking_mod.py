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
        :param camera_intrinsics: Intrinsic matrix K (3x3 numpy array)
        :param map_instance: Instance of the Map class that contains keyframes and map points.
        :param bow_database: Instance of BoWDatabase for relocalization and loop closure.
        """
        self.K = camera_intrinsics  # Camera intrinsic matrix
        self.map = map_instance  # Global Map
        self.bow_database = bow_database  # Bag of Words Database for relocalization and loop detection
        self.feature_extractor = FeatureExtractor()
        self.feature_matcher = FeatureMatcher()
        self.bundle_adjustment = BundleAdjustment(self.K)
        
        # Motion model: initially use keyframes from map initialization if available.
        # For a new sequence, ensure that at least two keyframes are present.
        self.motion_model = {
            "last_keyframe": self.map.get_keyframe(1),      # Assumes keyframe 1 exists.
            "second_last_keyframe": self.map.get_keyframe(0)  # Assumes keyframe 0 exists.
        }

        # Current Frame Information
        self.current_frame = None
        self.current_pose = None  # Camera pose estimation
        self.tracked_map_points = {}  # Dictionary of 2D keypoint index -> 3D MapPoint ID
        self.last_frame = None # Store previous frame for reference

    def process_frame(self, frame, frame_id):
        """
        Process an incoming frame for tracking.
        :param frame: Input image (grayscale)
        :param frame_id: ID of the frame in the sequence
        """
        self.current_frame = frame
        keypoints, descriptors = self.feature_extractor.extract(frame)
        
        success = False
        if frame_id > 0:
            # Attempt tracking using a cascade of methods.
            success = self.track_with_motion_model(keypoints, descriptors)
            if not success:
                success = self.track_with_map_points(keypoints, descriptors)
            if not success:
                success = self.global_relocalization(keypoints, descriptors)
            
            if success:
                self.track_local_map()
                self.check_new_keyframe(frame_id, keypoints, descriptors)
            else:
                print("Tracking lost!")
        else:
            # For the first frame, initialize pose as identity.
            self.current_pose = np.eye(4)
        
        # Update motion model and map if a valid pose was estimated.
        if self.current_pose is not None:
            new_keyframe = KeyFrame(frame_id, self.current_pose, self.K, keypoints, descriptors)
            self.map.add_keyframe(new_keyframe)
            # Update motion model: shift last keyframes.
            self.motion_model["second_last_keyframe"] = self.motion_model["last_keyframe"]
            self.motion_model["last_keyframe"] = new_keyframe
        
        self.last_frame = frame
        
    def track_with_motion_model(self, keypoints, descriptors):
        """
        Robust tracking using a constant velocity motion model combined with batch projection,
        spatial indexing (KD‑tree), and BFMatcher with a ratio test.
        
        Steps:
        1. Predict the current pose using a constant velocity model from the last two keyframes.
        2. Batch‑project all MapPoints from the last keyframe into the current image using cv2.projectPoints.
        3. Build a KD‑tree on current frame keypoint locations.
        4. For each projected MapPoint:
            a. Query the KD‑tree for candidate keypoints within a specified window.
            b. Use BFMatcher (with k=2 and a ratio test) to match the stored MapPoint descriptor
                against the candidate descriptors.
        5. If a sufficient number of matches is found, refine the current pose using motion‑only bundle adjustment.
        
        Returns:
        True if tracking is successful (enough matches found and pose refined), False otherwise.
        """
        # --- 1. Pose Prediction using Constant Velocity Model ---
        if (self.motion_model["last_keyframe"] is None or 
            self.motion_model["second_last_keyframe"] is None):
            return False

        last_pose = self.motion_model["last_keyframe"].pose
        second_last_pose = self.motion_model["second_last_keyframe"].pose
        # Compute velocity and predicted pose:
        velocity = np.dot(last_pose, np.linalg.inv(second_last_pose))
        predicted_pose = np.dot(velocity, last_pose)
        self.current_pose = predicted_pose

        # Convert rotation matrix to rotation vector for cv2.projectPoints
        R = self.current_pose[:3, :3]
        t = self.current_pose[:3, 3]
        rvec, _ = cv2.Rodrigues(R)

        # --- 2. Batch Project All MapPoints from Last Keyframe ---
        # Get map point indices from the last keyframe's stored associations.
        mp_keys = list(self.motion_model["last_keyframe"].map_points.keys())
        if len(mp_keys) == 0:
            return False

        mp_positions = []
        mp_descs = []
        mp_map_idx = []  # Store the corresponding key in the last keyframe's map_points dict.
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

        mp_positions = np.array(mp_positions, dtype=np.float32)  # (N, 3)
        mp_descs = np.array(mp_descs)  # (N, descriptor_length)

        # Use cv2.projectPoints to batch-project 3D points into 2D.
        imagePoints, _ = cv2.projectPoints(mp_positions, rvec, t, self.K, None)
        projected_pts = imagePoints.reshape(-1, 2)  # Shape: (N, 2)

        # --- 3. Build a KD-Tree for Current Frame Keypoints ---
        # Extract 2D positions from the current keypoints.
        current_kp_positions = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        from scipy.spatial import cKDTree
        kd_tree = cKDTree(current_kp_positions)

        # --- 4. For Each Projected MapPoint, Find Candidate Matches ---
        windowSize = 200  # Pixel search radius
        mfNNratio = 0.9   # Ratio test threshold
        TH_HIGH = 100     # Maximum acceptable Hamming distance
        projected_matches = {}  # Mapping: last keyframe's keypoint index -> current frame keypoint index
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        h, w = self.current_frame.shape  # Image dimensions (assumed grayscale)

        for i, proj in enumerate(projected_pts):
            x, y = proj
            # Discard if projected point is outside image bounds.
            if x < 0 or x >= w or y < 0 or y >= h:
                continue

            # Query KD-Tree for all keypoints within 'windowSize' of the projected point.
            candidate_indices = kd_tree.query_ball_point(proj, windowSize)
            if len(candidate_indices) == 0:
                continue

            # Gather candidate descriptors from the current frame.
            candidate_descs = [descriptors[idx] for idx in candidate_indices]
            candidate_descs = np.array(candidate_descs)

            # Use BFMatcher's knnMatch (with k=2) to compare the MapPoint descriptor with candidates.
            map_desc = np.array([mp_descs[i]])
            matches = bf.knnMatch(map_desc, candidate_descs, k=2)
            if matches is None or len(matches) == 0:
                continue

            best_matches = matches[0]
            # If only one candidate is found, check its distance.
            if len(best_matches) < 2:
                if best_matches[0].distance < TH_HIGH:
                    # Map back to the candidate index in the current frame.
                    current_idx = candidate_indices[best_matches[0].trainIdx]
                    projected_matches[mp_map_idx[i]] = current_idx
            else:
                if (best_matches[0].distance < mfNNratio * best_matches[1].distance and
                        best_matches[0].distance < TH_HIGH):
                    current_idx = candidate_indices[best_matches[0].trainIdx]
                    projected_matches[mp_map_idx[i]] = current_idx

        # --- 5. Validate and Refine Pose ---
        MIN_MATCHES = 20
        if len(projected_matches) < MIN_MATCHES:
            print(f"TrackWithMotionModel: Not enough matches ({len(projected_matches)})")
            return False

        # Create a temporary keyframe encapsulating current observations.
        temp_keyframe = KeyFrame(-1, self.current_pose, self.K, keypoints, descriptors)
        temp_keyframe.map_points = projected_matches

        # Refine the camera pose using motion-only Bundle Adjustment.
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
