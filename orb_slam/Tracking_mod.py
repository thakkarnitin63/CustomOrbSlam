import numpy as np
import cv2
import g2o
from orb_slam.feature_extractor import FeatureExtractor
from orb_slam.feature_matcher import FeatureMatcher
from orb_slam.bundle_adjustment import BundleAdjustment
from orb_slam.map import Map
from orb_slam.map_point import MapPoint
from orb_slam.keyframe import KeyFrame
from orb_slam.bow_database import BoWDatabase

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
        
        self.last_frame = None  # Store the last successfully tracked frame
        self.current_frame = None  # Store the current frame
        self.tracked_points = []  # 2D-3D correspondences for pose estimation
        self.motion_model = None  # Placeholder for constant velocity model
    
    def process_frame(self, frame, frame_id):
        """
        Process an incoming frame for tracking.
        :param frame: Input image (grayscale)
        :param frame_id: ID of the frame in the sequence
        """
        self.current_frame = frame
        
        # Extract ORB features
        keypoints, descriptors = self.feature_extractor.extract(frame)
        
        if self.last_frame is not None:
            success = self.track_with_motion_model(keypoints, descriptors)
            if not success:
                success = self.track_with_map_points(keypoints, descriptors)
            if not success:
                success = self.global_relocalization(keypoints, descriptors)
            
            if success:
                self.track_local_map()
                self.check_new_keyframe(frame_id)
        
        self.last_frame = frame  # Update last tracked frame
        
    def track_with_motion_model(self, keypoints, descriptors):
        """
        Estimate pose using motion model from the last frame.
        """
        # To be implemented in the next step
        pass
    
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
    
    def check_new_keyframe(self, frame_id):
        """
        Decide if the current frame should be inserted as a new keyframe.
        """
        # To be implemented in the next step
        pass
