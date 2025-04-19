import numpy as np
import cv2

from orbslam.core.map import Map
from orbslam.frontend.tracking import Tracking
from orbslam.utils.bow_database import BoWDatabase
from orbslam.frontend.feature_extractor import FeatureExtractor
from orbslam.frontend.feature_matcher import FeatureMatcher

class OrbSlam:
    """
    Main ORB-SLAM system class that coordinates all components.
    """
    
    def __init__(self, camera_intrinsics, settings_file=None):
        """
        Initialize the SLAM system.
        
        Args:
            camera_intrinsics: Camera intrinsic matrix (3x3 numpy array)
            settings_file: Path to a settings file (optional)
        """
        self.K = camera_intrinsics
        
        # Initialize components
        self.map = Map()
        self.feature_extractor = FeatureExtractor()
        self.feature_matcher = FeatureMatcher()
        self.bow_database = BoWDatabase()
        
        # Initialize tracking
        self.tracking = Tracking(self.K, self.map, self.bow_database)
        
    def process_frame(self, frame, frame_id):
        """
        Process a new frame.
        
        Args:
            frame: The input image frame
            frame_id: The frame ID
            
        Returns:
            The camera pose
        """
        return self.tracking.process_frame(frame, frame_id)
        
    def shutdown(self):
        """Shutdown the SLAM system."""
        print("SLAM system has been shut down.")