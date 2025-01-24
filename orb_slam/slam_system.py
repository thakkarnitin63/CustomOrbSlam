from orb_slam.image_loader import ImageLoader
from orb_slam.feature_extractor import FeatureExtractor
from orb_slam.feature_matcher import FeatureMatcher
from orb_slam.pose_estimator import PoseEstimator
from orb_slam.map_manager import MapManager
from orb_slam.optimizer import Optimizer
from orb_slam.visualizer import Visualizer

class SLAMSystem:
    def __init__(self, config):
        self.image_loader = ImageLoader(config['dataset_path'])
        self.feature_extractor = FeatureExtractor(config['nfeatures'])
        self.feature_matcher = FeatureMatcher()
        self.pose_estimator = PoseEstimator(config['K'])
        self.map_manager = MapManager()
        self.optimizer = Optimizer()
        self.visualizer = Visualizer()

    def run(self):
        # Main loop to process images and perform SLAM
        pass