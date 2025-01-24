from .image_loader import ImageLoader
from .feature_extractor import FeatureExtractor
from .feature_matcher import FeatureMatcher
from .pose_estimator import PoseEstimator
from .map_manager import MapManager
from .optimizer import Optimizer
from .visualizer import Visualizer

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
