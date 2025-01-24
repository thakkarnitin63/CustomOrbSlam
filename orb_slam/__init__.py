# orb_slam/__init__.py

# Importing key classes to simplify package interface
from .image_loader import ImageLoader
from .feature_extractor import FeatureExtractor
from .feature_matcher import FeatureMatcher
from .pose_estimator import PoseEstimator
from .map_manager import MapManager
from .optimizer import Optimizer
from .visualizer import Visualizer
from .slam_system import SLAMSystem

# Defining the public API of the package
__all__ = [
    'ImageLoader',
    'FeatureExtractor',
    'FeatureMatcher',
    'PoseEstimator',
    'MapManager',
    'Optimizer',
    'Visualizer',
    'SLAMSystem'
]
