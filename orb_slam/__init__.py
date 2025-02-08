# orb_slam/__init__.py

# Importing key classes to simplify package interface
from .image_loader import ImageLoader
from .feature_extractor import FeatureExtractor
from .feature_matcher import FeatureMatcher
from .pose_estimator import PoseEstimator
# from .global_init_map import MapManager
from .optimizer import Optimizer
from .visualizer import Visualizer
from .loop_closure_detector import LoopClosureDetector
from .keyframe_manager import KeyframeManager
from .mapping import SparseMapping, MapPoint, KeyFrame

# Defining the public API of the package
__all__ = [
    'ImageLoader',
    'FeatureExtractor',
    'FeatureMatcher',
    'PoseEstimator',
    # 'MapManager',
    'Optimizer',
    'Visualizer',
    'LoopClosureDetector',
    'KeyframeManager',
    'KeyFrame'
    'MapPoint',
    'SparseMapping'
]
