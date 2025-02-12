import numpy as np
import cv2
import matplotlib.pyplot as plt

from orb_slam.image_loader import ImageLoader
from orb_slam.feature_extractor import FeatureExtractor
from orb_slam.feature_matcher import FeatureMatcher
from orb_slam.global_init_map import MapInitializer



# Example usage with uploaded image

def main():
    # Paths (adjust these paths as needed)
    sequence_path = '/home/nitin/NitinWs/CustomOrbSlam/data/dataset/sequences/00'
    ground_truth_file = '/home/nitin/NitinWs/CustomOrbSlam/data/dataset/poses/00.txt'

    # Initialize system components
    image_loader = ImageLoader(sequence_path)
    image1 = image_loader.load_image(0)
    image2 = image_loader.load_image(1)

    calibration_matrices = image_loader._load_calibration()
    K = calibration_matrices['P0']

    feature_extractor = FeatureExtractor()
    matcher = FeatureMatcher()

    map_initializer= MapInitializer(K, feature_extractor,matcher)

    # Attempt map initialization.
    init_result = map_initializer.initialize_map(image1, image2)
    if init_result is not None:
        R = init_result['R']
        t = init_result['t']
        points3d = init_result['points3d']
        print("Map initialization succeeded.")
        print("Recovered Rotation:\n", R)
        print("Recovered Translation:\n", t)
        print("Triangulated 3D points shape:", points3d.shape)
    else:
        print("Map initialization failed.")   






if __name__ == "__main__":
    main()