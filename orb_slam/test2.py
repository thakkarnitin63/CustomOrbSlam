import numpy as np
import cv2
import matplotlib.pyplot as plt

from orb_slam.image_loader import ImageLoader
from orb_slam.feature_extractor import FeatureExtractor
from orb_slam.feature_matcher import FeatureMatcher




def draw_grid_on_image(image, grid_size=(4, 5)):
    """
    Draws a grid overlay on the given image based on the specified grid size.
    
    :param image_path: Path to the image file.
    :param grid_size: Tuple (rows, columns) defining the grid structure.
    """
    # Load the image


    # Get image dimensions
    h, w = image.shape
    grid_rows, grid_cols = grid_size

    # Compute cell dimensions
    cell_h, cell_w = h // grid_rows, w // grid_cols

    # Convert image to BGR for colored grid overlay
    image_with_grid = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Draw the grid lines
    for i in range(1, grid_rows):
        y = i * cell_h
        cv2.line(image_with_grid, (0, y), (w, y), (0, 0, 255), 1)  # Horizontal grid lines

    for j in range(1, grid_cols):
        x = j * cell_w
        cv2.line(image_with_grid, (x, 0), (x, h), (0, 0, 255), 1)  # Vertical grid lines

    # Display the image with grid overlay
    plt.figure(figsize=(12, 6))
    plt.imshow(image_with_grid)
    plt.title("Image with Grid Overlay")
    plt.axis("off")
    plt.show()

# Example usage with uploaded image

def main():
    # Paths (adjust these paths as needed)
    sequence_path = '/home/nitin/NitinWs/CustomOrbSlam/data/dataset/sequences/00'
    ground_truth_file = '/home/nitin/NitinWs/CustomOrbSlam/data/dataset/poses/00.txt'

    # Initialize system components
    image_loader = ImageLoader(sequence_path)
    image1 = image_loader.load_image(0)
    image2 = image_loader.load_image(1)
    # draw_grid_on_image(image1, grid_size=(6,18))
    feature_extractor = FeatureExtractor()
    matcher = FeatureMatcher()

    keypoints1, descriptors1 = feature_extractor.extract(image1)
    keypoints2, descriptors2 = feature_extractor.extract(image2)

    img_with_kp = cv2.drawKeypoints(image1, keypoints1, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # print(f"Keypoints {len(keypoints1)} Descriptors {len(descriptors1)}")
    # keypoints2, descriptors2 = feature_extractor.extract(image1)
    cv2.imshow("Image", img_with_kp)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print(f"Number of keypoints in Image 1: {len(keypoints1)}")
    print(f"Number of keypoints in Image 2: {len(keypoints2)}")
    
    matches = matcher.match(descriptors1, descriptors2)
    filtered_matches = matcher.filter_matches(matches, keypoints1, keypoints2)

    print(f"Number of matches: {len(matches)}")
    print(f"Filtered Matches: {len(filtered_matches)}")
    matcher.visualize_matches(image1, image2, keypoints1, keypoints2, filtered_matches)





if __name__ == "__main__":
    main()