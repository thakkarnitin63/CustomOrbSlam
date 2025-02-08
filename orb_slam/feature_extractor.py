import cv2
import numpy as np

class FeatureExtractor:
    def __init__(self, sample_image, scaleFactor=1.2, nlevels=8, iniThFAST=15, minThFAST=3, grid_size=(4, 13)):
        """
        Initializes the FeatureExtractor with parameters from the ORB-SLAM paper.
        """
        self.image_size = sample_image.shape[:2]  # (height, width)
        self.scaleFactor = scaleFactor
        self.nlevels = nlevels
        self.iniThFAST = iniThFAST
        self.minThFAST = minThFAST
        self.grid_size = grid_size

        # Define number of features based on image resolution
        if 384 <= self.image_size[0] <= 480 and 512 <= self.image_size[1] <= 752:
            self.nfeatures = 1000
        else:
            self.nfeatures = 2000  # KITTI dataset resolution

        print("Image size:", self.image_size, "-> Using", self.nfeatures, "features.")

        # Initialize ORB descriptor extractor
        self.orb = cv2.ORB_create(
            nfeatures=self.nfeatures,
            scaleFactor=scaleFactor,
            nlevels=nlevels,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=iniThFAST
        )

    def extract(self, image):
        """
        Detects FAST keypoints ensuring even distribution.
        """
        h, w = image.shape
        keypoints = []

        # STEP 1: Detect keypoints in the full image FIRST
        full_keypoints = cv2.FastFeatureDetector_create(threshold=self.iniThFAST, nonmaxSuppression=True).detect(image)

        # STEP 2: Divide the image into a grid and enforce per-cell keypoints
        grid_rows, grid_cols = self.grid_size
        cell_h, cell_w = h // grid_rows, w // grid_cols

        final_keypoints = []
        for i in range(grid_rows):
            for j in range(grid_cols):
                # Define cell region
                x_start, y_start = j * cell_w, i * cell_h
                x_end, y_end = min(x_start + cell_w, w), min(y_start + cell_h, h)

                # Select keypoints that fall inside this cell
                cell_keypoints = [
                    kp for kp in full_keypoints if x_start <= kp.pt[0] < x_end and y_start <= kp.pt[1] < y_end
                ]

                # Sort by response and take the strongest ones
                max_kp_per_cell = max(10, int((self.nfeatures / (grid_rows * grid_cols)) * 2))
                cell_keypoints = sorted(cell_keypoints, key=lambda kp: kp.response, reverse=True)[:max_kp_per_cell]

                # Append final keypoints
                final_keypoints.extend(cell_keypoints)

        # STEP 3: Compute descriptors
        keypoints, descriptors = self.orb.compute(image, final_keypoints)

        # If descriptors are None, return empty
        if descriptors is None:
            print("⚠️ No valid descriptors found.")
            return [], None

        # Ensure keypoints and descriptors count match
        keypoints = keypoints[:len(descriptors)]

        print(f"Final keypoints count: {len(keypoints)}, Descriptors shape: {descriptors.shape}")
        return keypoints, descriptors
