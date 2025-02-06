import cv2
import numpy as np

class FeatureExtractor:
    def __init__(self, sample_image, scaleFactor=1.2, nlevels=8, iniThFAST=20, minThFAST=5, grid_size=(8, 8)):
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

        # Initialize CLAHE for contrast enhancement
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def extract(self, image):
        """
        Detects FAST keypoints in a grid-based approach ensuring homogeneous distribution.
        """
        # Apply CLAHE to enhance contrast
        image = self.clahe.apply(image)

        h, w = image.shape
        keypoints = []
        grid_rows, grid_cols = self.grid_size
        cell_h, cell_w = h // grid_rows, w // grid_cols

        for i in range(grid_rows):
            for j in range(grid_cols):
                # Define cell region
                x_start, y_start = j * cell_w, i * cell_h
                x_end, y_end = x_start + cell_w, y_start + cell_h
                cell_img = image[y_start:y_end, x_start:x_end]

                # Detect keypoints in cell with adaptive threshold
                threshold = self.iniThFAST
                detector = cv2.FastFeatureDetector_create(threshold=threshold, nonmaxSuppression=True)
                cell_keypoints = detector.detect(cell_img)

                # If fewer than 5 keypoints, try lowering threshold dynamically
                while len(cell_keypoints) < 5 and threshold >= self.minThFAST:
                    threshold -= 5
                    detector = cv2.FastFeatureDetector_create(threshold=threshold, nonmaxSuppression=True)
                    cell_keypoints = detector.detect(cell_img)

                # Adaptively allocate more keypoints to high-texture areas
                max_kp_per_cell = int((self.nfeatures / (grid_rows * grid_cols)) * 2)  # Dynamic allocation
                cell_keypoints = sorted(cell_keypoints, key=lambda kp: kp.response, reverse=True)[:max_kp_per_cell]

                # Transform coordinates to global image space
                for kp in cell_keypoints:
                    kp.pt = (kp.pt[0] + x_start, kp.pt[1] + y_start)
                    keypoints.append(kp)

        # Compute ORB descriptors before final filtering
        keypoints, descriptors = self.orb.compute(image, keypoints)

        # Keep only the strongest `self.nfeatures` keypoints globally
        keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)[:self.nfeatures]
        return keypoints, descriptors