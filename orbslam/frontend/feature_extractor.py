import cv2
import numpy as np

class FeatureExtractor:
    def __init__(self,
                 iniThFAST=20,
                 minThFAST=5,
                 nfeatures=2400,
                 nlevels=8,
                 scaleFactor=1.2,
                 grid_rows=4,
                 grid_cols=4,
                 minCornersPerCell=5):
        """
        ORB-SLAM–style feature extractor.
        
        Parameters:
          - iniThFAST: initial FAST threshold.
          - minThFAST: minimum FAST threshold allowed.
          - nfeatures: total number of desired features (e.g. 1000 for lower-resolution images,
                       2000 for higher resolutions such as KITTI).
          - nlevels: number of scale levels (ORB-SLAM uses 8).
          - scaleFactor: scale factor between levels (1.2).
          - grid_rows, grid_cols: number of grid cells per level.
          - minCornersPerCell: we try to extract at least 5 corners per cell.
          
        The extractor first detects FAST corners (with adaptive thresholding per grid cell)
        and then computes the orientation and ORB descriptor on the retained corners.
        """
        self.iniThFAST = iniThFAST
        self.minThFAST = minThFAST
        self.nfeatures = nfeatures
        self.nlevels = nlevels
        self.scaleFactor = scaleFactor
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.minCornersPerCell = minCornersPerCell
        
        # Create an ORB extractor (its "compute" function will later compute orientation and descriptors)
        self.orb = cv2.ORB_create(
            nfeatures=self.nfeatures,
            scaleFactor=self.scaleFactor,
            nlevels=self.nlevels,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=self.iniThFAST  # (Not used in our detection, but required for ORB)
        )
        
    def extract(self, image):
        """
        Extract features following the ORB-SLAM design:
          1. Build an image pyramid (8 levels, scale factor 1.2).
          2. At each level, divide the image into a grid.
          3. In each grid cell, detect FAST corners, lowering the threshold
             until at least 'minCornersPerCell' are found (if possible).
          4. Convert keypoint coordinates back to the original image space,
             assign the octave and scale (size) accordingly.
          5. Finally, compute orientation and descriptors with ORB.
          
        Parameters:
          - image: a grayscale image.
          
        Returns:
          - keypoints: a list of cv2.KeyPoint objects.
          - descriptors: a NumPy array of descriptors.
        """
        if len(image.shape) != 2:
            raise ValueError("Input image must be grayscale")
            
        h, w = image.shape
        keypoints_all = []
        
        # Divide the total features roughly equally among the pyramid levels
        nfeatures_per_level = self.nfeatures // self.nlevels
        
        # Process each pyramid level.
        for level in range(self.nlevels):
            # Compute the scaling factor for this level.
            # Level 0 is original; level i is scaled by s = 1 / (scaleFactor^i)
            s = 1.0 / (self.scaleFactor ** level)
            level_w = int(round(w * s))
            level_h = int(round(h * s))
            
            # Resize the image to form the current pyramid level.
            scaled_img = cv2.resize(image, (level_w, level_h), interpolation=cv2.INTER_LINEAR)
            
            # Determine grid cell size at this level.
            cell_w = level_w // self.grid_cols
            cell_h = level_h // self.grid_rows
            
            # Compute an ideal number of keypoints per cell.
            n_cells = self.grid_rows * self.grid_cols
            # We want at least 'minCornersPerCell' per cell if possible; otherwise, share nfeatures evenly.
            ideal_per_cell = max(self.minCornersPerCell, nfeatures_per_level // n_cells)
            
            # Process each grid cell.
            for i in range(self.grid_rows):
                for j in range(self.grid_cols):
                    # Compute the cell boundaries (make sure the last cell extends to the image border)
                    x_start = j * cell_w
                    y_start = i * cell_h
                    x_end = (j + 1) * cell_w if j < self.grid_cols - 1 else level_w
                    y_end = (i + 1) * cell_h if i < self.grid_rows - 1 else level_h
                    cell_img = scaled_img[y_start:y_end, x_start:x_end]
                    
                    # Adaptive FAST detection: start with the initial threshold and lower it until
                    # we get at least minCornersPerCell (if possible).
                    adaptive_th = self.iniThFAST
                    cell_kps = []
                    while adaptive_th >= self.minThFAST:
                        fast = cv2.FastFeatureDetector_create(
                            threshold=adaptive_th,
                            nonmaxSuppression=True
                        )
                        cell_kps = fast.detect(cell_img, None)
                        if len(cell_kps) >= self.minCornersPerCell:
                            break
                        adaptive_th -= 1
                        
                    # If keypoints were found in the cell, sort them by response and keep at most the ideal number.
                    if cell_kps:
                        cell_kps = sorted(cell_kps, key=lambda kp: kp.response, reverse=True)
                        if len(cell_kps) > ideal_per_cell:
                            cell_kps = cell_kps[:ideal_per_cell]
                        
                        # For each keypoint, adjust its coordinates:
                        # (a) They are relative to the cell image, so add the cell offset.
                        # (b) Convert from the scaled image back to the original image coordinates.
                        for kp in cell_kps:
                            x, y = kp.pt
                            x += x_start
                            y += y_start
                            kp.pt = (x / s, y / s)
                            # Record which pyramid level (octave) this keypoint comes from.
                            kp.octave = level
                            # Set an approximate scale (the patch size increases with level).
                            kp.size = 31 * (self.scaleFactor ** level)
                        
                        keypoints_all.extend(cell_kps)
                        
        # If we have more keypoints than desired, select only the best nfeatures.
        if len(keypoints_all) > self.nfeatures:
            keypoints_all = sorted(keypoints_all, key=lambda kp: kp.response, reverse=True)[:self.nfeatures]
        
        # Finally, compute the orientation and descriptors on the original image.
        keypoints_all, descriptors = self.orb.compute(image, keypoints_all)
        
        return keypoints_all, descriptors