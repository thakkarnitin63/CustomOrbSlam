import os
import cv2
import numpy as np

class ImageLoader:
    def __init__(self, sequence_path):
        """
        Initialize the ImageLoader with the path to a KITTI sequence.
        :param sequence_path: Path to KITTI sequence directory
        """
        self.sequence_path = sequence_path
        self.image_dir = os.path.join(sequence_path, 'image_0')
        self.timestamps = self._load_timestamps()
        self.calibration = self._load_calibration()


    def _load_timestamps(self):
        """
        Loads timestamps from times.txt.

        : return : List of timestamps
        """
        timestamps_path = os.path.join(self.sequence_path, 'times.txt')
        with open(timestamps_path, 'r') as f:
            timestamps = [float(line.strip()) for line in f]
        return timestamps
    
    def _load_calibration(self):
        """
        Loads calibration parameters from calib.txt.
        :return: Dictionary containing calibration parameters.
        """
        calib_path = os.path.join(self.sequence_path, 'calib.txt')
        calib = {}
        with open(calib_path, 'r') as f:
            for line in f:
                key, value = line.split(':', 1)
                # Convert the string of numbers into a numpy array
                projection_matrix = np.array([float(x) for x in value.split()])
                # Reshape into a 3x4 matrix
                projection_matrix = projection_matrix.reshape(3, 4)
                # Extract the intrinsic matrix K (3x3)
                K = projection_matrix[:, :3]
                calib[key] = K
        return calib

    def load_image(self, frame_id):
        """
        Load the image corresponding to the given frame idx.

        :param frame_id: index  of the frame to load.
        : return : grayscale image as Numpy array
        """
        image_path = os.path.join(self.image_dir, f"{frame_id:06d}.png")
        image =cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Image at path '{image_path}' could not be loaded")
        return image
    
    def get_timestamp(self, frame_id):
        """
        Retrieves the timestamp for given frame index.
        :param frame_id: index of the frame.
        :return: Timestamp as float.
        """

        return self.timestamps[frame_id]


    