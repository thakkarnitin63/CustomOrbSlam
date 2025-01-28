import numpy as np
from scipy.spatial.transform import Rotation as R
from collections import deque

class KeyframeManager:
    def __init__(self, translation_threshold=2.0, rotation_threshold=np.deg2rad(10), min_feature_matches=100):
        """
        Initializes the keyframe manager with thresholds for selection criteria.
        :param translation_threshold: Minimum translation change required to add a keyframe.
        :param rotation_threshold: Minimum rotation change (in radians) required to add a keyframe.
        :param min_feature_matches: Minimum number of feature matches required to avoid adding a new keyframe.
        """
        self.translation_threshold = translation_threshold
        self.rotation_threshold = rotation_threshold
        self.min_feature_matches = min_feature_matches
        self.keyframes = []  # Stores keyframe poses
        self.recent_keyframes = deque(maxlen=5)  # Store recent keyframes for optimization


    def is_new_keyframe(self, current_pose, last_keyframe_pose, num_feature_matches):
        """
        Checks whether the current frame should be selected as a new keyframe.
        :param current_pose: 4x4 pose matrix of the current frame.
        :param last_keyframe_pose: 4x4 pose matrix of the last keyframe.
        :param num_feature_matches: Number of feature matches with the last keyframe.
        :return: True if the current frame should be a new keyframe, False otherwise.
        """
        # Translation and rotation change
        translation_vector = current_pose[:3, 3] - last_keyframe_pose[:3, 3]
        translation_diff = np.linalg.norm(translation_vector)

        
        # Calculate rotation difference
        relative_rotation = np.dot(current_pose[:3, :3], last_keyframe_pose[:3, :3].T)
        rotation_diff = R.from_matrix(relative_rotation).magnitude()



        return (
            translation_diff > self.translation_threshold
            or rotation_diff > self.rotation_threshold
            or num_feature_matches < self.min_feature_matches
        )

    def add_keyframe(self, keyframe):
        """
        Adds the current pose to the list of keyframes.
        :param keyframe: KeyFrame object containing pose, keypoints, and descriptors
        """
        self.keyframes.append(keyframe)
        self.recent_keyframes.append(keyframe)

    def get_last_keyframe(self):
        """
        Retrieves the most recently added keyframe.
        :return: The last KeyFrame object or None if no keyframes exist.
        """
        return self.keyframes[-1] if self.keyframes else None
    

    def get_recent_keyframes(self):
        """
        Retrieves the most recent keyframes for optimization.
        :return: List of recent KeyFrame objects.
        """
        return list(self.recent_keyframes)
    
    def query_keyframes_in_radius(self, position, radius):
        """
        Retrieves keyframes within a spatial radius.
        :param position: 3D position to query around.
        :param radius: Spatial radius for the query.
        :return: List of KeyFrame objects within the radius.
        """
        keyframes_in_radius = []
        for keyframe in self.keyframes:
            keyframe_position = keyframe.pose[:3, 3]
            distance = np.linalg.norm(keyframe_position - position)
            if distance <= radius:
                keyframes_in_radius.append(keyframe)
        return keyframes_in_radius
