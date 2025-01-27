import numpy as np
from scipy.spatial.transform import Rotation as R
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
        rotation_diff = R.from_matrix(np.dot(current_pose[:3, :3], last_keyframe_pose[:3, :3].T)).magnitude()



        # New keyframe criteria
        if (
            translation_diff > self.translation_threshold
            or rotation_diff > self.rotation_threshold
            or num_feature_matches < self.min_feature_matches
        ):
            return True
        return False

    def add_keyframe(self, pose):
        """
        Adds the current pose to the list of keyframes.
        :param pose: 4x4 pose matrix of the new keyframe.
        """
        self.keyframes.append(pose)

    def get_last_keyframe_pose(self):
        """
        Retrieves the pose of the last keyframe.
        :return: 4x4 pose matrix of the last keyframe.
        """
        if self.keyframes:
            return self.keyframes[-1]
        return None
