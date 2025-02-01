import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree
import gtsam
from gtsam import symbol
from gtsam.utils import plot
from scipy.spatial.transform import Rotation as R


class SparseMapping:
    def __init__(self, camera_matrix):
        """
        Initializes the sparse mapping module.
        
        Args:
            camera_matrix (np.ndarray): The camera intrinsic matrix.
        """
        self.camera_matrix = camera_matrix
        self.map_points = []  # List to store MapPoint objects
        self.keyframes = []   # List to store KeyFrame objects
        self.kd_tree = None

    def add_keyframe(self, keyframe):
        """Adds a keyframe to the mapping module."""
        self.keyframes.append(keyframe)


    def update_kd_tree(self):
        """ Rebuilds the KD-Tree after adding new MapPoints. """
        if self.map_points:
            self.kd_tree = KDTree([mp.position for mp in self.map_points])


    def find_nearest_map_point(self, new_point, threshold=0.05):
        """
        Searches for an existing MapPoint near the new point.
        
        Args:
            new_point (np.ndarray): 3D point to check.
            threshold (float): Distance threshold.
            
        Returns:
            MapPoint or None: The nearest MapPoint if within threshold; otherwise, None.
        """
        if self.kd_tree is None or not self.map_points:
            return None
        dist, index = self.kd_tree.query(new_point)
        return self.map_points[index] if dist < threshold else None
    


    def triangulate_points(self, keyframe1, keyframe2, matches, R, t,
                             dbscan_eps=7, dbscan_min_samples=5):
        """
        Triangulates 3D points from matches between two keyframes.
        
        Args:
            keyframe1: First keyframe.
            keyframe2: Second keyframe.
            matches: List of matched keypoints between the keyframes.
            R (np.ndarray): Rotation matrix from keyframe1 to keyframe2.
            t (np.ndarray): Translation vector from keyframe1 to keyframe2.
            dbscan_eps (float): DBSCAN neighborhood radius.
            dbscan_min_samples (int): Minimum samples for DBSCAN core point.
            
        Returns:
            list: Newly added MapPoint objects.
        """
        # Extract matched keypoints
        points1 = np.float32([keyframe1.keypoints[m.queryIdx].pt for m in matches])
        points2 = np.float32([keyframe2.keypoints[m.trainIdx].pt for m in matches])

        # Projection matrices
        P1 = self.camera_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = self.camera_matrix @ np.hstack((R, t))

        # Triangulate
        points_4d = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
        points_3d = (points_4d[:3, :] / points_4d[3, :]).T

        # Filter points in front of the camera
        valid_idx = points_3d[:, 2] > 0
        points_3d = points_3d[valid_idx]
        valid_points1 = points1[valid_idx]
        valid_points2 = points2[valid_idx]

        # DBSCAN filtering to remove noise
        clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(points_3d)
        labels = clustering.labels_
        idx = (labels != -1)
        filtered_points_3d = points_3d[idx]
        filtered_points1 = valid_points1[idx]
        filtered_points2 = valid_points2[idx]

        new_map_points = []
        for i, point in enumerate(filtered_points_3d):
            existing_mp = self.find_nearest_map_point(point)
            if existing_mp is None:
                # Use the descriptor from keyframe1 corresponding to this point.
                descriptor = keyframe1.descriptors[i] if i < len(keyframe1.descriptors) else None

                map_point = MapPoint(
                    position=point,
                    observations={keyframe1.id: filtered_points1[i], keyframe2.id: filtered_points2[i]},
                    descriptors = descriptor
                )
                new_map_points.append(map_point)
                self.map_points.append(map_point)
            else:
                existing_mp.add_observation(keyframe1.id, filtered_points1[i])
                existing_mp.add_observation(keyframe2.id, filtered_points2[i])

        self.update_kd_tree()
        keyframe1.add_observation(new_map_points, filtered_points1)
        keyframe2.add_observation(new_map_points, filtered_points2)
        return new_map_points

class MapPoint:
    def __init__(self, position, observations=None, descriptors=None):
        """
        Represents a 3D point in the map.
        
        Attributes:
            position (np.ndarray): 3D coordinates of the point.
            observations (dict): Dictionary mapping keyframe IDs to 2D points in the image.
            descriptors (np.ndarray): Optional aggregated descriptor for the point.
        """
        self.position = np.array(position)
        self.observations = observations if observations else {}
        self.descriptors = descriptors

    def add_observation(self, keyframe_id, keypoint):
        """
        Adds a new observation to the MapPoint.
        
        Args:
            keyframe_id (int): Identifier of the keyframe.
            keypoint: 2D point in the image.
        """
        self.observations[keyframe_id] = keypoint

    def merge_descriptors(self, new_descriptor): # OPTIONAL
        """ (Optional) Merges a new descriptor into the current descriptor. """
        if self.descriptors is None:
            self.descriptors = new_descriptor
        else:
            self.descriptors = (self.descriptors + new_descriptor) / 2  # Simple averaging

class KeyFrame:
    def __init__(self, id, pose, keypoints, descriptors):
        """
        Represents a keyframe in the SLAM system.
        :param id: Unique identifier for the keyframe.
        :param pose: 4x4 pose matrix of the keyframe.
        :param keypoints: Keypoints detected in this keyframe.
        :param descriptors: Descriptors associated with the keypoints.
        """
        self.id = id
        self.pose = pose  # 4x4 transformation matrix
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.observations = {}  # MapPoint observations

    def add_observation(self, map_points, points_2d):
        """
        Adds observations of 3D points to this keyframe.
        :param map_points: List of MapPoint objects.
        :param points_2d: Corresponding 2D points in the image.
        """
        for map_point, point_2d in zip(map_points, points_2d):
            self.observations[map_point] = point_2d

    def remove_observation(self, map_point):
        """
        Removes a specific observation of a map point.
        :param map_point: The MapPoint object to be removed.
        """
        if map_point in self.observations:
            del self.observations[map_point]

    def num_observations(self):
        """
        Returns the number of map points observed by this keyframe.
        :return: Number of observations.
        """
        return len(self.observations)
