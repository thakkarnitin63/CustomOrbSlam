import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree

class SparseMapping:
    def __init__(self, camera_matrix):
        self.camera_matrix = camera_matrix
        self.map_points = []  # List to store MapPoint objects
        self.keyframes = []   # List to store KeyFrame objects
        self.kd_tree = None

    def add_keyframe(self, keyframe):
        self.keyframes.append(keyframe)
        # print(f"Added keyframe at frame {keyframe.id}.")

    def update_kd_tree(self):
        """ Rebuilds the KD-Tree after adding new MapPoints. """
        if len(self.map_points) > 0:
            self.kd_tree = KDTree([mp.position for mp in self.map_points])


    def find_nearest_map_point(self, new_point, threshold=0.05):
        """
        Searches for a nearby existing MapPoint using KD-Tree.
        :param new_point: The 3D coordinates of the point to check.
        :param threshold: Distance threshold to consider a duplicate.
        :return: Closest MapPoint if found, else None.
        """
        if self.kd_tree is None or len(self.map_points) == 0:
            return None
        
        dist, index = self.kd_tree.query(new_point)
        if dist < threshold:
            return self.map_points[index]
        return None

    def triangulate_points(self, keyframe1, keyframe2, matches, R, t, dbscan_eps=7, dbscan_min_samples=5):
        """
        Triangulates 3D points from matched keypoints and relative pose between two keyframes.
        Applies DBSCAN clustering to filter noise and prevents duplicate MapPoint creation.

        :param keyframe1: First KeyFrame object.
        :param keyframe2: Second KeyFrame object.
        :param matches: List of matched keypoints between the two keyframes.
        :param R: Rotation matrix from keyframe1 to keyframe2.
        :param t: Translation vector from keyframe1 to keyframe2.
        :param dbscan_eps: DBSCAN neighborhood radius.
        :param dbscan_min_samples: Minimum samples for a point to be considered a core point in DBSCAN.
        :return: List of newly added MapPoint objects.
        """
        # Extract matched keypoints
        points1 = np.float32([keyframe1.keypoints[m.queryIdx].pt for m in matches])
        points2 = np.float32([keyframe2.keypoints[m.trainIdx].pt for m in matches])

        # Projection matrices
        P1 = np.dot(self.camera_matrix, np.hstack((np.eye(3), np.zeros((3, 1)))))
        P2 = np.dot(self.camera_matrix, np.hstack((R, t)))

        # Triangulate points
        points_4d = cv2.triangulatePoints(P1, P2, points1.T, points2.T)

        # Convert to inhomogeneous coordinates
        points_3d = points_4d[:3, :] / points_4d[3, :]
        points_3d = points_3d.T  # Shape: (N, 3)

        # Filter points in front of both cameras
        valid_indices = (points_3d[:, 2] > 0)  # Z > 0 for front-of-camera points
        points_3d = points_3d[valid_indices]
        valid_points1 = points1[valid_indices]
        valid_points2 = points2[valid_indices]

        # Apply DBSCAN to filter out noisy points
        clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(points_3d)
        labels = clustering.labels_

        # Retain only non-noise points
        valid_indices = np.where(labels != -1)[0]
        filtered_points_3d = points_3d[valid_indices]
        filtered_points1 = valid_points1[valid_indices]
        filtered_points2 = valid_points2[valid_indices]

        # Create MapPoint objects
        new_map_points = []
        for i, point in enumerate(filtered_points_3d):
            existing_mp = self.find_nearest_map_point(point)
            if existing_mp is None:  # No duplicate found, create a new MapPoint
                map_point = MapPoint(
                    position=point,
                    observations={keyframe1.id: filtered_points1[i], keyframe2.id: filtered_points2[i]},
                    descriptors=None  # Add descriptor pooling if available
                )
                new_map_points.append(map_point)
                self.map_points.append(map_point)
            else:
                # (Optional) Merge observations if close enough
                existing_mp.add_observation(keyframe1.id, filtered_points1[i])
                existing_mp.add_observation(keyframe2.id, filtered_points2[i])

        # Update KD-Tree after adding new MapPoints
        self.update_kd_tree()

        # print(f"Triangulated {len(new_map_points)} valid 3D points between KeyFrame {keyframe1.id} and {keyframe2.id}.")

        # Add observations to KeyFrames
        keyframe1.add_observation(new_map_points, filtered_points1)
        keyframe2.add_observation(new_map_points, filtered_points2)

        return new_map_points

class MapPoint:
    def __init__(self, position, observations=None, descriptors=None):
        """
        Represents a 3D point in the map.
        :param position: 3D coordinates of the point.
        :param observations: Dictionary of {keyframe_id: 2D point in the image}.
        :param descriptors: Optional aggregated descriptor for the point.
        """
        self.position = np.array(position)
        self.observations = observations if observations else {}
        self.descriptors = descriptors

    def add_observation(self, keyframe_id, keypoint):
        """Adds a new observation to the MapPoint."""
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
