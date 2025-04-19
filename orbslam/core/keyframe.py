import numpy as np
from typing import Dict, List, Optional

class KeyFrame:
    """
    A keyframe stores:
    - Camera pose
    - Camera intrinsic parameters
    - ORB features (keypoints and descriptors)
    - Connections to map points
    - Connections to other keyframes in the covisibility graph
    """
    def __init__(self, id, pose, camera_intrinsics, keypoints, descriptors):
        """
        Initialize a keyframe.

        Args:
            id: Unique identifier for the keyframe.
            pose: 4x4 transformation matrix representing camera pose (world to camera).
            camera_intrinsics: Intrinsic Matrix K (3x3 numpy array).
            keypoints: Orb keypoints detected in this keyframe.
            descriptros: Orb descriptors corresponding to keypoints. 
        """
        self.id = id  # Unique ID for the keyframe
        self.pose = pose  # Camera pose (4x4 transformation matrix)
        self.camera_intrinsics = camera_intrinsics  # Intrinsic matrix K
        self.keypoints = keypoints  # ORB KeyPoints
        self.descriptors = descriptors  # ORB Descriptors
        
      # Map point associations (keypoint index 2D point -> 3D map point ID )
        self.map_points = {}
        
        # Covisibility graph (keyframe ID -> number of shared points)
        self.covisibility_graph = {}
    
    def get_camera_center(self):
        """
        Returns the camera center C in world coordinates.

        Returns: 
            3D position vector of camera center
        """
        R = self.pose[:3, :3]  # Extract rotation
        t = self.pose[:3, 3]  # Extract translation
        return -np.dot(R.T, t)  # Camera center C = -R^T * t
    
    def add_map_point(self, keypoint_idx, map_point_id):
        """
        Associates a keypoint with a 3D MapPoint ID.
        
        Args:
            keypoints_idx: Index of keypoint in the keyframe.
            map_point_id: The corresponding MapPoint ID.
        """
        if map_point_id is None:
            print(f"ERROR: KeyFrame {self.id} tried to store a None map point ID")
            return
            
        self.map_points[keypoint_idx] = map_point_id
    
    def get_descriptor(self, map_point_id, global_map):
        """
        Returns the descriptor of the keypoint associated with a given MapPoint ID.

        Args:
            map_point_id: The ID of the MapPoint whose descriptor is needed.
            global_map: The global map instance to fetch the MapPoint.
            
        Returns:
            ORB descriptor (numpy array) or None if not found.
        """
        map_point = global_map.get_map_point(map_point_id)
        if map_point:
            return map_point.descriptor  # Return the descriptor from the map point
        return None
    
    def add_covisibility_link(self, other_keyframe_id, shared_points):
        """
        Adds a covisibility link between this keyframe and another.
        
        Args:
            other_keyframe_id: The ID of the connected KeyFrame.
            shared_points: Number of shared MapPoints.
        """
        self.covisibility_graph[other_keyframe_id] = shared_points

    
    def get_best_covisibility_keyframes(self, min_shared_points=15):
        """
        Returns keyframes with at least `min_shared_points` shared map points.
        
        Args:
            min_shared_points: Minimum number of shared points for a connection.
            
        Returns:
            List of keyframe IDs.
        """
        return [kf_id for kf_id, shared in self.covisibility_graph.items() 
                if shared >= min_shared_points]


    def update_connections(self, map_instance):
        """
        Updates covisibility graph links based on shared map points with other keyframes.
        
        Args:
            map_instance: The global map instance
            
        Returns:
            Number of keyframes with strong connections (>=15 shared points)
        """
        # Get my map points
        my_map_points = set(self.map_points.values())
        
        # Check connections with all other keyframes
        connections = {}
        for kf_id, kf in map_instance.keyframes.items():
            if kf_id == self.id:
                continue  # Skip self
            
            # Get map points in the other keyframe
            other_map_points = set(kf.map_points.values())
            
            # Count shared points
            shared_points = len(my_map_points.intersection(other_map_points))
            
            if shared_points > 0:
                connections[kf_id] = shared_points
        
        # Update covisibility graph
        self.covisibility_graph = connections
        
        # Count strong connections
        strong_connections = len([kf_id for kf_id, count in connections.items() 
                                 if count >= 15])
        
        return strong_connections