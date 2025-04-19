import numpy as np
from orbslam.core.keyframe import KeyFrame
from orbslam.core.map_point import MapPoint
from typing import Dict, List, Optional, Set

class Map:
    """
        Represents the global map storing KeyFrames and MapPoints.
    """
    def __init__(self):
        """ Initialize an empty map."""
        # Primary data structures
        self.keyframes = {}  # Dictionary of keyframe_id -> KeyFrame
        self.map_points = {}  # Dictionary of map_point_id -> MapPoint
        self.next_map_point_id = 0  # Counter for generating unique map point IDs


    def add_keyframe(self, keyframe):
        """
        Adds a new keyframe object to the map.
        
        :param keyframe: KeyFrame object to be added.
        """
        self.keyframes[keyframe.id] = keyframe
    
    def add_map_point(self, map_point):
        """  Add a new MapPoint to the map and assign it a unique ID.
        
        Args:
            map_points: MapPoint object to be added.

        Returns:
            The assigned MapPoint ID.
        """
        map_point.id = self.next_map_point_id
        self.map_points[self.next_map_point_id] = map_point
        self.next_map_point_id += 1
        return map_point.id
    
    def get_keyframe(self, keyframe_id) -> Optional['KeyFrame']:
        """
        Retrieves a keyframe by its ID.
        
        Args:
            keyframe_id: ID of the keyframe.
        Returns:
            KeyFrame object or None if not found.
        """
        return self.keyframes.get(keyframe_id, None)
    
    def get_map_point(self, map_point_id) -> Optional['MapPoint']:
        """
        Retrieves a map point by its ID.
        
        Args:
            map_point_id: ID of the map point.
        Returns:
            MapPoint object or None if not found.
        """
        return self.map_points.get(map_point_id, None)
    
    def remove_keyframe(self, keyframe_id):
        """
        Removes a keyframe from the map.
        
        Args:
            keyframe_id: ID of the keyframe to be removed.
        """
        if keyframe_id in self.keyframes:
            del self.keyframes[keyframe_id]
    
    def remove_map_point(self, map_point_id):
        """ Removes a map point from the map and all keyframes that observe it. 
        
        Args:
            map_point_id: ID of the map point to be removed.
        """
        if map_point_id not in self.map_points:
            return
        # Get the map point
        map_point = self.map_points[map_point_id]

        # Remove references from all keyframes that observed it
        for keyframe_id in map_point.keyframes_observed:
            keyframe = self.get_keyframe(keyframe_id)
            if keyframe:
                # Find and remove all associations to this map point
                keypoint_indices = []
                for kp_idx, mp_id in keyframe.map_points.items():
                    if mp_id == map_point_id:
                        keypoint_indices.append(kp_idx)
                
                for kp_idx in keypoint_indices:
                    if kp_idx in keyframe.map_points:
                        del keyframe.map_points[kp_idx]
        # Remove the map point itself
        del self.map_points[map_point_id]

    def update_map_point_descriptor(self, map_point_id): #(Verify)
        """
        Update a map point's descriptor based on all its observations.
        
        Args:
            map_point_id: ID of the map point to update.
        """
        map_point = self.get_map_point(map_point_id)
        if not map_point:
            return
        
        descriptors = []
        
        # Collect descriptors from all keyframes that observe this point
        for keyframe_id in map_point.keyframes_observed:
            keyframe = self.get_keyframe(keyframe_id)
            if keyframe:
                # Find the keypoint index associated with this map point
                for kp_idx, mp_id in keyframe.map_points.items():
                    if mp_id == map_point_id:
                        # Get the descriptor for this keypoint
                        if kp_idx < len(keyframe.descriptors):
                            descriptors.append(keyframe.descriptors[kp_idx])
                        break
        
        # Update the map point's descriptor
        if descriptors:
            map_point.update_descriptor(descriptors)
    
    
    def get_best_covisibility_keyframes(self, keyframe, min_shared_points=15):
        """
        Get a list of keyframes that share at least min_shared_points with the given keyframe.
        
        Args:
            keyframe: The keyframe for which to find covisible keyframes
            min_shared_points: Minimum number of shared points required
            
        Returns:
            List of KeyFrame objects that are covisible with the input keyframe
        """
        covisibility_ids = keyframe.get_best_covisibility_keyframes(min_shared_points)
        return [self.get_keyframe(kf_id) for kf_id in covisibility_ids 
                if kf_id in self.keyframes]



#Finish 2 functions
