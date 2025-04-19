import numpy as np
from typing import List, Set, Optional


class MapPoint:
    """
    Represents a 3D map point in the world coordinate system

    Each map point stores:
    - 3D position in world coordinates
    - Viewing direction (mean of viewing rays)
    - ORB descriptor
    - Observation information for tracking and matching

    """
    def __init__(self, position, descriptor):
        """
        Initialize a map point with its 3D position and descriptor.

        Args:
            position: 3D position in world coordinates (numpy array of shape (3,))
            descriptor: Represntative orb descriptor (numpy array)
        """

        self.id = None  #  ID will be assigned when added to Map
        self.position = np.array(position, dtype=np.float32)  # 3D world position
        self.descriptor = descriptor  # ORB Descriptor
        
        # Viewing information:
        self.viewing_directions = []  # Stores unit vectors of viewing directions
        self.keyframes_observed = set()  # Stores keyframe IDs that observe this point
        
        # Scale invariance limits 
        self.d_min = None  # Minimum scale-invariant observation distance
        self.d_max = None  # Maximum scale-invariant observation distance

        # Tracking Stats:
        self.visible_count = 0 # Number of times the point was visible in tracking
        self.found_count = 0 # Number of times the point was matched in tracking
    
    def add_observation(self, keyframe_id, viewing_direction):
        """
        Adds an observation of this map point from a keyframe.
        
        Args:
            keyframe_id: ID of the KeyFrame observing this point
            viewing_direction: Viewing direction unit vector from camera to point
        """

        self.keyframes_observed.add(keyframe_id)

        # Normalize and add the viewing direction
        norm = np.linalg.norm(viewing_direction)
        if norm > 0:
            self.viewing_directions.append(viewing_direction / norm)
    
    def compute_mean_viewing_direction(self):
        """
        Computes the mean viewing direction of this map point.

        Returns:
            Normalized mean viewing direction vector or None if no observations
        """
        if not self.viewing_directions:
            return None
        mean_direction = np.mean(self.viewing_directions, axis=0)
        norm = np.linalg.norm(mean_direction)

        if norm > 0:
            return mean_direction / norm
        return None
    
    def update_descriptor(self, descriptors):
        """
        Updates the representative ORB descriptor as the one with the minimum 
        Hamming distance to all associated descriptors from keyframes.
        
        Args:
            descriptors: List of Orb descriptors observed for this map point
        """
        if not descriptors or len(descriptors) == 0:
            return
        
        # For a single descriptor, just use it
        if len(descriptors) ==1:
            self.descriptor =descriptors[0]
            return
        
        # Find descriptor with minimum Hamming distance to all others
        best_sum_distance =float('inf')
        best_descriptor = None
    
        for i, desc1 in enumerate(descriptors):
            sum_distance = 0
            for j, desc2 in enumerate(descriptors):
                if i != j:
                    # Compute Hamming distance (count of different bits)
                    # For ORB descriptors which are binary
                    distance = np.count_nonzero(desc1 != desc2)
                    sum_distance += distance
        
            if sum_distance < best_sum_distance:
                best_sum_distance = sum_distance
                best_descriptor = desc1
    
        if best_descriptor is not None:
            self.descriptor = best_descriptor
    
    def set_scale_invariance_limits(self, d_min, d_max):
        """
        Sets the minimum and maximum distances for scale invariance of this point.
        
        Args:
            d_min: Minimum valid observation distance
            d_max: Maximum valid observation distance
        """
        self.d_min = d_min
        self.d_max = d_max
    
    def is_within_observation_range(self, distance):
        """
        Checks if the map point is within the valid scale-invariant observation range.
        
        Args:
            distance: Distance from the camera to the point

        Returns: 
            True if within [d_min, d_max], False otherwise
        """
        if self.d_min is None or self.d_max is None:
            return True  # No restriction
        
        return self.d_min <= distance <= self.d_max
    
    def increase_visible(self):
        """Increase the visibility counter."""
        self.visible_count += 1
    
    def increase_found(self):
        """Increase the successfully matched counter."""
        self.found_count += 1
    
    def get_found_ratio(self):
        """
        Get the ratio of successful matches to visibility.
        
        Returns:
            Ratio of successful matches or 0 if never visible
        """
        if self.visible_count == 0:
            return 0
        return self.found_count / self.visible_count
