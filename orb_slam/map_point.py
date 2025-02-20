import numpy as np

class MapPoint:
    def __init__(self, id, position, descriptor):
        """
        Represents a 3D map point in the world coordinate system.
        
        :param id: Unique identifier for the map point.
        :param position: 3D position (numpy array of shape (3,)).
        :param descriptor: Representative ORB descriptor (numpy array of shape (32,)).
        """
        self.id = id  # Unique ID for the map point
        self.position = np.array(position, dtype=np.float32)  # 3D world position
        self.descriptor = descriptor  # ORB Descriptor
        
        self.viewing_directions = []  # Stores unit vectors of viewing directions
        self.keyframes_observed = set()  # Stores keyframes that observe this point
        
        self.d_min = None  # Minimum scale-invariant observation distance
        self.d_max = None  # Maximum scale-invariant observation distance
    
    def add_observation(self, keyframe, viewing_direction):
        """
        Adds an observation of this map point from a keyframe.
        
        :param keyframe: KeyFrame instance that observes this point.
        :param viewing_direction: Viewing direction unit vector (numpy array of shape (3,)).
        """
        self.keyframes_observed.add(keyframe)
        self.viewing_directions.append(viewing_direction / np.linalg.norm(viewing_direction))  # Normalize
    
    def compute_mean_viewing_direction(self):
        """
        Computes the mean viewing direction of this map point.
        """
        if len(self.viewing_directions) > 0:
            mean_direction = np.mean(self.viewing_directions, axis=0)
            return mean_direction / np.linalg.norm(mean_direction)  # Normalize
        return None
    
    def update_descriptor(self):
        """
        Updates the representative ORB descriptor as the one with the minimum Hamming distance
        among all associated descriptors from keyframes.
        """
        if len(self.keyframes_observed) == 0:
            return
        
        # Collect all descriptors from keyframes
        descriptors = np.array([kf.get_descriptor(self) for kf in self.keyframes_observed if kf.get_descriptor(self) is not None])
        if len(descriptors) == 0:
            return
        
        # Compute the best descriptor (min Hamming distance to all others)
        best_descriptor = min(descriptors, key=lambda d: sum(np.count_nonzero(d != desc) for desc in descriptors))
        self.descriptor = best_descriptor
    
    def set_scale_invariance_limits(self, d_min, d_max):
        """
        Sets the minimum and maximum distances for scale invariance of this point.
        
        :param d_min: Minimum valid observation distance.
        :param d_max: Maximum valid observation distance.
        """
        self.d_min = d_min
        self.d_max = d_max
    
    def is_within_observation_range(self, distance):
        """
        Checks if the map point is within the valid scale-invariant observation range.
        
        :param distance: Distance from the camera to the point.
        :return: True if within [d_min, d_max], False otherwise.
        """
        if self.d_min is None or self.d_max is None:
            return True  # No restriction
        return self.d_min <= distance <= self.d_max
