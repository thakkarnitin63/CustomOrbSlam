class Map:
    def __init__(self):
        """
        Represents the global map storing KeyFrames and MapPoints.
        """
        self.keyframes = {}  # Dictionary of keyframe_id -> KeyFrame
        self.map_points = {}  # Dictionary of map_point_id -> MapPoint
    
    def add_keyframe(self, keyframe):
        """
        Adds a new keyframe to the map.
        
        :param keyframe: KeyFrame object to be added.
        """
        self.keyframes[keyframe.id] = keyframe
    
    def add_map_point(self, map_point):
        """
        Adds a new map point to the map.
        
        :param map_point: MapPoint object to be added.
        """
        self.map_points[map_point.id] = map_point
    
    def get_keyframe(self, keyframe_id):
        """
        Retrieves a keyframe by its ID.
        
        :param keyframe_id: ID of the keyframe.
        :return: KeyFrame object or None if not found.
        """
        return self.keyframes.get(keyframe_id, None)
    
    def get_map_point(self, map_point_id):
        """
        Retrieves a map point by its ID.
        
        :param map_point_id: ID of the map point.
        :return: MapPoint object or None if not found.
        """
        return self.map_points.get(map_point_id, None)
    
    def remove_keyframe(self, keyframe_id):
        """
        Removes a keyframe from the map.
        
        :param keyframe_id: ID of the keyframe to be removed.
        """
        if keyframe_id in self.keyframes:
            del self.keyframes[keyframe_id]
    
    def remove_map_point(self, map_point_id):
        """
        Removes a map point from the map.
        
        :param map_point_id: ID of the map point to be removed.
        """
        if map_point_id in self.map_points:
            del self.map_points[map_point_id]
    
    def get_best_covisibility_keyframes(self, keyframe, min_shared_points=15):
        """
        Returns a list of the best covisibility keyframes for a given keyframe.
        
        :param keyframe: The keyframe for which we find covisibility keyframes.
        :param min_shared_points: Minimum shared MapPoints for connection.
        :return: List of best covisibility KeyFrame objects.
        """
        covisibility_ids = keyframe.get_best_covisibility_keyframes(min_shared_points)
        return [self.get_keyframe(kf_id) for kf_id in covisibility_ids if kf_id in self.keyframes]
