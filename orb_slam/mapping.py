class MapPoint:
    def __init__(self, position):
        self.position = position   # 3D coordinates of the point in the world
        self.observations = {}   # Dictionary to store observations of this point
                                 # Format: {keyframe_id: (x, y)}
        self.visible = True      # Whether the point is currently visible in any keyframe
        self.times_observed = 0  # Number of times this point has been observed

    def add_observation(self, keyframe_id, point_2d):
        self.observations[keyframe_id] = point_2d
        self.times_observed += 1

    def remove_observation(self, keyframe_id):
        if keyframe_id in self.observations:
            del self.observations[keyframe_id]
            self.times_observed -= 1
