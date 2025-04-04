import numpy as np

class KeyFrame:
    def __init__(self, id, pose, camera_intrinsics, keypoints, descriptors):
        """
        Represents a keyframe in ORB-SLAM.
        
        :param id: Unique identifier for the keyframe.
        :param pose: 4x4 transformation matrix representing camera pose (world to camera).
        :param camera_intrinsics: Intrinsic matrix K (3x3 numpy array).
        :param keypoints: ORB keypoints detected in this keyframe.
        :param descriptors: ORB descriptors corresponding to keypoints.
        """
        self.id = id  # Unique ID for the keyframe
        self.pose = pose  # Camera pose (4x4 transformation matrix)
        self.camera_intrinsics = camera_intrinsics  # Intrinsic matrix K
        self.keypoints = keypoints  # ORB KeyPoints
        self.descriptors = descriptors  # ORB Descriptors
        
        self.map_points = {}  # Dictionary mapping keypoint indices to MapPoint **IDs**
        self.covisibility_graph = {}  # Stores connections to other keyframes
    
    def get_camera_center(self):
        """
        Returns the camera center C from the pose matrix.
        """
        R = self.pose[:3, :3]  # Extract rotation
        t = self.pose[:3, 3]  # Extract translation
        return -np.dot(R.T, t)  # Compute camera center C
    
    def add_map_point(self, keypoint_idx, map_point_id):
        """
        Associates a keypoint with a 3D MapPoint's id.
        
        :param keypoint_idx: Index of the keypoint in the keyframe.
        :param map_point: The corresponding MapPoint global id.
        """
        if map_point_id is None:
            print(f"❌ ERROR: KeyFrame {self.id} tried to store a None map point ID")
        self.map_points[keypoint_idx] = map_point_id # 🔹 Store MapPoint **ID**, not the object
    
    def get_descriptor(self, map_point_id, global_map):
        """
        Returns the descriptor of the keypoint associated with a given MapPoint ID.

        :param map_point_id: The ID of the MapPoint whose descriptor is needed.
        :param global_map: The global map instance to fetch the MapPoint.
        :return: ORB descriptor (numpy array) or None if not found.
        """
        map_point = global_map.get_map_point(map_point_id)
        if map_point:
            return map_point.descriptor  # 🔹 Always return the latest descriptor
        return None
    
    def add_covisibility_link(self, other_keyframe, shared_points):
        """
        Adds a covisibility link between this keyframe and another.
        
        :param other_keyframe: The connected KeyFrame.
        :param shared_points: Number of shared MapPoints.
        """
        self.covisibility_graph[other_keyframe.id] = shared_points
    
    def get_best_covisibility_keyframes(self, min_shared_points=15):
        """
        Returns keyframes with at least `min_shared_points` shared map points.
        
        :param min_shared_points: Minimum number of shared points for a connection.
        :return: List of keyframe IDs.
        """
        return [kf_id for kf_id, shared in self.covisibility_graph.items() if shared >= min_shared_points]
