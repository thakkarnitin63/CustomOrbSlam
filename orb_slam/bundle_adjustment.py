import numpy as np
import g2o
import copy

class BundleAdjustment(g2o.SparseOptimizer):
    def __init__(self, K, iterations=10):
        """
        Implements Bundle Adjustment using g2o in Python.

        Parameters:
          - K: Camera intrinsic matrix (3x3 numpy array)
          - iterations: Number of optimization iterations (default: 10)
        """
        super().__init__()
        self.K = K
        self.iterations = iterations

        # Initialize solver with Levenberg-Marquardt optimization
        solver = g2o.BlockSolverSE3(g2o.LinearSolverDenseSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)

    def optimize_full(self, map_instance):
        """
        Full Bundle Adjustment: Optimizes all keyframes and map points.
        """
        self.clear()

        # --- Add Camera Parameters ---
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        cam_params = g2o.CameraParameters(fx, (cx, cy), 0)
        cam_params.set_id(0)
        super().add_parameter(cam_params)

        # --- Add Keyframes (Poses) ---
        for keyframe in map_instance.keyframes.values():
            self.add_pose(keyframe.id, keyframe.pose, cam_params, fixed=(keyframe.id == 0))

        # --- Add 3D Map Points ---
        for map_point in map_instance.map_points.values():
            self.add_point(map_point.id, map_point.position, fixed=False)

        # --- Add Observations (Edges) ---
        for keyframe in map_instance.keyframes.values():
            for keypoint_idx, map_point in keyframe.map_points.items():
                self.add_edge(map_point.id, keyframe.id, keyframe.keypoints[keypoint_idx].pt)

        # --- Optimize ---
        super().initialize_optimization()
        print("\n🔹 Performing **Full Bundle Adjustment**")
        super().optimize(self.iterations)

        # --- Update the Map ---
        for keyframe in map_instance.keyframes.values():
            keyframe.pose = self.get_pose(keyframe.id)

        for map_point in map_instance.map_points.values():
            map_point.position = self.get_point(map_point.id)

    def optimize_local(self, map_instance, recent_keyframes):
        """
        Local Bundle Adjustment: Optimizes the last few keyframes and their map points.
        """
        self.clear()

        # --- Add Camera Parameters ---
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        cam_params = g2o.CameraParameters(fx, (cx, cy), 0)
        cam_params.set_id(0)
        super().add_parameter(cam_params)

        # --- Add Recent Keyframes ---
        for keyframe in recent_keyframes:
            self.add_pose(keyframe.id, keyframe.pose, cam_params, fixed=False)

        # --- Add Connected Keyframes (Fixed) ---
        for keyframe in recent_keyframes:
            for neighbor_id in keyframe.get_best_covisibility_keyframes(min_shared_points=20):
                neighbor = map_instance.get_keyframe(neighbor_id)
                self.add_pose(neighbor.id, neighbor.pose, cam_params, fixed=True)

        # --- Add Relevant 3D Map Points ---
        map_points_to_optimize = set()
        for keyframe in recent_keyframes:
            for keypoint_idx, map_point in keyframe.map_points.items():
                map_points_to_optimize.add(map_point)

        for map_point in map_points_to_optimize:
            self.add_point(map_point.id, map_point.position, fixed=False)

        # --- Add Observations (Edges) ---
        for keyframe in recent_keyframes:
            for keypoint_idx, map_point in keyframe.map_points.items():
                self.add_edge(map_point.id, keyframe.id, keyframe.keypoints[keypoint_idx].pt)

        # --- Optimize ---
        super().initialize_optimization()
        print("\n🔹 Performing **Local Bundle Adjustment**")
        super().optimize(self.iterations)

        # --- Update the Map ---
        for keyframe in recent_keyframes:
            keyframe.pose = self.get_pose(keyframe.id)

        for map_point in map_points_to_optimize:
            map_point.position = self.get_point(map_point.id)

    def optimize_pose(self, keyframe):
        """
        Motion-only Bundle Adjustment: Optimizes the pose of a single keyframe.
        """
        self.clear()

        # --- Add Camera Parameters ---
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        cam_params = g2o.CameraParameters(fx, (cx, cy), 0)
        cam_params.set_id(0)
        super().add_parameter(cam_params)

        # --- Add the Pose (Single Keyframe) ---
        self.add_pose(keyframe.id, keyframe.pose, cam_params, fixed=False)

        # --- Add 3D Map Points (Fixed) ---
        for keypoint_idx, map_point in keyframe.map_points.items():
            self.add_point(map_point.id, map_point.position, fixed=True)

        # --- Add Observations (Edges) ---
        for keypoint_idx, map_point in keyframe.map_points.items():
            self.add_edge(map_point.id, keyframe.id, keyframe.keypoints[keypoint_idx].pt)

        # --- Optimize ---
        super().initialize_optimization()
        print("\n🔹 Performing **Pose-only Bundle Adjustment**")
        super().optimize(self.iterations)

        # --- Update the Pose ---
        keyframe.pose = self.get_pose(keyframe.id)

    def add_pose(self, pose_id, pose, cam, fixed=False):
        """ Adds a pose (camera pose) to the optimizer. """
        
        # Extract rotation matrix (R) and translation vector (t)
        R = pose[:3, :3]
        t = pose[:3, 3]

        # Convert to g2o SE3Quat (rotation + translation)
        se3 = g2o.SE3Quat(R, t)

        # Create SBACam object
        sbacam = g2o.SBACam(se3)
        sbacam.set_cam(self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2], 0)

        # Add camera vertex to optimizer
        v_se3 = g2o.VertexCam()
        v_se3.set_id(pose_id * 2)  
        v_se3.set_estimate(sbacam)
        v_se3.set_fixed(fixed)
        super().add_vertex(v_se3)



    def add_point(self, point_id, point, fixed=False, marginalized=True):
        """ Adds a 3D point to the optimizer. """
        v_p = g2o.VertexPointXYZ()
        v_p.set_id(point_id * 2 + 1)
        v_p.set_estimate(point)
        v_p.set_marginalized(marginalized)
        v_p.set_fixed(fixed)
        super().add_vertex(v_p)

    def add_edge(self, point_id, pose_id, measurement):
        """ Adds an edge (observation constraint) between keyframe and 3D point. """
        edge = g2o.EdgeProjectP2MC()
        edge.set_vertex(0, self.vertex(point_id * 2 + 1))
        edge.set_vertex(1, self.vertex(pose_id * 2))
        edge.set_measurement(measurement)
        edge.set_information(10 * np.identity(2))

        # Apply Robust Kernel
        edge.set_robust_kernel(g2o.RobustKernelHuber(np.sqrt(5.991)))  
        super().add_edge(edge)

    def get_pose(self, pose_id):
        """ Retrieves the optimized pose of a keyframe. """
        estimate = self.vertex(pose_id * 2).estimate()

        # ✅ Convert Quaternion to Rotation Matrix
        R = estimate.rotation().matrix()  # Convert to 3x3 rotation matrix
        t = estimate.translation().reshape(3, 1)  # Ensure it's a (3,1) vector

        # ✅ Concatenate Rotation and Translation to form 4x4 Pose Matrix
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = t.flatten()  # Flatten ensures correct dimensions

        return pose


    def get_point(self, point_id):
        """ Retrieves the optimized 3D point. """
        return self.vertex(point_id * 2 + 1).estimate()
