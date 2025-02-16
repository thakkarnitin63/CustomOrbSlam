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

    def optimize(self, init_result):
        """
        Performs Bundle Adjustment optimization.

        Parameters:
          - init_result: Dictionary with:
            - 'R': 3x3 rotation matrix (Keyframe 2)
            - 't': 3x1 translation vector (Keyframe 2)
            - 'points3d': (3 x N) 3D points before BA
            - 'pts_ref': (N x 2) 2D keypoints in Keyframe 1
            - 'pts_cur': (N x 2) 2D keypoints in Keyframe 2

        Returns:
          Optimized keyframe poses and refined 3D points.
        """
        # ✅ Prevents deepcopy issues
        init_result = {k: (copy.deepcopy(v) if k != 'matches' else v) for k, v in init_result.items()}

        # Extract Data
        R = init_result['R']
        t = init_result['t']
        points3d = init_result['points3d']
        pts_ref = init_result['pts_ref']
        pts_cur = init_result['pts_cur']
        N = points3d.shape[1]

        # ✅ Ensure valid 3D points
        valid_mask = np.isfinite(points3d).all(axis=0)
        points3d = points3d[:, valid_mask]
        pts_ref = pts_ref[valid_mask, :]
        pts_cur = pts_cur[valid_mask, :]

        # --- Add Camera Parameters ---
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        cam_params = g2o.CameraParameters(fx, (cx, cy), 0)
        cam_params.set_id(0)
        super().add_parameter(cam_params)

        # --- Add Keyframe 1 (Fixed Identity Pose) ---
        self.add_pose(0, g2o.SE3Quat(np.eye(3), np.zeros(3)), cam_params, fixed=True)

        # --- Add Keyframe 2 ---
        self.add_pose(1, g2o.SE3Quat(R, t), cam_params, fixed=False)

        # --- Add 3D Points (Landmarks) ---
        for i in range(N):
            self.add_point(i, points3d[:, i], fixed=False)

        # --- Add Edges (Observations) ---
        for i in range(N):
            self.add_edge(i, 0, pts_ref[i])
            self.add_edge(i, 1, pts_cur[i])

        # --- Run Optimization ---
        super().initialize_optimization()
        print("\nPerforming full BA:")
        super().optimize(self.iterations)

        # --- Extract Results ---
        refined_pose0 = self.get_pose(0)
        refined_pose1 = self.get_pose(1)
        refined_points3d = np.array([self.get_point(i) for i in range(N)]).T

        return refined_pose0, refined_pose1, refined_points3d

    def add_pose(self, pose_id, pose, cam, fixed=False):
        """ Adds a pose (camera pose) to the optimizer. """
        sbacam = g2o.SBACam(pose.rotation(), pose.translation())
        sbacam.set_cam(self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2], 0)


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
        return self.vertex(pose_id * 2).estimate()

    def get_point(self, point_id):
        """ Retrieves the optimized 3D point. """
        return self.vertex(point_id * 2 + 1).estimate()
