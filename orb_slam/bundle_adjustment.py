import g2o
import numpy as np
import copy  # Add at the top


class BundleAdjuster:
    def __init__(self, K, max_iterations=30):
        """
        Bundle Adjustment using g2o for full BA optimization.

        Parameters:
          - K: Camera calibration matrix (3x3 numpy array)
          - max_iterations: Maximum number of iterations for the optimizer.
        """
        self.K = K
        self.max_iterations = max_iterations

    def compute_reprojection_error(self, pose, points3d, measurements):
        """
        Compute the mean reprojection error in pixels.

        Parameters:
          - pose: g2o.SE3Quat representing the camera pose.
          - points3d: (3 x N) numpy array.
          - measurements: (N x 2) numpy array.

        Returns:
          - Mean reprojection error.
        """
        R = pose.rotation().matrix()
        t = pose.translation().reshape(3, 1)
        num_points = points3d.shape[1]

        pts_hom = np.vstack((points3d, np.ones((1, num_points))))
        proj = self.K @ (R @ points3d + t)
        proj /= proj[2, :]  # Normalize
        proj = proj[:2, :].T  # Convert to (N, 2)

        errors = np.linalg.norm(proj - measurements, axis=1)  # Compute L2 error
        return np.mean(errors)

    def optimize(self, init_result):
        """
        Run full bundle adjustment on the initial two-keyframe reconstruction.

        Parameters:
          init_result: Dictionary containing:
             - 'R': 3x3 rotation for keyframe 2.
             - 't': 3-element translation for keyframe 2.
             - 'points3d': (3 x N) numpy array of triangulated 3D map points.
             - 'pts_ref': (N x 2) 2D keypoints in Keyframe 1.
             - 'pts_cur': (N x 2) 2D keypoints in Keyframe 2.

        Returns:
          A tuple (refined_pose0, refined_pose1, refined_points3d).
        """
        # --- Unpack Initialization Data ---
        R = init_result['R']
        t = init_result['t']
        points3d = init_result['points3d']
        pts_ref = init_result['pts_ref']
        pts_cur = init_result['pts_cur']
        N = points3d.shape[1]

        # --- Compute Pre-BA Reprojection Errors ---
        pose1_initial = g2o.SE3Quat(np.eye(3), np.zeros(3))  # Keyframe 1 (Fixed)
        pose2_initial = g2o.SE3Quat(R, t.reshape(3))

        error1_pre = self.compute_reprojection_error(pose1_initial, points3d, pts_ref)
        error2_pre = self.compute_reprojection_error(pose2_initial, points3d, pts_cur)
        print(f"🔍 Pre-BA Error: Keyframe1 = {error1_pre:.2f} px, Keyframe2 = {error2_pre:.2f} px")

        # --- Setup g2o Optimizer ---
        optimizer = g2o.SparseOptimizer()
        
        # Try using Cholesky Solver, fallback to Dense Solver if unavailable
        try:
            solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())  # Preferred (if available)
        except AttributeError:
            print("⚠️ g2o: LinearSolverCholmodSE3 not found, using LinearSolverDenseSE3 instead.")
            solver = g2o.BlockSolverSE3(g2o.LinearSolverDenseSE3())  # Fallback

        solver = g2o.OptimizationAlgorithmLevenberg(solver)  
        optimizer.set_algorithm(solver)

        # --- Add Camera Parameters ---
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        cam_params = g2o.CameraParameters(fx, (cx, cy), 0)
        cam_params.set_id(0)
        optimizer.add_parameter(cam_params)

        # --- Add Keyframe 1 (Fixed as Identity) ---
        v_se3_0 = g2o.VertexSE3Expmap()
        v_se3_0.set_id(0)
        v_se3_0.set_estimate(g2o.SE3Quat(np.eye(3), np.zeros(3)))  
        v_se3_0.set_fixed(True)  
        optimizer.add_vertex(v_se3_0)

        # --- Add Keyframe 2 ---
        v_se3_1 = g2o.VertexSE3Expmap()
        v_se3_1.set_id(1)
        v_se3_1.set_estimate(g2o.SE3Quat(R, t))
        optimizer.add_vertex(v_se3_1)

        # --- Add Map Points ---
        point_vertices = []
        for i in range(N):
            v_point = g2o.VertexPointXYZ()
            v_point.set_id(i + 2)
            v_point.set_estimate(points3d[:, i])
            v_point.set_marginalized(True)
            optimizer.add_vertex(v_point)
            point_vertices.append(v_point)

        # --- Add Observation Edges ---
        for i in range(N):
            for keyframe_id, measurements in [(0, pts_ref), (1, pts_cur)]:
                edge = g2o.EdgeProjectXYZ2UV()
                edge.set_vertex(0, point_vertices[i])
                edge.set_vertex(1, optimizer.vertex(keyframe_id))
                edge.set_measurement(measurements[i])
                edge.set_information(np.identity(2) * (1.0 / 1.0**2))
                edge.set_robust_kernel(g2o.RobustKernelHuber())
                edge.set_parameter_id(0, 0)
                optimizer.add_edge(edge)

        # --- Optimize ---
        optimizer.initialize_optimization()
        optimizer.optimize(self.max_iterations)

        # --- Extract Optimized Poses & Points ---
        refined_pose0 = optimizer.vertex(0).estimate()  # Should remain identity
        refined_pose1 = optimizer.vertex(1).estimate()
        refined_points3d = np.array([v.estimate() for v in point_vertices]).T  

        # --- Compute Post-BA Reprojection Errors ---
        error1_post = self.compute_reprojection_error(refined_pose0, refined_points3d, pts_ref)
        error2_post = self.compute_reprojection_error(refined_pose1, refined_points3d, pts_cur)
        print(f"✅ Post-BA Error: Keyframe1 = {error1_post:.2f} px, Keyframe2 = {error2_post:.2f} px")

        return refined_pose0, refined_pose1, refined_points3d
