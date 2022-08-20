import numpy as np
import cv2


class CameraCalibration:
    def __init__(self, images):
        self.images = images
        self.homographies = []
        self.intrinsic_matrix = None
        self.points_rectangles = np.array([[0, 0], [19.1, 0], [19.1, 14], [0, 14]])

    def calculateHomographies(self):
        points_destination = []
        for angle in self.images:
            image = self.images[angle]
            markers_horizontal = [image.marker_front, image.marker_back]
            markers_vertical = [image.marker_right, image.marker_left]

            for marker in markers_horizontal:
                if marker is not None:
                    points_destination.append(marker.getCornersAsArray())

            for marker in markers_vertical:
                if marker is not None:
                    corners = marker.getCornersAsArray()
                    corners = np.concatenate((corners[1:], [corners[0]]), axis=0)
                    points_destination.append(corners)

        for points in points_destination:
            h, status = cv2.findHomography(self.points_rectangles, points)
            self.homographies.append(h)

    def svd_solve(self, A):
        """Solve a homogeneous least squares problem with the SVD
           method.
        Args:
           A: Matrix of constraints.
        Returns:
           The solution to the system.
        """
        U, S, V_t = np.linalg.svd(A)
        idx = np.argmin(S)

        least_squares_solution = V_t[idx]

        return least_squares_solution

    def generate_v_ij(self, H_stack, i, j):
        """Generate intrinsic orthogonality constraints. See Zhang pg. 6 for
           details.
        """
        M = H_stack.shape[0]

        v_ij = np.zeros((M, 6))
        v_ij[:, 0] = H_stack[:, 0, i] * H_stack[:, 0, j]
        v_ij[:, 1] = H_stack[:, 0, i] * H_stack[:, 1, j] + H_stack[:, 1, i] * H_stack[:, 0, j]
        v_ij[:, 2] = H_stack[:, 1, i] * H_stack[:, 1, j]
        v_ij[:, 3] = H_stack[:, 2, i] * H_stack[:, 0, j] + H_stack[:, 0, i] * H_stack[:, 2, j]
        v_ij[:, 4] = H_stack[:, 2, i] * H_stack[:, 1, j] + H_stack[:, 1, i] * H_stack[:, 2, j]
        v_ij[:, 5] = H_stack[:, 2, i] * H_stack[:, 2, j]

        return v_ij

    def recover_intrinsics(self):
        """Use computed homographies to calculate intrinsic matrix.
           Requires >= 3 homographies for a full 5-parameter intrinsic matrix.
        """
        M = len(self.homographies)

        # Stack homographies
        H_stack = np.zeros((M, 3, 3))
        for h, H in enumerate(self.homographies):
            H_stack[h] = H

        # Generate constraints
        v_00 = self.generate_v_ij(H_stack, 0, 0)
        v_01 = self.generate_v_ij(H_stack, 0, 1)
        v_11 = self.generate_v_ij(H_stack, 1, 1)

        # Mount constraint matrix
        V = np.zeros((2 * M, 6))
        V[:M] = v_01
        V[M:] = v_00 - v_11

        # Use SVD to solve the homogeneous system Vb = 0
        b = self.svd_solve(V)

        B0, B1, B2, B3, B4, B5 = b

        # Form B = K_-T K_-1
        B = np.array([[B0, B1, B3],
                      [B1, B2, B4],
                      [B3, B4, B5]])

        # Form auxilliaries
        w = B0 * B2 * B5 - B1 ** 2 * B5 - B0 * B4 ** 2 + 2. * B1 * B3 * B4 - B2 * B3 ** 2
        d = B0 * B2 - B1 ** 2

        # Use Zhang's closed form solution for intrinsic parameters (Zhang, Appendix B, pg. 18)
        v0 = (B[0, 1] * B[0, 2] - B[0, 0] * B[1, 2]) / (B[0, 0] * B[1, 1] - B[0, 1] * B[0, 1])
        lambda_ = B[2, 2] - (B[0, 2] * B[0, 2] + v0 * (B[0, 1] * B[0, 2] - B[0, 0] * B[1, 2])) / B[0, 0]
        alpha = np.sqrt(lambda_ / B[0, 0])
        beta = np.sqrt(lambda_ * B[0, 0] / (B[0, 0] * B[1, 1] - B[0, 1] * B[0, 1]))
        gamma = -B[0, 1] * alpha * alpha * beta / lambda_
        u0 = gamma * v0 / beta - B[0, 2] * alpha * alpha / lambda_

        # Reconstitute intrinsic matrix
        K = np.array([[alpha, gamma, u0],
                      [0., beta, v0],
                      [0., 0., 1.]])

        K = np.array([[alpha, 0., u0],
                      [0., beta, v0],
                      [0., 0., 1.]])

        self.intrinsic_matrix = K

    def calibrateCamera(self):
        self.calculateHomographies()
        self.recover_intrinsics()
        return self.intrinsic_matrix
