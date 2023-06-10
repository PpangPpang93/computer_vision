import numpy as np
import cv2

def estimateReprojectionErrorDistortion(calib_matrix, extrinsic, imgpoints, objpoints):
    err = []
    reproject_points = []

    u0, v0 = calib_matrix[0, 2], calib_matrix[1, 2]

    for impt, objpt in zip(imgpoints, objpoints):
        model = np.array([[objpt[0]], [objpt[1]], [0], [1]])

        proj_point = np.dot(extrinsic, model)
        proj_point = proj_point / proj_point[2]
        x, y = proj_point[0], proj_point[1]

        U = np.dot(calib_matrix, proj_point)
        U = U / U[2]
        u, v = U[0], U[1]
        k1 = 0.04909
        k2 = -0.3353
        t = x ** 2 + y ** 2
        u_cap = u + (u - u0) * (k1 * t + k2 * (t ** 2))
        v_cap = v + (v - v0) * (k1 * t + k2 * (t ** 2))

        reproject_points.append([u_cap, v_cap])

        err.append(np.sqrt((impt[0] - u_cap) ** 2 + (impt[1] - v_cap) ** 2))

    return np.mean(err), reproject_points