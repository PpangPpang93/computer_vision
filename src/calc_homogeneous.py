import numpy as np
from scipy import optimize as opt

def compute_view_based_homography(correspondence):
    """
    correspondence[0~7]
    - 0 image_points : 이미지 point
    - 1 object_points : object point
    - 2 normalized_imp : 정규화된 이미지 point
    - 3 normalized_objp : 정규화된 object point
    - 4 n_matrix_imp : 정규화된 이미지 point matrix
    - 5 n_matrix_obp : 정규화된 object point matrix
    - 6 n_matrix_imp_inv : 정규화된 이미지 point inverse matrix
    - 7 n_matrix_obp_inv : 정규화된 object point inverse matrix
    """
    img_pts = correspondence[0]
    obj_pts = correspondence[1]
    norm_img_pts = correspondence[2]
    norm_obj_pts = correspondence[3]
    N_x = correspondence[5]
    N_u_inv = correspondence[6]
    
    N = len(img_pts) # 54
    M = np.zeros((2 * N, 9), dtype=np.float64) #(108, 9)

    for i in range(len(img_pts)):
        # 정규화된 image point
        u, v = norm_img_pts[i]
        # 정규화된 object point
        x, y = norm_obj_pts[i]
        r1 = np.array([-x, -y, -1, 0, 0, 0, x * u, y * u, u])
        r2 = np.array([0, 0, 0, -x, -y, -1, x * v, y * v, v])
        M[2 * i] = r1
        M[(2 * i) + 1] = r2

    # M.h  = 0 . Solve the homogeneous system (e. g. by singular value decomposition):
    # svd분해 (108,108) (9,1), (9,9)로 분리
    u, s, v_h = np.linalg.svd(M)

    # obtaining the minimum eigen value
    # 분해된 값 중에 s가 최소인 v 행렬을 채택
    h_norm = v_h[np.argmin(s)]
    h_norm = h_norm.reshape(3, 3)

    # de normalize equation 68 of Burger – Zhang’s Camera Calibration Algorithm
    # (이미지 point 정규화값 matrix * h_norm) * object point 정규화값 matrix
    h = np.matmul(np.matmul(N_u_inv, h_norm), N_x)
    # 3행 3열 값으로 전체 matrix 정규화
    h = h[:, :] / h[2, 2]

    reprojection_error = 0
    for i in range(len(img_pts)):
        t1 = np.array([[obj_pts[i][0]], [obj_pts[i][1]], [1.0]])
        t = np.matmul(h, t1).reshape(1, 3)
        t = t / t[0][-1]
        ## (image 좌표와 object 좌표의 x,y 오차 연산)
        reprojection_error += np.sum(np.abs(img_pts[i] - t[0][:-1]))

    reprojection_error = np.sqrt(reprojection_error / N)
    print("Reprojection error : ", reprojection_error)

    return h


def func_jacobian(initial_guess, X, Y, h, N):
    x_j = X.reshape(N, 2)
    jacobian = np.zeros( (2*N, 9) , np.float64)
    for j in range(N):
        x, y = x_j[j]
        s_x = np.float64(h[0]*x + h[1]*y + h[2])
        s_y = np.float64(h[3]*x + h[4]*y + h[5])
        w = np.float64(h[6]*x + h[7]*y + h[8])
        jacobian[2*j] = np.array([x/w, y/w, 1/w, 0, 0, 0, -s_x*x/w**2, -s_x*y/w**2, -s_x/w**2])
        jacobian[2*j + 1] = np.array([0, 0, 0, x/w, y/w, 1/w, -s_y*x/w**2, -s_y*y/w**2, -s_y/w**2])

    return jacobian


def func_minimize(initial_guess, X, Y, h, N):
    """
    a minimizer function
    :param initial_guess:
    :param X: normalized object points flattened
    :param Y: normalized image points flattened
    :param h: homography flattened
    :param N: number of points
    :return: aboslute difference of estimated value and normalised image points
    """
    x_j = X.reshape(N, 2)

    estimated = []
    for i in range(2*N):
        i == 0
        estimated.append(i)

    for j in range(N):
        x, y = x_j[j]
        estimated[2*j + 1] = (h[3] * x + h[4] * y + h[5]) / h[6]*x + h[7]*y + h[8]
        estimated[2*j] = (h[0] * x + h[1] * y + h[2]) / h[6]*x + h[7]*y + h[8]

    # return estimated
    return (np.abs(estimated-Y))**2


def h_refined(H, correspondences):
    """
    a function to return the refined homography
    :param H: homography matrix
    :param correspondence[0~7]: 
    - 0 image_points : 이미지 point
    - 1 object_points : object point
    - 2 normalized_imp : 정규화된 이미지 point
    - 3 normalized_objp : 정규화된 object point
    - 4 n_matrix_imp : 정규화된 이미지 point matrix
    - 5 n_matrix_obp : 정규화된 object point matrix
    - 6 n_matrix_imp_inv : 정규화된 이미지 point inverse matrix
    - 7 n_matrix_obp_inv : 정규화된 object point inverse matrix
    :return: refined homography matrix
    """
    # flateening the h matrix
    h = H.flatten()
    # flatten image points
    img_points = correspondences[0]
    y = img_points.flatten()
    # flatten objects points
    obj_points = correspondences[1]
    X = obj_points.flatten()
    # normalised object points
    norm_obj_points = correspondences[3]
    n = norm_obj_points.shape[0]

    ## jachobian 방식, minimize로 object point, image point, homogeneous matrix, normalized object point 값 넣어서 prime값 optimize
    h_prime = opt.least_squares(fun=func_minimize, x0=h, jac=func_jacobian, method="lm", args=[X, y, h, n], verbose=0)

    if h_prime.success:
        H = h_prime.x.reshape(3, 3)
    H = H / H[2, 2]
    return H