import numpy as np
import cv2
import glob

DATA_DIR = '/Users/ppangppang/Documents/dev/cv/data/Calibration_Imgs/'
DEBUG_DIR = '/Users/ppangppang/Documents/dev/cv/data/getConers/'
PATTERN_SIZE = (9, 6)
SQUARE_SIZE = 1.0


def getChessboardCorners(images=None, visualize=True):
    ## 실제 좌표계 object point init
    ## 실제 좌표계에서 (0,0), (0,1) ... (9,6)까지의 54개 좌표 위치를 물리적으로 지정
    objp = np.zeros((PATTERN_SIZE[1] * PATTERN_SIZE[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    image_points = []
    object_points = []
    correspondences = []
    ctr = 0
    images = [each for each in glob.glob(DATA_DIR + "*.jpg")]
    images = sorted(images)
    for path in images:  # images:
        img = cv2.imread(path, 0)
        ret, corners = cv2.findChessboardCorners(img, patternSize=PATTERN_SIZE)
        if ret:
            ## (54,2) 형태로 reshape
            corners = corners.reshape(-1, 2)
            if corners.shape[0] == objp.shape[0]:
                image_points.append(corners) 
                ## chess board의 world 좌표를 z=0을 가지도록 x,y만 사용(크기와 구조를 알고 있으므로)
                object_points.append(objp[:,:-1])
                correspondences.append([corners.astype(np.int), objp[:, :-1].astype(np.int)])
            ## cv2.findChessboardCorners로 찾은 corner 시각화
            if visualize:
                # Draw and display the corners
                ec = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                cv2.drawChessboardCorners(ec, PATTERN_SIZE, corners, ret)
                cv2.imwrite(DEBUG_DIR + 'getCorners ' + str(ctr) + ".png", ec)
        else:
            print("Error in detection points", ctr)

        ctr += 1

    return correspondences


def get_normalization_matrix(pts, name):
    """
    실질적으로 normalize 하는 함수(image, opt 구분해서 input)
    :param pts: point 값
    :param name: objects points or image points
    :return: normalisation and inverse normalisation matrix
    """
    # changing the type of points
    pts = pts.astype(np.float64)

    # 좌표의 mean, var, standard deviation 연산
    x_mean, y_mean = np.mean(pts, axis=0)
    x_var, y_var = np.var(pts, axis=0)
    s_x = np.sqrt(2 / x_var)
    s_y = np.sqrt(2 / y_var)

    # normalisation matrix
    n_matrix = np.array([[s_x,   0,  -s_x * x_mean],
                         [0,   s_y,  -s_y * y_mean],
                         [0,     0,              1]])

    # inverse of normalisation matrix
    n_matrix_inv = np.array([[1.0/s_x,     0,          x_mean],
                             [0,            1.0/s_y,   y_mean],
                             [0,            0,              1]])

    return n_matrix.astype(np.float64), n_matrix_inv.astype(np.float64)


def normalize_points(chessboard_correspondences):
    """
    image 좌표와 object 좌표 normalize
    :param chessboard_correspondences: corresponding points
    :return:
    """
    ret_correspondences = []

    for i in range(len(chessboard_correspondences)):

        image_points, object_points = chessboard_correspondences[i]

        # image points
        homogenous_image_pts = np.array([[[pt[0]], [pt[1]], [1.0]] for pt in image_points])
        normalized_hom_imp = homogenous_image_pts
        n_matrix_imp, n_matrix_imp_inv = get_normalization_matrix(image_points, "image points")

        # object points
        homogenous_object_pts = np.array([[[pt[0]], [pt[1]], [1.0]] for pt in object_points])
        normalized_hom_objp = homogenous_object_pts
        n_matrix_obp, n_matrix_obp_inv = get_normalization_matrix(object_points, "object points")

        for i in range(normalized_hom_objp.shape[0]):

            # object point 정규화
            n_o = np.matmul(n_matrix_obp, normalized_hom_objp[i])
            normalized_hom_objp[i] = n_o / n_o[-1]

            # image point 정규화
            n_u = np.matmul(n_matrix_imp, normalized_hom_imp[i])
            normalized_hom_imp[i] = n_u / n_u[-1]

        normalized_objp = normalized_hom_objp.reshape(normalized_hom_objp.shape[0], normalized_hom_objp.shape[1])
        normalized_imp = normalized_hom_imp.reshape(normalized_hom_imp.shape[0], normalized_hom_imp.shape[1])

        normalized_objp = normalized_objp[:, :-1]
        normalized_imp = normalized_imp[:, :-1]

        ret_correspondences.append((image_points, object_points, normalized_imp, normalized_objp, n_matrix_imp,
                                    n_matrix_obp, n_matrix_imp_inv, n_matrix_obp_inv))

    return ret_correspondences