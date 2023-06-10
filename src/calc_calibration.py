import numpy as np

## intrinsic
def vpq(p, q, homography):
    v = np.array([homography[0, p] * homography[0, q],
                  homography[0, p] * homography[1, q] + homography[1, p] * homography[0, q],
                  homography[1, p] * homography[1, q],
                  homography[2, p] * homography[0, q] + homography[0, p] * homography[2, q],
                  homography[2, p] * homography[1, q] + homography[1, p] * homography[2, q],
                  homography[2, p] * homography[2, q]])
    return v


def get_intrinsic_parameters(r_h):
    M = len(r_h)
    V = np.zeros((2*M, 6), np.float64)

    for i in range(M):
        H = r_h[i]
        V[2*i] = vpq(p=0, q=1, homography=H)
        V[2*i + 1] = np.subtract(vpq(p=0, q=0, homography=H), vpq(p=1, q=1, homography=H))

    u, s, v_h = np.linalg.svd(V)
    b = v_h[np.argmin(s)]

    # according to zhangs equations
    vc = (b[1]*b[3] - b[0]*b[4])/(b[0]*b[2] - b[1]**2)
    l = b[5] - (b[3]**2 + vc*(b[1]*b[2] - b[0]*b[4]))/b[0]
    alpha = np.sqrt((l/b[0]))
    beta = np.sqrt(((l*b[0])/(b[0]*b[2] - b[1]**2)))
    gamma = -1*((b[1])*(alpha**2) * (beta/l))
    uc = (gamma*vc/beta) - (b[3]*(alpha**2)/l)

    print([vc, l, alpha, beta, gamma, uc])

    A = np.array([[alpha, gamma, uc],
                  [0,     beta,  vc],
                  [0,      0,   1.0]])
    return A


## extrinsic
def estimateExtrinsicParams(K, H):
    """
    a function for estimating the camera extrinsic parameters
    :param K: calibration matrix K
    :param H: homogrpahy
    :return: extrinsic matrix containing r and t
    """
    inv_k = np.linalg.inv(K)
    rot_1 = np.dot(inv_k, H[:, 0])
    lamda = np.linalg.norm(rot_1, ord=2)

    rot_1 = rot_1 / lamda
    rot_2 = np.dot(inv_k, H[:, 1])
    rot_2 = rot_2 / lamda
    rot_3 = np.cross(rot_1, rot_2)
    t = np.dot(inv_k, H[:, 2]) / lamda
    R = np.asarray([rot_1, rot_2, rot_3])
    R = R.T
    extrinsic = np.zeros((3, 4))
    extrinsic[:, :-1] = R
    extrinsic[:, -1] = t

    return extrinsic