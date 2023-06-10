import glob
import numpy as np
import cv2

from src.get_corners import getChessboardCorners, normalize_points
from src.calc_homogeneous import compute_view_based_homography, h_refined
from src.calc_calibration import get_intrinsic_parameters, estimateExtrinsicParams
from src.calc_distortion import estimateReprojectionErrorDistortion
from src.visualization import visualize_pts

DATA_DIR = '/Users/ppangppang/Documents/dev/cv/data/Calibration_Imgs/'


def main():
    # Chessboard의 image 좌표, world 좌표 load
    chessboard_correspondences = getChessboardCorners(images=None, visualize = True)
    print('Get corner info and saved images Done')
    
    # Normalizing the chessboard points for better results of homography
    chessboard_correspondences_normalized = normalize_points(chessboard_correspondences)
    print('Init matrix Done')
    
    # obtaining homography
    homography = []
    for correspondence in chessboard_correspondences_normalized:
        homography.append(compute_view_based_homography(correspondence))
    print('Init homogeneous matrix Done')
    
    # refining the obtsined homography
    homography_refined = []
    for i in range(len(homography)):
        h_opt = h_refined(homography[i], chessboard_correspondences_normalized[i])
        homography_refined.append(h_opt)
    print('Refine homogeneous matrix Done')
    
    # obtaining the calibration matrix
    K = get_intrinsic_parameters(homography_refined)
    print("camera calibration matrix: ", K)

    extrinsic_para = []
    for i in range(len(homography_refined)):
        extrinsic = estimateExtrinsicParams(K, homography_refined[i])
        extrinsic_para.append(extrinsic)
        print("extrinsic params: ", extrinsic)
        
    print('Calculate parameter Done')
    
    # Distortion 보정
    optpoints = []
    for i in range(len(chessboard_correspondences)):
        image_points, object_points = chessboard_correspondences[i]
        points = estimateReprojectionErrorDistortion(K, extrinsic_para[i], image_points, object_points)
        optpoints.append(points)
    optpoints = np.array(optpoints)
    print('Solve distortion error Done')
    
    # save outputs
    images = [each for each in glob.glob(DATA_DIR + "*.jpg")]
    images = sorted(images)
    for i in range(len(optpoints)):
        image_points, object_points = chessboard_correspondences[i]
        visualize_pts(image_points, optpoints, images[i], i)
    print('getConers with calculated parameters and save Done')
    
    
if __name__ == '__main__':
    main()
