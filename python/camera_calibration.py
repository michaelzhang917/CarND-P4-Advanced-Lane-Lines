import glob
import pickle
import numpy as np
import cv2

def cameraCalibration():
    # number of inside points on chessboard
    nx = 9
    ny = 6

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('../camera_cal/calibration*.jpg')

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

        # If found, add object points, image points
        if ret == True:
            print(fname)
            objpoints.append(objp)
            imgpoints.append(corners)
            # Draw and save the corners
            cv2.drawChessboardCorners(img, (8,6), corners, ret)
            write_name = '../results/corners_found'+str(idx)+'.jpg'
            cv2.imwrite(write_name, img)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    dist_pickle = {'mtx': mtx, 'dist': dist}
    with open('../results/wide_dist_pickle.p', 'wb') as f:
        pickle.dump(dist_pickle, f)
    return mtx, dist

if __name__ == '__main__':
    cameraCalibration()