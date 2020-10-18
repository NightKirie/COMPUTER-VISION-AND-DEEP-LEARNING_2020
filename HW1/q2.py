import cv2
import numpy as np

WIDTH = 11
HEIGHT = 8 

def draw(img, corners, imgpts):
    img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), (0,0,255), 5)
    img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[2].ravel()), (0,0,255), 5)
    img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[3].ravel()), (0,0,255), 5)
    
    img = cv2.line(img, tuple(imgpts[1].ravel()), tuple(imgpts[2].ravel()), (0,0,255), 5)
    img = cv2.line(img, tuple(imgpts[1].ravel()), tuple(imgpts[3].ravel()), (0,0,255), 5)
    
    img = cv2.line(img, tuple(imgpts[2].ravel()), tuple(imgpts[3].ravel()), (0,0,255), 5)
    return img

def augmentedReality():
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    obj_point_list = [] 
    img_point_list = []
    img_list = []
    objp = np.zeros((WIDTH*HEIGHT,3), np.float32)
    objp[:,:2] = np.mgrid[0:WIDTH,0:HEIGHT].T.reshape(-1,2)
    axis = np.float32([[3, 3, -3], [1, 1, 0], [3, 5, 0], [5, 1, 0]]).reshape(-1,3)
    for i in range(1, 6):
        img = cv2.imread(f"Q1_image/{i}.bmp")
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray_img, (WIDTH, HEIGHT), None)
        if ret == True:
            obj_point_list.append(objp)
            img_point_list.append(corners)
            
    ret, mtx_p, dist, rvecs, tvecs = cv2.calibrateCamera(obj_point_list, img_point_list, gray_img.shape[::-1], None, None)  
     
    for i in range(1, 6):
        img = cv2.imread(f"Q2_image/{i}.bmp")
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray_img, (WIDTH, HEIGHT), None)
        if ret == True:
            corners2 = cv2.cornerSubPix(gray_img,corners,(11,11),(-1,-1),criteria)
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objp], [corners], gray_img.shape[::-1], None, None)
            
            # Find the rotation and translation vectors.
            ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx_p, dist)
            
            # project 3D points to image plane
            imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx_p, dist)
            img = draw(img,corners2,imgpts)
            img = cv2.resize(img, (720, 720))
            img_list.append(img)
    
    for i in range(0, 5):
        cv2.imshow('img',img_list[i])
        cv2.waitKey(500)
        if i == 4:
            cv2.destroyAllWindows()