import cv2
import numpy as np

WIDTH = 11
HEIGHT = 8 

def findCorners():
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for i in range(1, 16):
        img = cv2.imread(f"Q1_image/{i}.bmp")
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray_img, (WIDTH, HEIGHT), None)
        if ret == True:
            corners2 = cv2.cornerSubPix(gray_img, corners, (11,11), (-1,-1), criteria)
            cv2.drawChessboardCorners(img, (11,8), corners2, ret)
            img = cv2.resize(img, (720, 720))
            cv2.imshow('12x9 chessborad', img) 
            cv2.waitKey(2000)
    cv2.destroyAllWindows()

def findInstrinsic():
    obj_point_list = [] 
    img_point_list = []
    objp = np.zeros((WIDTH*HEIGHT,3), np.float32)
    objp[:,:2] = np.mgrid[0:WIDTH,0:HEIGHT].T.reshape(-1,2)
    for i in range(1, 16):
        img = cv2.imread(f"Q1_image/{i}.bmp")
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray_img, (WIDTH, HEIGHT), None)
        if ret == True:
            obj_point_list.append(objp)
            img_point_list.append(corners)
            
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_point_list, img_point_list, gray_img.shape[::-1], None, None)   
    print(f"Camera's intrinsic matrix is:")
    np.set_printoptions(suppress=True, precision=6)
    for i in range(0, 3):
        for j in range(0, 3):
            print("{:.6f}".format(mtx[i, j]), end=" ")
        print()
    print()

def findExtrinsic(i):
    objp = np.zeros((WIDTH*HEIGHT,3), np.float32)
    objp[:,:2] = np.mgrid[0:WIDTH,0:HEIGHT].T.reshape(-1,2)
    if i != "":
        img = cv2.imread(f"Q1_image/{i}.bmp")
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray_img, (WIDTH, HEIGHT), None)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objp], [corners], gray_img.shape[::-1], None, None)   
        rvecs = cv2.Rodrigues(rvecs[0])
        print(f"Extrinsic matrix of image {i}:")
        for i in range(0, 3):
            print("{:.6f} {:.6f} {:.6f} {:.6f}".format(rvecs[0][i][0], rvecs[0][i][1], rvecs[0][i][2], tvecs[0][i][0]))
        print()

def findDistorsion():
    obj_point_list = [] 
    img_point_list = []
    objp = np.zeros((WIDTH*HEIGHT,3), np.float32)
    objp[:,:2] = np.mgrid[0:WIDTH,0:HEIGHT].T.reshape(-1,2)
    for i in range(1, 16):
        img = cv2.imread(f"Q1_image/{i}.bmp")
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray_img, (WIDTH, HEIGHT), None)
        if ret == True:
            obj_point_list.append(objp)
            img_point_list.append(corners)
            
    img = cv2.imread(f"Q1_image/{3}.bmp")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_point_list, img_point_list, gray_img.shape[::-1], None, None)   
    print(f"Camera's distorsion matrix is:")
    np.set_printoptions(suppress=True, precision=6)
    for i in range(0, 5):
        print("{:.6f}".format(dist[0][i]), end=" ")
    print("")