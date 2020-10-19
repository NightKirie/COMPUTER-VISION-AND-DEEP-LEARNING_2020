import cv2
import numpy as np
import matplotlib.pyplot as plt

def createKeyPoint():
    img1 = cv2.imread(f'Q4_Image/Aerial1.jpg')
    img2 = cv2.imread(f'Q4_Image/Aerial2.jpg')
    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(gray_img1, None)
    kp2, des2 = sift.detectAndCompute(gray_img2, None)
        
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    good = []
    kp1_good = []
    kp2_good = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good.append([m])
            kp1_good.append(kp1[m.queryIdx])
            kp2_good.append(kp2[m.trainIdx])

    draw_params = dict(
        color = (0,255,0),
        flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


    img1 = cv2.drawKeypoints(gray_img1, kp1_good[:6], img1, **draw_params)
    img2 = cv2.drawKeypoints(gray_img2, kp2_good[:6], img2, **draw_params)

    cv2.imshow("img1", img1)
    cv2.imshow("img2", img2)

    cv2.imwrite('Q4_Image/FeatureAerial1.jpg', img1)
    cv2.imwrite('Q4_Image/FeatureAerial2.jpg', img2)
    
    print("createKeyPoint")

def matchedKeyPoint():
    img1 = cv2.imread(f'Q4_Image/Aerial1.jpg')
    img2 = cv2.imread(f'Q4_Image/Aerial2.jpg')
    
    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(gray_img1, None)
    kp2, des2 = sift.detectAndCompute(gray_img2, None)
        
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    kp1_good = []
    kp2_good = []
    for i, (m1, m2) in enumerate(matches):
        if m1.distance < 0.5 * m2.distance:
            good.append([m1])
            kp1_good.append(kp1[m1.queryIdx])
            kp2_good.append(kp2[m1.trainIdx])

    draw_params = dict(
        matchColor = (0,255,0),
        singlePointColor = (255,0,0),
        flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


    img3 = cv2.drawMatchesKnn(gray_img1, kp1, gray_img2, kp2, good[0:6], None, **draw_params)
    plt.imshow(img3)
    plt.show()
