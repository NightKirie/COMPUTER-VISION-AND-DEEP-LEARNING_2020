import cv2
import numpy as np

def Perspective_Transform():
    im_src = cv2.imread("./Q3_Image/rl.jpg")
    
    cap = cv2.VideoCapture("./Q3_Image/test4perspective.mp4")
    fps = cap.get(cv2.cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter('./Q3_Image/output.mp4', -1, fps, (1280, 720))
    refPt1 = []
    refPt2 = []
    refPt3 = []
    refPt4 = []
    while True:
        ret, frame = cap.read()
        frame_size = (frame.shape[1], frame.shape[0])
        if not ret:
            break

        dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        params = cv2.aruco.DetectorParameters_create()
        markerCorners, markerIDs, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=params)
        index1 = np.squeeze(np.where(markerIDs == 25))
        if index1.size != 0:
            refPt1 = np.squeeze(markerCorners[index1[0]])[1]
        index2 = np.squeeze(np.where(markerIDs == 33))
        if index2.size != 0:
            refPt2 = np.squeeze(markerCorners[index2[0]])[2]
        index3 = np.squeeze(np.where(markerIDs == 30))
        if index3.size != 0:
            refPt3 = np.squeeze(markerCorners[index3[0]])[0]
        index4 = np.squeeze(np.where(markerIDs == 23))
        if index4.size != 0:
            refPt4 = np.squeeze(markerCorners[index4[0]])[0]
        
        distance = np.linalg.norm(refPt1 - refPt2)

        scalingFac = 0.02
        pts_dst = [[refPt1[0] - round(scalingFac*distance), refPt1[1] - round(scalingFac*distance)]]
        pts_dst = pts_dst + [[refPt2[0] + round(scalingFac*distance), refPt2[1] - round(scalingFac*distance)]]
        pts_dst = pts_dst + [[refPt3[0] + round(scalingFac*distance), refPt3[1] + round(scalingFac*distance)]]
        pts_dst = pts_dst + [[refPt4[0] - round(scalingFac*distance), refPt4[1] + round(scalingFac*distance)]]
        pts_dst = np.float32(pts_dst).astype("int32")
        pts_src = np.float32([[0, 0], [im_src.shape[1], 0], [im_src.shape[1], im_src.shape[0]], [0, im_src.shape[0]]]).astype("int32")

        retval, mask = cv2.findHomography(srcPoints=pts_src, dstPoints=pts_dst)#, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        im_per = cv2.warpPerspective(src=im_src, M=retval, dsize=frame_size)
        frame_per = cv2.fillConvexPoly(frame, pts_dst, 0)
        out.write(frame_per+im_per)
        cv2.imshow("img", frame_per+im_per)
        if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
            break

    
    cap.release()    
    out.release()
    cv2.destroyAllWindows()
    print('Perspective_Transform')


if __name__ == "__main__":
    Perspective_Transform()