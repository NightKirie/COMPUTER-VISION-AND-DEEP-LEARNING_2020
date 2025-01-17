import cv2
import numpy as np

def Preprocessing():
    cap = cv2.VideoCapture('./Q2_Image/opticalFlow.mp4')
    fps = cap.get(cv2.cv2.CAP_PROP_FPS)
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.filterByCircularity = True
    params.filterByColor = True
    params.filterByConvexity = True
    params.filterByInertia = True
    params.minArea = 30.0
    params.maxArea = 100.0
    params.minCircularity = 0.83
    params.minThreshold = 10
    params.maxThreshold = 150
    params.minConvexity = 0.8
    params.minInertiaRatio = 0.4
    while True:
        ret, img = cap.read()
        if not ret:
            break
        detector = cv2.SimpleBlobDetector_create(params)
        # Detect blobs.
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints = detector.detect(gray_img)
        
        # Draw keypoints
        for kp in keypoints:
            x = np.int(kp.pt[0])
            y = np.int(kp.pt[1])
            size = np.int(kp.size)
            # Draw 4 small square to construct a shape of crosshair in square
            top_left_1 = (int(x-size/2)-1, int(y-size/2)-1)
            bottom_right_1 = (int(x-size/2)+4, int(y-size/2)+4)
            top_left_2 = (int(x-size/2)+5, int(y-size/2)-1)
            bottom_right_2 = (int(x-size/2)+10, int(y-size/2)+4)
            top_left_3 = (int(x-size/2)-1, int(y-size/2)+5)
            bottom_right_3 = (int(x-size/2)+4, int(y-size/2)+10)
            top_left_4 = (int(x-size/2)+5, int(y-size/2)+5)
            bottom_right_4 = (int(x-size/2)+10, int(y-size/2)+10)
            
            cv2.rectangle(img, top_left_1, bottom_right_1, (0, 0, 255), 1)
            cv2.rectangle(img, top_left_2, bottom_right_2, (0, 0, 255), 1)
            cv2.rectangle(img, top_left_3, bottom_right_3, (0, 0, 255), 1)
            cv2.rectangle(img, top_left_4, bottom_right_4, (0, 0, 255), 1)

        # Show keypoints
        cv2.imshow("Keypoints", img)
        if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    print('Preprocessing')

def Video_Tracking():
    cap = cv2.VideoCapture('./Q2_Image/opticalFlow.mp4')
    fps = cap.get(cv2.cv2.CAP_PROP_FPS)
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.filterByCircularity = True
    params.filterByColor = True
    params.filterByConvexity = True
    params.filterByInertia = True
    params.minArea = 30.0
    params.maxArea = 100.0
    params.minCircularity = 0.8
    params.minThreshold = 50
    params.maxThreshold = 255
    params.minConvexity = 0.7
    params.minInertiaRatio = 0.4
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    color = np.random.randint(0,255,(100,3))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(old_frame)
    p0 = []
    for kp in keypoints:
        p0.append([kp.pt[0], kp.pt[1]])
    p0 = np.array(p0, dtype=np.float32).reshape(-1,1,2)
    mask = np.zeros_like(old_frame)
    while(1):
        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]
        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new, good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b), (c,d), (0, 0, 255), 2)
            frame = cv2.circle(frame, (a,b), 5, (0, 0, 255), -1)
        img = cv2.add(frame, mask)
        cv2.imshow('frame', img)
        if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
            break
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    Preprocessing()