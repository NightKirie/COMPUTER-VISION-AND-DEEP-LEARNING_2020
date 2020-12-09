import cv2
import numpy as np

def Background_Subtraction():
    cap = cv2.VideoCapture("./Q1_Image/bgSub.mp4")
    fps = cap.get(cv2.cv2.CAP_PROP_FPS)
    frame_num = cap.get(cv2.cv2.CAP_PROP_FRAME_COUNT)
    background_list = np.empty(0)
    
    for i in range(50):
        ret, frame = cap.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        
        if background_list.size == 0:
            background_list = np.array([gray_frame])
        else:
            background_list = np.vstack([background_list, [gray_frame]])

    background_mean = np.mean(background_list, axis=(0))
    background_std = background_list.std(axis=(0))
    
    # if std value is < 5.0, set to 5.0
    background_std[background_std < 5.0] = 5.0
    
    # go back to the start of the video
    cap.set(cv2.cv2.CAP_PROP_POS_FRAMES, 0)
    ori_list = np.empty(0)
    sub_list = np.empty(0)
    while(True):
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print(f"Working on frame {int(cap.get(cv2.cv2.CAP_PROP_POS_FRAMES)-1)}")
        gray_frame[abs(gray_frame - background_mean) >  5 * background_std] = 255
        gray_frame[abs(gray_frame - background_mean) <=  5 * background_std] = 0

        if ori_list.size == 0:
            ori_list = np.array([frame])
        else: 
            ori_list = np.vstack([ori_list, [frame]])
        if sub_list.size == 0:
            sub_list = np.array([gray_frame])
        else: 
            sub_list = np.vstack([sub_list, [gray_frame]])

    cap.release()
    i = 0
    while(True):
        cv2.imshow('ori', ori_list[i])
        cv2.imshow('sub', sub_list[i])
        i += 1
        if i == frame_num:
            i = 0
        if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
