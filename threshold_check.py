import cv2
import numpy as np
import time
import sys

src = cv2.VideoCapture("rear_lane_test2.mp4")
if src.isOpened() == False:
    print("동영상 안불러와짐")
    sys.exit()

# src.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# src.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

def onChange(value):
    pass

cv2.namedWindow("white_mask")
cv2.namedWindow("yellow_mask")

# test 트랙바 생성
cv2.createTrackbar("LH_white", "white_mask", 0, 180, onChange)
cv2.createTrackbar("UH_white", "white_mask", 180, 180, onChange)
cv2.createTrackbar("LH_yellow", "yellow_mask", 20, 180, onChange)
cv2.createTrackbar("UH_yellow", "yellow_mask", 30, 180, onChange)

while True:
    ret, frame = src.read()
    if ret == False:
        print("동영상 출력 불가")
        break
    
    dst = cv2.resize(frame, (1280, 720))
    
    # 관심영역(ROI) 설정
    # 위에 배경과 아래 범퍼 제거
    ROI = dst[int(dst.shape[0] / 2) + 40:dst.shape[0] - 10, int(dst.shape[1] / 4) - 10:dst.shape[1] - int(dst.shape[1] / 4) + 10]
    
    # BGR -> HSV로 색상 변환
    HSV = cv2.cvtColor(ROI, cv2.COLOR_BGR2HSV)

    # 트랙바 값 받아오기
    lh_white = cv2.getTrackbarPos("LH_white", "white_mask")
    uh_white = cv2.getTrackbarPos("UH_white", "white_mask")
    lh_yellow = cv2.getTrackbarPos("LH_yellow", "yellow_mask")
    uh_yellow = cv2.getTrackbarPos("UH_yellow", "yellow_mask")

    # 하얀색 차선범위 설정
    lower_white = np.array([lh_white, 0, 195])  # 200
    upper_white = np.array([uh_white, 255, 255]) # 255
    white_mask = cv2.inRange(HSV, lower_white, upper_white)

    # 노란색 차선범위 설정
    lower_yellow = np.array([lh_yellow, 100, 100])
    upper_yellow = np.array([uh_yellow, 255, 255])
    yellow_mask = cv2.inRange(HSV, lower_yellow, upper_yellow)

    # 노란색 흰색 마스크 결합
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)

    result = cv2.bitwise_and(ROI, ROI, mask=combined_mask)

    # 원본
    cv2.imshow("dst", dst)
    
    cv2.imshow("white_mask", white_mask)
    cv2.imshow("yellow_mask", yellow_mask)
    cv2.imshow("result", result)
    if cv2.waitKey(33) == 27:
        break

src.release()
cv2.destroyAllWindows()
