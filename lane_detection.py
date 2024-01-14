# import cv2
# import numpy as np
# import time
# import sys

# src = cv2.VideoCapture("rear_lane_test2.mp4")
# if src.isOpened() == False:
#     print("동영상 안불러와짐")
#     sys.exit()

# # src.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# # src.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
# prev_time = time.time()

# while True:
#     ret, frame = src.read()
#     if ret == False:
#         print("동영상 출력 불가")
#         break
    
#     dst = cv2.resize(frame, (1280, 720))
    
#     # 관심영역(ROI) 설정
#     # 위에 배경과 아래 범퍼 제거
#     ROI = dst[int(dst.shape[0] / 2) + 40:dst.shape[0] - 10, int(dst.shape[1] / 4) - 10:dst.shape[1] - int(dst.shape[1] / 4) + 10]
    
#     # BGR -> HSV로 색상 변환
#     HSV = cv2.cvtColor(ROI, cv2.COLOR_BGR2HSV)

#     # 하얀색 차선범위 설정
#     lower_white = np.array([0, 0, 150])
#     upper_white = np.array([180, 255, 255])
#     white_mask = cv2.inRange(HSV, lower_white, upper_white)

#     # 노란색 차선범위 설정
#     lower_yellow = np.array([20, 100, 100])
#     upper_yellow = np.array([30, 255, 255])
#     yellow_mask = cv2.inRange(HSV, lower_yellow, upper_yellow)

#     # 노란색 흰색 마스크 결합
#     combined_mask = cv2.bitwise_or(white_mask, yellow_mask)

#     result = cv2.bitwise_and(ROI, ROI, mask=combined_mask)
    
#     # 필터링 and 에지검출
#     #bilateral_filter = cv2.bilateralFilter(result, 5, 100, 100)
#     blur_conversion = cv2.GaussianBlur(result, (5,5), 0)
#     canny = cv2.Canny(bilateral_filter, 50, 100)

#     lineResult = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
#     lines = cv2.HoughLinesP(canny, 1, np.pi / 180, 30, minLineLength = 30, maxLineGap = 5)
#     for line in lines:
#         x1, y1, x2, y2 = line[0]
#         cv2.line(lineResult, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)
#         if x2 - x1 == 0 or abs(float(y2 - y1) / float(x2 - x1)) < 0.7:
#             continue

#         # ROI 내의 좌표를 전체 이미지(dst)의 좌표로 변환
#         start_point = (x1 + int(dst.shape[1] / 4) - 10, y1 + int(dst.shape[0] / 2) + 40)
#         end_point = (x2 + int(dst.shape[1] / 4) - 10, y2 + int(dst.shape[0] / 2) + 40)
#         cv2.line(dst, start_point, end_point, (0, 0, 255), 2, cv2.LINE_AA)

#     if cv2.waitKey(33) == 27:
#         break

#     current_time = time.time()  # 현재 시간을 가져옴
#     fps = 1 / (current_time - prev_time)  # FPS 계산
#     prev_time = current_time  # 현재 시간을 이전 시간으로 설정
#     cv2.putText(dst, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)  # 화면에 FPS 표시

#     # 원본
#     cv2.imshow("dst", dst)  

#     cv2.imshow("result", result)
#     cv2.imshow("bilateral_filter", bilateral_filter)
#     cv2.imshow("canny", canny)
#     cv2.imshow("line_result", lineResult)

# src.release()
# cv2.destroyAllWindows()

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
prev_time = time.time()

while True:
    ret, frame = src.read()
    if ret == False:
        print("동영상 출력 불가")
        break
    
    dst = cv2.resize(frame, (1280, 720))
    
    dst_height, dst_width = dst.shape[:2]
    # 사진에서 보여진 대로 관심 영역을 정의
    # 좌표는 이미지 크기와 관측된 위치에 따라 조정되어야 함
    polygons = np.array([
        [(int(0.05 * dst_width), dst_height),
         (int(0.05 * dst_width), int(0.4 * dst_height)),
         (int(0.95 * dst_width), int(0.4 * dst_height)),
         (int(0.95 * dst_width), dst_height)]
    ])
    mask = np.zeros_like(dst)
    cv2.fillPoly(mask, polygons, 255)
    ROI = cv2.bitwise_and(dst, mask)

    # BGR -> HSV로 색상 변환
    HSV = cv2.cvtColor(ROI, cv2.COLOR_BGR2HSV)

    # 하얀색 차선범위 설정
    lower_white = np.array([0, 0, 150])
    upper_white = np.array([180, 255, 255])
    white_mask = cv2.inRange(HSV, lower_white, upper_white)

    # 노란색 차선범위 설정
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    yellow_mask = cv2.inRange(HSV, lower_yellow, upper_yellow)

    # 노란색 흰색 마스크 결합
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)

    result = cv2.bitwise_and(ROI, ROI, mask=combined_mask)
    
    # 필터링 and 에지검출
    bilateral_filter = cv2.bilateralFilter(result, 5, 100, 100)
    canny = cv2.Canny(bilateral_filter, 50, 100)

    lineResult = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
    lines = cv2.HoughLinesP(canny, 1, np.pi / 180, 30, minLineLength = 30, maxLineGap = 5)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(lineResult, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)
        if x2 - x1 == 0 or abs(float(y2 - y1) / float(x2 - x1)) < 0.7:
            continue

        # ROI 내의 좌표를 전체 이미지(dst)의 좌표로 변환
        start_point = (x1 + int(dst.shape[1] / 4) - 10, y1 + int(dst.shape[0] / 2) + 40)
        end_point = (x2 + int(dst.shape[1] / 4) - 10, y2 + int(dst.shape[0] / 2) + 40)
        cv2.line(dst, start_point, end_point, (0, 0, 255), 2, cv2.LINE_AA)

    if cv2.waitKey(33) == 27:
        break

    current_time = time.time()  # 현재 시간을 가져옴
    fps = 1 / (current_time - prev_time)  # FPS 계산
    prev_time = current_time  # 현재 시간을 이전 시간으로 설정
    cv2.putText(dst, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)  # 화면에 FPS 표시

    # 원본
    cv2.imshow("dst", dst)  

    cv2.imshow("result", result)
    cv2.imshow("bilateral_filter", bilateral_filter)
    cv2.imshow("canny", canny)
    cv2.imshow("line_result", lineResult)

src.release()
cv2.destroyAllWindows()