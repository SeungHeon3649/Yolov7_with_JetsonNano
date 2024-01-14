import cv2
import numpy as np
from RoadLaneDetection import RoadLaneDetector  # 차선 감지 클래스를 불러옵니다.

# 객체 생성
detector = RoadLaneDetector()

# 비디오 캡처 객체 생성
video = cv2.VideoCapture('sc_test.mp4')  # 입력 비디오 파일을 열기
# 비디오 파일 열기 검사
if not video.isOpened():
    print("동영상 파일을 열 수 없습니다.")
    exit()  # 파일을 열 수 없으면 종료

# 비디오 프레임의 크기를 가져옵니다.
ret, frame = video.read()
if not ret:
    exit()  # 프레임을 읽을 수 없으면 종료

# 결과 파일을 위한 비디오 라이터 설정
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 비디오 코덱 설정
fps = 25.0  # 프레임 속도 설정
writer = cv2.VideoWriter('result.avi', fourcc, fps, (frame.shape[1], frame.shape[0]))  # 비디오 라이터 객체 생성

cnt = 0

# 비디오를 읽고 처리하는 메인 루프
while True:
    ret, frame = video.read()  # 비디오에서 프레임 읽기
    if not ret:
        print("영상 못받음")
        break  # 더 이상 읽을 프레임이 없으면 종료
    # 필터링 및 차선 감지 로직
    filtered_img = detector.filter_colors(frame)  # 색상 필터링
    gray_img = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2GRAY)  # 그레이스케일로 변환
    edges = cv2.Canny(gray_img, 50, 150)  # Canny 에지 감지
    mask = detector.limit_region(edges)  # 관심 영역 제한
    mask_l = detector.limit_region_left(edges)  # 관심 영역 제한
    mask_r = detector.limit_region_right(edges)  # 관심 영역 제한
    # cv2.imshow('mask_c', mask)
    # cv2.imshow('mask_l', mask_l)
    # cv2.imshow('mask_r', mask_r)
    
    lines = detector.houghLines(mask)  # Hough 변환을 이용한 선 감지
    lines_l = detector.houghLines(mask_l)  # Hough 변환을 이용한 선 감지
    lines_r = detector.houghLines(mask_r)  # Hough 변환을 이용한 선 감지
    print(lines)
    if lines is not None:
        separated_lines = detector.separateLine(mask, lines)  # 선 분리
        separated_lines_l = detector.separateLine(mask_l, lines_l)  # 선 분리
        separated_lines_r = detector.separateLine(mask_r, lines_r)  # 선 분리
        lane = detector.regression(separated_lines, frame)  # 선형 회귀로 차선 계산
        lane_l = detector.regression(separated_lines_l, frame)  # 선형 회귀로 차선 계산
        lane_r = detector.regression(separated_lines_r, frame)  # 선형 회귀로 차선 계산
        result_img = detector.drawLine(frame, lane)  # 결과 이미지에 차선 그리기
        result_img = detector.drawLine(frame, lane_l)  # 결과 이미지에 차선 그리기
        result_img = detector.drawLine(frame, lane_r)  # 결과 이미지에 차선 그리기

        # 결과 저장
        writer.write(result_img)  # 비디오 파일에 결과 프레임 저장
        if cnt == 15:
            cv2.imwrite("img_result.jpg", result_img)  # 결과 이미지 저장
        cnt += 1
        # 결과 출력
        #cv2.imshow('result', result_img)  # 결과 프레임을 화면에 표시

        # ESC 키로 종료
        if cv2.waitKey(10) == 27:
            break  # ESC 키가 눌리면 루프 종료

# 자원 해제
video.release()  # 비디오 캡처 객체 해제
writer.release()  # 비디오 라이터 객체 해제
cv2.destroyAllWindows()  # 생성된 모든 창 닫기
