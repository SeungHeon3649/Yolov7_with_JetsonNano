import cv2
import numpy as np

class RoadLaneDetector:
    def __init__(self):
        # 클래스 초기화 시 필요한 변수들을 설정합니다.
        self.img_size = self.img_center = None  # 이미지 크기와 중심점 초기화
        self.left_m = self.right_m = None  # 왼쪽, 오른쪽 차선의 기울기 초기화
        self.left_b = self.right_b = None  # 왼쪽, 오른쪽 차선의 y절편 초기화
        self.left_detect = self.right_detect = False  # 왼쪽, 오른쪽 차선 감지 여부 초기화
        self.poly_bottom_width = 0.85  # 다각형 하단 너비 비율 설정
        self.poly_top_width = 0.07  # 다각형 상단 너비 비율 설정
        self.poly_height = 0.4  # 다각형 높이 비율 설정

    def filter_colors(self, img_frame):
        # 이미지에서 특정 색상을 필터링하여 차선 후보를 추출합니다.
        output = img_frame.copy()  # 원본 이미지 복사
        white_mask = cv2.inRange(output, (200, 200, 200), (255, 255, 255))  # 흰색 마스크 생성
        white_image = cv2.bitwise_and(output, output, mask=white_mask)  # 흰색 영역 추출
        img_hsv = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)  # 이미지를 HSV 색공간으로 변환
        yellow_mask = cv2.inRange(img_hsv, (10, 100, 100), (40, 255, 255))  # 노란색 마스크 생성
        yellow_image = cv2.bitwise_and(output, output, mask=yellow_mask)  # 노란색 영역 추출
        output = cv2.addWeighted(white_image, 1.0, yellow_image, 1.0, 0.0)  # 흰색 및 노란색 영역 합성
        return output  # 처리된 이미지 반환

    def limit_region(self, img_edges):
        # 이미지의 관심 영역을 설정합니다.
        height, width = img_edges.shape[:2]  # 이미지의 높이와 너비를 추출
        mask = np.zeros((height, width), dtype=np.uint8)  # 마스크를 생성 (초기값은 모두 0)
        # 관심 영역을 정의하는 다각형의 점을 계산
        points = np.array([[
            ((width * (1 - self.poly_bottom_width)) // 2, height),
            ((width * (1 - self.poly_top_width)) // 2, height - int(height * self.poly_height) * 1.2),
            (width - (width * (1 - self.poly_top_width)) // 2, height - int(height * self.poly_height) * 1.2),
            (width - (width * (1 - self.poly_bottom_width)) // 2, height)
        ]], dtype=np.int32)
        cv2.fillConvexPoly(mask, points, 255)  # 다각형 영역을 255로 채움
        # cv2.imshow('mask_r',mask)
        output = cv2.bitwise_and(img_edges, img_edges, mask=mask)  # 마스크를 적용하여 관심 영역만 추출
        
        return output  # 처리된 이미지 반환

    def houghLines(self, img_mask):
        # Hough 변환을 사용하여 직선 성분을 추출합니다.
        lines = cv2.HoughLinesP(img_mask, 1, np.pi / 180, 20, minLineLength=10, maxLineGap=5)
        return lines  # 감지된 선들을 반환

    def separateLine(self, img_edges, lines):
        output = [[], []]  # 좌, 우 차선을 저장할 리스트 초기화
        slopes = []  # 기울기를 저장할 리스트
        final_lines = []  # 최종 선들을 저장할 리스트
        slope_thresh = 0.2  # 기울기 임곗값 설정
        self.img_center = img_edges.shape[1] / 2  # 이미지 중심점 계산

        if lines is not None:  # lines가 None이 아닐 때만 반복문 실행
            for line in lines:  # 각 선에 대해 반복
                for x1, y1, x2, y2 in line:
                    # 선의 기울기 계산 (분모가 0이면 큰 값을 할당)
                    slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 999.0  
                    if abs(slope) > slope_thresh:  # 기울기 임곗값을 넘는 경우에만 처리
                        slopes.append(slope)  # 기울기 저장
                        final_lines.append((x1, y1, x2, y2))  # 선 저장

        for i, line in enumerate(final_lines):  # 최종 선들에 대해 반복
            x1, y1, x2, y2 = line
            # 선이 오른쪽에 위치하는지, 왼쪽에 위치하는지 판단
            if slopes[i] > 0 and x1 > self.img_center and x2 > self.img_center:
                self.right_detect = True  # 오른쪽 차선 감지됨
                output[0].append(line)  # 오른쪽 차선 리스트에 추가
            elif slopes[i] < 0 and x1 < self.img_center and x2 < self.img_center:
                self.left_detect = True  # 왼쪽 차선 감지됨
                output[1].append(line)  # 왼쪽 차선 리스트에 추가

        return output  # 분리된 선들 반환


    def regression(self, separated_lines, img_input):
        # 선형 회귀를 사용하여 차선의 대표적인 선을 계산합니다.
        output = [None] * 4  # 최종 선 좌표를 저장할 리스트 초기화
        # 오른쪽 및 왼쪽 차선의 점들 추출
        right_points = [pt for line in separated_lines[0] for pt in [tuple(line[:2]), tuple(line[2:])]]
        left_points = [pt for line in separated_lines[1] for pt in [tuple(line[:2]), tuple(line[2:])]]

        if right_points:
            # 오른쪽 차선에 대한 선형 회귀 수행
            right_line = cv2.fitLine(np.array(right_points, dtype=np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
            self.right_m = right_line[1] / right_line[0]  # 기울기 계산
            self.right_b = (right_line[2], right_line[3])  # y절편 계산

        if left_points:
            # 왼쪽 차선에 대한 선형 회귀 수행
            left_line = cv2.fitLine(np.array(left_points, dtype=np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
            self.left_m = left_line[1] / left_line[0]  # 기울기 계산
            self.left_b = (left_line[2], left_line[3])  # y절편 계산

        y1 = img_input.shape[0]  # 이미지의 하단 y 좌표
        y2 = 470  # 차선을 그릴 상단 y 좌표

        # 오른쪽 및 왼쪽 차선의 시작점과 끝점 계산
        if self.right_detect:
            right_x1 = (y1 - self.right_b[1]) / self.right_m + self.right_b[0]
            right_x2 = (y2 - self.right_b[1]) / self.right_m + self.right_b[0]
            output[0] = (right_x1, y1)
            output[1] = (right_x2, y2)

        if self.left_detect:
            left_x1 = (y1 - self.left_b[1]) / self.left_m + self.left_b[0]
            left_x2 = (y2 - self.left_b[1]) / self.left_m + self.left_b[0]
            output[2] = (left_x1, y1)
            output[3] = (left_x2, y2)

        return output  # 계산된 차선의 좌표 반환

    def drawLine(self, img_input, lane):
        # 각 차선의 점이 유효한지 확인하고 이미지에 차선을 그립니다.
        lane = [(0, 0) if pt is None else pt for pt in lane]
        if lane[0] is not None and lane[1] is not None:
            try:
                # 정수형 좌표로 변환하여 선 그리기
                pt1 = (int(lane[0][0]), int(lane[0][1]))
                pt2 = (int(lane[1][0]), int(lane[1][1]))
                cv2.line(img_input, pt1, pt2, (0, 255, 255), 5, cv2.LINE_AA)
            except Exception as e:
                print(f"Error drawing line: {e}")  # 오류 발생 시 출력

        if lane[2] is not None and lane[3] is not None:
            try:
                pt3 = (int(lane[2][0]), int(lane[2][1]))
                pt4 = (int(lane[3][0]), int(lane[3][1]))
                cv2.line(img_input, pt3, pt4, (0, 255, 255), 5, cv2.LINE_AA)
            except Exception as e:
                print(f"Error drawing line: {e}")  # 오류 발생 시 출력
        poly_points = np.array([lane[2], lane[0], lane[1], lane[3]], dtype=np.int32)
        overlay = img_input.copy()  # 오버레이 이미지 복사
        cv2.fillConvexPoly(overlay, poly_points, (0, 230, 30))  # 다각형 영역 채우기
        cv2.addWeighted(overlay, 0.3, img_input, 0.7, 0, img_input)  # 오버레이 적용
        # 이미지에 차선 그리기
        if lane[0] and lane[1]:
            cv2.line(img_input, pt1, pt2, (0, 255, 255), 5, cv2.LINE_AA)
        if lane[2] and lane[3]:
            cv2.line(img_input, pt3, pt4, (0, 255, 255), 5, cv2.LINE_AA)
        return img_input  # 처리된 이미지 반환

#################################################################

    def limit_region_left(self, img_edges):
        # 이미지의 관심 영역을 설정합니다.
        height, width = img_edges.shape[:2]  # 이미지의 높이와 너비를 추출
        mask = np.zeros((height, width), dtype=np.uint8)  # 마스크를 생성 (초기값은 모두 0)
        # 관심 영역을 정의하는 다각형의 점을 계산
        points = np.array([[
            (0, height),
            (0, height - int(height * self.poly_height) * 0.7),
            ((width * (1 - self.poly_top_width)) // 2, height - int(height * self.poly_height)),
            ((width * (1 - self.poly_bottom_width)) // 2, height)
        ]], dtype=np.int32)
        cv2.fillConvexPoly(mask, points, 255)  # 다각형 영역을 255로 채움
        # cv2.imshow("mask_rl",mask)
        output = cv2.bitwise_and(img_edges, img_edges, mask=mask)  # 마스크를 적용하여 관심 영역만 추출
        
        return output  # 처리된 이미지 반환
        
    def limit_region_right(self, img_edges):
        # 이미지의 관심 영역을 설정합니다.
        height, width = img_edges.shape[:2]  # 이미지의 높이와 너비를 추출
        mask = np.zeros((height, width), dtype=np.uint8)  # 마스크를 생성 (초기값은 모두 0)
        # 관심 영역을 정의하는 다각형의 점을 계산
        points = np.array([[
            (width - (width * (1 - self.poly_bottom_width)) // 2, height),
            (width - (width * (1 - self.poly_top_width)) // 2, height - int(height * self.poly_height)),
            (width, height - int(height * self.poly_height) * 0.7),
            (width, height)
        ]], dtype=np.int32)
        cv2.fillConvexPoly(mask, points, 255)  # 다각형 영역을 255로 채움
        # cv2.imshow("mask_rr",mask)
        output = cv2.bitwise_and(img_edges, img_edges, mask=mask)  # 마스크를 적용하여 관심 영역만 추출
        
        return output  # 처리된 이미지 반환
