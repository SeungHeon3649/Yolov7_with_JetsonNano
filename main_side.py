#import 함수들
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import argparse
import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
# 전역 변수 선언
start_x, start_y, end_x, end_y = -1, -1, -1, -1
x_y = []  # 클릭한 점들을 저장할 리스트
count = 0  # 클릭된 점의 수
cnt=0
image=None
roi_selected = False  # ROI 선택 완료 상태
frame = None  # 현재 프레임
max_line_x = None  # 이전 프레임의 최대 차선 위치
min_line_x = None  # 이전 프레임의 최소 차선 위치
max_line_y = None  # 이전 프레임의 최대 차선 위치
min_line_y = None  # 이전 프레임의 최소 차선 위치
previous_lines = None
center=None #객체의 밑변의 중심점 좌표
extended_line=None


def draw_roi(event, x, y, flags, param): 
    global count, x_y, roi_selected, image

    # 왼쪽 마우스 버튼 뗄 때마다 점을 추가
    if event == cv2.EVENT_LBUTTONUP:
        if count < 4:
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # image에 점 그리기
            x_y.append((x, y))
            count += 1
            cv2.imshow('Your Video', image)  # image 업데이트

        # 4개의 점이 선택되면, ROI 선택 완료
        if count == 4:
            roi_selected = True
            # ROI 그리기
            if len(x_y) == 4:
                cv2.line(image, x_y[0], x_y[1], (255, 0, 0), 2)
                cv2.line(image, x_y[1], x_y[2], (255, 0, 0), 2)
                cv2.line(image, x_y[2], x_y[3], (255, 0, 0), 2)
                cv2.line(image, x_y[3], x_y[0], (255, 0, 0), 2)
                cv2.imshow('Your Video', image)  # ROI가 그려진 이미지 업데이트
# 흰색과 노란색 선택 함수
def select_white_yellow(image):
    # HSV 색 공간으로 변환
    converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 흰색 색상 범위
    lower_white = np.array([0, 0, 120])
    upper_white = np.array([255, 80, 255])
    white_mask = cv2.inRange(converted, lower_white, upper_white)

    # 노란색 색상 범위
    lower_yellow = np.array([18, 80, 80])
    upper_yellow = np.array([30, 255, 255])
    yellow_mask = cv2.inRange(converted, lower_yellow, upper_yellow)

    # 흰색과 노란색 마스크 결합
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)

    # 원본 이미지에 마스크 적용
    result = cv2.bitwise_and(image, image, mask=combined_mask)
    return result

def extend_line_to_edges(connected_line, frame_width, frame_height):
    x1, y1, x2, y2 = connected_line

    # 선분이 수직인 경우
    if x2 == x1:
        return x1, 0, x1, frame_height

    # 선분의 기울기와 y절편을 계산
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1

    # 프레임의 왼쪽 가장자리(x=0)에서 y 좌표 계산
    y_at_x0 = intercept

    # 프레임의 오른쪽 가장자리(x=frame_width)에서 y 좌표 계산
    y_at_xmax = slope * frame_width + intercept

    # 연장된 선분을 반환
    return 0, int(y_at_x0), frame_width, int(y_at_xmax)

def find_extreme_lines(lines):
    if lines is None or len(lines) == 0:
        return None, None

    # 왼쪽과 오른쪽 선분을 초기화합니다.
    left_line = right_line = lines[0][0]

    # 모든 선분을 순회하며 왼쪽과 오른쪽 선분을 찾습니다.
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 < left_line[0] or x2 < left_line[0]:
                left_line = (x1, y1, x2, y2)
            if x1 > right_line[2] or x2 > right_line[2]:
                right_line = (x1, y1, x2, y2)

    return left_line, right_line

def connect_lines(line1, line2):
    if line1 is None or line2 is None:
        return None

    # 두 선분을 연결하는 선분을 생성합니다.
    connected_line = (line1[0], line1[1], line2[2], line2[3])
    return connected_line

def detect_lanes(frame, mask):

    global previous_lines, center,extended_line
    # 노란색과 흰색만 추출
    white_yellow = select_white_yellow(frame)
    # 메디안 블러
    median_blurred = cv2.medianBlur(white_yellow, 1)
    # 그레이스케일로 변환
    gray = cv2.cvtColor(median_blurred, cv2.COLOR_BGR2GRAY)
    # Canny 엣지 검출기 사용
    edges = cv2.Canny(gray, 100, 200)
   
    # 마스크 적용
    masked_edges = cv2.bitwise_and(edges, edges, mask=mask)

    # Hough 변환을 사용하여 선분 검출
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=20, minLineLength=60, maxLineGap=500)

    # 연결된 선분을 계산
    left_line, right_line = find_extreme_lines(lines)
    connected_line = connect_lines(left_line, right_line)

    if connected_line is not None:
        # 연장된 선분을 계산
        extended_line = extend_line_to_edges(connected_line, frame.shape[1], frame.shape[0])

        # 연장된 선분을 이미지에 그립니다.
        cv2.line(frame, (extended_line[0], extended_line[1]), (extended_line[2], extended_line[3]), (0, 255, 0), 2)

 
    triangle_points = np.array([
        (0,480),
        (extended_line[0],extended_line[1]),
        (extended_line[2],extended_line[3]),
        (640,480),        
    ], dtype=np.int32)

    alpha = 0.5
    overlay = np.zeros_like(frame, dtype=np.uint8)
    print(center)
    if center !=None:
        center_x = int(center[0])
        center_y = int(center[1])

        # 변환된 좌표를 사용
        center = (center_x, center_y)

        print(center)
        # center 좌표가 삼각형 영역 안에 있는지 확인
        dist=cv2.pointPolygonTest(triangle_points, center, False)

        # dist 값이 0보다 크면 점은 내부에 있음, 0이면 경계 위에 있음, -1이면 외부에 있음
        inside_triangle = dist >= 0
        print(center)
        
        if inside_triangle:
            cv2.fillConvexPoly(overlay, triangle_points, color=(100, 100, 255))
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            print("객체검출됨")

        else:
            cv2.fillConvexPoly(overlay, triangle_points, color=(100, 255, 100))

            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            print('객체검출없음')
    else:

            cv2.fillConvexPoly(overlay, triangle_points, color=(100, 255, 100))

            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.line(frame, (extended_line[0], extended_line[1]), (extended_line[2], extended_line[3]), (255, 255, 255),3) # 왼쪽 옆차선 검출





   
    return frame



SOME_THRESHOLD = 15000
def detect(save_img=False):
    global center
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
 
    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count #im0 -> 이미지
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            center=None
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if conf>=0.6:
                        x1,y1,x2,y2 = xyxy
                        x_center=xyxy[0]+(xyxy[2] - xyxy[0])
                        center=x_center, xyxy[3]
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or view_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'

                            x1, y1, x2, y2 = xyxy
                            box_area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]) 
        
                            # Change color based on the area of the bounding box
                            color = (0, 0, 255) if box_area >= SOME_THRESHOLD else colors[int(cls)]
                            plot_one_box(xyxy, im0, label=label, color=color, line_thickness=1)
                            # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)# 경계상자와 레이블을 그려줌

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                global cnt
                if cnt == 0:
                    cnt += 1

                    global image
                    image = im0  # 현재 프레임을 image로 설정
                    # 창을 미리 생성
                    cv2.namedWindow('Your Video')
                    cv2.setMouseCallback('Your Video', draw_roi)

                    while not roi_selected:  # 사용자가 ROI를 선택할 때까지 반복
                        cv2.imshow('Your Video', image)
                        cv2.waitKey(1)  # 이벤트 대기
                image=im0
                # 사용자가 4개의 점을 선택한 후 마스크 생성
                if roi_selected:
                    # points 배열을 np.int32 형식으로 변환
                    points = np.array([x_y], dtype=np.int32)

                    # 마스크 생성
                    mask = np.zeros((480, 640), dtype=np.uint8)
                    cv2.fillPoly(mask, [points], (255,))

                    
                    lanes_detected = detect_lanes(image, mask)  # image에 차선 그리기
                    image = lanes_detected  # 변경된 이미지를 image 변수에 할당

                cv2.imshow('Final Output', image)  # 최종 결과 출력
                cv2.waitKey(1)  # 이벤트 대기
            # Save results (image with detections)
            

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='best_aihub30000.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='side_car_car_640.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.6, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    opt.view_img = True
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7-tiny.pt']:
                detect()
                strip_optimizer(opt.weights)

        else:
            detect()