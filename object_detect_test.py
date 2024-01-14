# 객체 검출
# test_prev_time = time.time()
# test_current_time = time.time()
# print(test_current_time - test_prev_time)
# test_prev_time = test_current_time



# import argparse
# import time
# from pathlib import Path

# import cv2
# import torch
# import torch.backends.cudnn as cudnn
# from numpy import random
# import numpy as np
# import matplotlib.pyplot as plt

# from models.experimental import attempt_load
# from utils.datasets import LoadStreams, LoadImages
# from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
#     scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
# from utils.plots import plot_one_box
# from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


# def detect(save_img=False):
#     source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
#     save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
#     webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
#         ('rtsp://', 'rtmp://', 'http://', 'https://'))

#     # Directories
#     save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
#     (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

#     # Initialize
#     set_logging()
#     device = select_device(opt.device)
#     half = device.type != 'cpu'  # half precision only supported on CUDA

#     # Load model
#     # 모델을 불러옴
#     model = attempt_load(weights, map_location=device) # 가중치와 장치 설정을 함
#     stride = int(model.stride.max())  # 이미지를 처리하는 단위 크기
#     # print("stride", stride)
#     imgsz = check_img_size(imgsz, s=stride)  # check img_size # 모델이 반정밀도(FP16)로 연산을 수행할지 결정

#     if trace:
#         model = TracedModel(model, device, opt.img_size)

#     # print(half)
#     if half:
#         model.half()  # to FP16

#     # 입력데이터 로드
#     vid_path, vid_writer = None, None

#     if webcam:
#         view_img = check_imshow()
#         cudnn.benchmark = True  # set True to speed up constant image size inference
#         dataset = LoadStreams(source, img_size=imgsz, stride=stride)
#     else:
#         dataset = LoadImages(source, img_size=imgsz, stride=stride)

#     # 이름과 색상
#     names = model.module.names if hasattr(model, 'module') else model.names
#     colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
#     # colors = [(0, 0, 255) for _ in names]

#     # 추론실행
#     if device.type != 'cpu':
#         model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
#     old_img_w = old_img_h = imgsz
#     old_img_b = 1

#     t0 = time.time()

#     # fps를 계산하기 위한 이전 시간 초기화
#     prev_time = time.time()

#     for path, img, im0s, vid_cap in dataset:
#         # im0s = cv2.resize(im0s, (640, 480))
#         # img = cv2.resize(img, (640, 480))
#         img = torch.from_numpy(img).to(device)  # numpy 배열 형식으로 불러온 이미지를 pytorch의 tensor 객체로 변환, device는 GPU, CPU 실행 가능한 환경에 맞게 설정 후 텐서를 할당
#         img = img.half() if half else img.float()  # 텐서의 데이터 타입을 지원가능하면 fp16, 불가능하면 fp32로 함
#         img /= 255.0  # 픽셀값의 범위를 [0, 255] -> [0.0, 1.0]
#         # webcam 일 때, [1, 3, 480, 640]
#         # video 일 때, [3, 384, 640]
#         # print(img.shape)
#         if img.ndimension() == 3:
#             img = img.unsqueeze(0)
        
#         #im0s = cv2.resize(im0s, (860, 640))

#         # 추론
#         t1 = time_synchronized()
#         with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
#             pred = model(img, augment=opt.augment)[0]
#         t2 = time_synchronized()

#         # FPS 계산
#         current_time = time.time()  # 현재 시간을 가져옴
#         fps = 1 / (current_time - prev_time)  # FPS 계산
#         prev_time = current_time  # 현재 시간을 이전 시간으로 설정

#         # NMS(비최대억제) 적용
#         pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
#         t3 = time_synchronized()

#         # 프로세스 감지
#         for i, det in enumerate(pred):  # 이미지당 탐지
#             if webcam:  # batch_size >= 1
#                 p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
#             else:
#                 p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

#             p = Path(p)  # to Path

#             cv2.putText(im0, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)  # 화면에 FPS 표시
#             #print(f"FPS : {fps:.2f}")

#             save_path = str(save_dir / p.name)  # img.jpg
#             txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
#             gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
#             if len(det):
#                 # img_size에서 im0 크기로 바운딩 박스 크기 조정
#                 # print("1", det[:, :4])
#                 det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
#                 # print("2", det[:, :4])
#                 # det[:, :4] = scale_coords(img.shape[2:], det[:, :4], resized_im0s.shape).round()

#                 # Print results
#                 for c in det[:, -1].unique():
#                     n = (det[:, -1] == c).sum()  # detections per class
#                     s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
#                 # 결과 쓰기
#                 for *xyxy, conf, cls in reversed(det):
#                     if conf >= 0.5:
#                         label = f'{names[int(cls)]} {conf:.2f}'
#                         plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

#             # Print time (inference + NMS)
#             #print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

#             # Stream results
#             if view_img:
#                 cv2.imshow(str(p), im0)
#                 key = cv2.waitKey(33)  # 1 millisecond
#                 if key == 27:
#                     cv2.destroyAllWindows()
#                     return

#     # print(f'Done. ({time.time() - t0:.3f}s)')


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model.pt path(s)') # 가중치 파일의 경로
#     parser.add_argument('--source', type=str, default='result_640.mp4', help='source')  # file/folder, 0 for webcam # 검출할 입력데이터 경로
#     parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)') # 신경망에 입력되는 이미지의 크기
#     parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold') # 신뢰도 기준
#     parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS') # NMS 과정에 사용되는 임계값
#     parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu') # cpu, gpu 사용 지정
#     parser.add_argument('--view-img', action='store_true', help='display results') # 결과 이미지를 화면에 표시할지 여부
#     parser.add_argument('--save-txt', action='store_true', help='save results to *.txt') # 감지된 객체 정보 txt파일로 저장할지 여부(클래스, 바운딩 박스 좌표,s 신뢰도 점수)
#     parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels') # 신뢰도 점수 저장할지 여부
#     parser.add_argument('--nosave', action='store_true', help='do not save images/videos') # 감지된 이미지 또는 비디오를 저장할지 ㅂ여부
#     parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3') # 필터링할 클래스 ID 지정
#     parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS') #클래스에 구애받지 않는 NMS를 수행할 것인지 결정
#     parser.add_argument('--augment', action='store_true', help='augmented inference') # 추론 시 데이터 증강 사용할지 여부
#     parser.add_argument('--update', action='store_true', help='update all models') # 모든 모델을 최신 상태로 업데이트 할 것인지 여부
#     parser.add_argument('--project', default='runs/detect', help='save results to project/name') # 결과 파일을 저장할 폴더 경로
#     parser.add_argument('--name', default='exp', help='save results to project/name') # 결과 파일을 저장할 하위 디렉토리 이름
#     parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment') # 옵션이 활성화 되면 경로가 이미 존재해도 새로운 디렉토리를 생성하지 않고 기존 디렉토리 사용
#     parser.add_argument('--no-trace', action='store_true', help='don`t trace model') # 모델 추적을 사용할지 여부
#     opt = parser.parse_args()
#     #opt.update = True
#     opt.view_img = True
#     opt.nosave = True
#     print(opt)

#     with torch.no_grad():
#         if opt.update:  # update all models (to fix SourceChangeWarning)
#             for opt.weights in ['best.pt']:
#                 detect()
#                 strip_optimizer(opt.weights)
#         else:
#             detect()


## 객체검출 & 속도추정인데 깜빡이 추가
import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
import matplotlib.pyplot as plt

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import pickle
from keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# # 모델 로드
with open('./model.pkl', 'rb') as file:
    loaded_scaler = pickle.load(file)
test_x = loaded_scaler['mms_x']
test_y = loaded_scaler['mms_y']
test_model = load_model('./model.h5')


def get_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    prev_boxes = []
    speed_list = []
    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    # 모델을 불러옴
    model = attempt_load(weights, map_location=device) # 가중치와 장치 설정을 함
    stride = int(model.stride.max())  # 이미지를 처리하는 단위 크기
    # print("stride", stride)
    imgsz = check_img_size(imgsz, s=stride)  # check img_size # 모델이 반정밀도(FP16)로 연산을 수행할지 결정

    if trace:
        model = TracedModel(model, device, opt.img_size) 

    # print(half)
    if half:
        model.half()  # to FP16

    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # 이름과 색상
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    prev_time = time.time()

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)  # numpy 배열 형식으로 불러온 이미지를 pytorch의 tensor 객체로 변환, device는 GPU, CPU 실행 가능한 환경에 맞게 설정 후 텐서를 할당
        img = img.half() if half else img.float()  # 텐서의 데이터 타입을 지원가능하면 fp16, 불가능하면 fp32로 함
        img /= 255.0  # 픽셀값의 범위를 [0, 255] -> [0.0, 1.0]
        img = img.unsqueeze(0)

        # 추론
        # with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
        pred = model(img, augment=opt.augment)[0]

        # FPS 계산
        current_time = time.time()  # 현재 시간을 가져옴
        fps = 1 / (current_time - prev_time)  # FPS 계산
        prev_time = current_time  # 현재 시간을 이전 시간으로 설정

        # NMS(비최대억제) 적용
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        # 프로세스 감지
        for i, det in enumerate(pred):  # 이미지당 탐지
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                r_det = reversed(det)
                input_data = np.array(r_det[:, :4].cpu()).astype(int)
                input_data = test_x.transform(input_data)
                y_pred = test_y.inverse_transform(test_model.predict(input_data, verbose = 0))
                color = 30
                sp_list = []
                if len(prev_boxes) == 0:
                    prev_boxes = [[a, b] for a, b in zip(r_det[:, :4], [[] for _ in range(len(r_det[:, :4]))])]
                    speed_list = y_pred 

                for idx_p, (prev_box, m_speed) in enumerate(prev_boxes):
                    for idx_c, (*current_box, conf, cls) in enumerate(r_det):
                        te = [int(current_box[0]), int(current_box[1]), int(current_box[2]), int(current_box[3])]
                        iou = get_iou(prev_box, te)
                        if iou >= 0.6:  # 임계값
                            label = f'{names[int(cls)]} {conf:.2f}'
                            cv2.circle(im0, (int(te[0]), int(te[1])), 5, (0, color + (int(idx_c) * 100), 0), -1)
                            # s = round(abs(speed_list[idx_p][0] - y_pred[idx_c][0]) * 5, 2)
                            # if s > 20: 
                            #     s = speed_list[idx_p][0]
                            before = round(speed_list[idx_p][0], 2)
                            after = round(y_pred[idx_c][0], 2)
                            cha_sp = round(before - after, 2)
                        
                            if len(m_speed) == 0:
                                sp_list.append([cha_sp, False])
                                break
                            if abs(cha_sp) > abs(m_speed[0]) + 0.05 or abs(cha_sp) < abs(m_speed[0]) - 0.05:
                                cha_sp = round((cha_sp + m_speed[0]) / 2, 2)
                            
                            
                            speed = round(((m_speed[0] + cha_sp)) * 10, 2)
     
                            # speed = round((before - after) * 5, 2)
                            if after <= 5. or (speed * 10 > after):
                                plot_one_box(str(cha_sp) + 'm/s', current_box, im0, label=label, color=(0, 0, 255), line_thickness=1)
                                sp_list.append([cha_sp, True])
                                continue
                            # elif speed < 0:
                            #     plot_one_box(str(speed) + 'm/s', current_box, im0, label=label, color=(255, 0, 0), line_thickness=1)
                            #     sp_list.append([cha_sp, False])
                            #     continue
                            plot_one_box(str(cha_sp) + 'm/s', current_box, im0, label=label, color=(0, 255, 0), line_thickness=1)
                            sp_list.append([cha_sp, False])


                prev_boxes = [i for i in zip(r_det[:, :4], sp_list)]
                speed_list = y_pred            

            cv2.putText(im0, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)  # 화면에 FPS 표시
            # Stream results
            cv2.imshow(str(p), im0)
            key = cv2.waitKey(1)  # 1 millisecond
            if key == 27:
                cv2.destroyAllWindows()
                return




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7-tiny.pt', help='model.pt path(s)') # 가중치 파일의 경로
    parser.add_argument('--source', type=str, default='result_640.mp4', help='source')  # file/folder, 0 for webcam # 검출할 입력데이터 경로
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)') # 신경망에 입력되는 이미지의 크기
    parser.add_argument('--conf-thres', type=float, default=0.6, help='object confidence threshold') # 신뢰도 기준
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS') # NMS 과정에 사용되는 임계값
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu') # cpu, gpu 사용 지정
    parser.add_argument('--view-img', action='store_true', help='display results') # 결과 이미지를 화면에 표시할지 여부
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt') # 감지된 객체 정보 txt파일로 저장할지 여부(클래스, 바운딩 박스 좌표,s 신뢰도 점수)
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels') # 신뢰도 점수 저장할지 여부
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos') # 감지된 이미지 또는 비디오를 저장할지 ㅂ여부
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3') # 필터링할 클래스 ID 지정
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS') #클래스에 구애받지 않는 NMS를 수행할 것인지 결정
    parser.add_argument('--augment', action='store_true', help='augmented inference') # 추론 시 데이터 증강 사용할지 여부
    parser.add_argument('--update', action='store_true', help='update all models') # 모든 모델을 최신 상태로 업데이트 할 것인지 여부
    parser.add_argument('--project', default='runs/detect', help='save results to project/name') # 결과 파일을 저장할 폴더 경로
    parser.add_argument('--name', default='exp', help='save results to project/name') # 결과 파일을 저장할 하위 디렉토리 이름
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment') # 옵션이 활성화 되면 경로가 이미 존재해도 새로운 디렉토리를 생성하지 않고 기존 디렉토리 사용
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model') # 모델 추적을 사용할지 여부
    opt = parser.parse_args()
    opt.view_img = True
    opt.nosave = True
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['best.pt']:
                print(detect())
                strip_optimizer(opt.weights)
        else:
            detect()