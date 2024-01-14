import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import cv2
import torch
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)

import pickle
import matplotlib.pyplot as plt
from keras.models import load_model
from pygame import mixer
mixer.init()
class BaseEngine(object):
    def __init__(self, engine_path):
        self.mean = None
        self.std = None
        self.n_classes = 2
        self.class_names = [ 'car', 'person' ]

        logger = trt.Logger(trt.Logger.WARNING)
        logger.min_severity = trt.Logger.Severity.ERROR
        runtime = trt.Runtime(logger)
        trt.init_libnvinfer_plugins(logger,'') # initialize TensorRT plugins
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
        # print(serialized_engine)
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.imgsz = engine.get_binding_shape(0)[2:]  # get the read shape of model, in case user input it wrong
        # print(self.imgsz)
        self.context = engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})


    def infer(self, img):
        self.inputs[0]['host'] = np.ravel(img)
        # transfer data to the gpu
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        # run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        # fetch outputs from gpu
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        # synchronize stream
        self.stream.synchronize()

        data = [out['host'] for out in self.outputs]
        return data

    def detect_video(self, video_path, conf=0.5, end2end=False):
        cap = cv2.VideoCapture(video_path)
        fps = int(round(cap.get(cv2.CAP_PROP_FPS)))

        sound = mixer.Sound('alarm.wav')
        with open('./model.pkl', 'rb') as file:
            loaded_scaler = pickle.load(file)
        test_x = loaded_scaler['mms_x']
        test_y = loaded_scaler['mms_y']
        test_model = load_model('./model.h5')
        fps = 0
        import time
        t1 = time.time()
        names = ['car','person']
        speed_list = []
        prev_boxes = []
        active_side = ''
        cnt_red = 0
        cnt_green = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            key = cv2.waitKey(1)
            low_img = frame.shape[0]
            upper_img = int(frame.shape[0]*0.6)
            blob, ratio = preproc(frame, self.imgsz, self.mean, self.std)
            data = self.infer(blob)
            current_time = time.time()
            fps = (1 / (current_time - t1))
            t1 = current_time
            frame = cv2.putText(frame, "FPS:%d " %fps, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 2)
            if end2end:
                num, final_boxes, final_scores, final_cls_inds = data
                final_boxes = np.reshape(final_boxes/ratio, (-1, 4))
                dets = np.concatenate([final_boxes[:num[0]], np.array(final_scores)[:num[0]].reshape(-1, 1), np.array(final_cls_inds)[:num[0]].reshape(-1, 1)], axis=-1)
            else:
                predictions = np.reshape(data, (1, -1, int(5+self.n_classes)))[0]
                dets = self.postprocess(predictions,ratio)

            if dets is not None:
                final_boxes, final_scores, final_cls_inds = dets[:,
                                                                :4], dets[:, 4], dets[:, 5]

            if key == ord('q'):
                active_side  = 'left'
            elif key == ord('e'):
                active_side  = 'right'
            elif key == ord('s'):
                active_side  = ''
            
            
            r_det_all = torch.tensor(dets)
            # print(r_det_all)
            tensor = torch.tensor(final_boxes)
            if torch.cuda.is_available():
                tensor = tensor.to('cuda:0')
            else:
                print("GPU is not available. Using CPU instead.")
            tensor_rounded = tensor.round()
            if torch.cuda.is_available():
                tensor_rounded = tensor_rounded.to('cuda:0')
            else:
                print("GPU is not available. Using CPU instead.")
            # print(tensor_rounded)
            if len(r_det_all):
                # tensor[:, :4] = scale_coords(frame.shape[2:], tensor[:, :4], frame.shape).round()
            
                r_det = reversed(tensor_rounded)
                
                input_data = np.array(r_det.cpu()).astype(int)
                input_data = test_x.transform(input_data)
                y_pred = test_y.inverse_transform(test_model.predict(input_data, verbose = 0))
                sp_list = []

                if len(prev_boxes) == 0:
                    prev_boxes = [[a, b] for a, b in zip(r_det, [[] for _ in range(len(r_det))])]
                    speed_list = y_pred

                for idx_p, (prev_box, m_speed) in enumerate(prev_boxes):
                    for idx_c, (*current_box, conf, cls) in enumerate(r_det_all):
                        center = current_box[0]+(current_box[2] - current_box[0]), current_box[3]

                        # x1, y1 : 왼쪽 위 좌표
                        # x2, y2 : 오른쪽 아래 좌표
                        x1, y1, x2, y2 = current_box
                        
                        te = [int(current_box[0]), int(current_box[1]), int(current_box[2]), int(current_box[3])]
                        xA = max(prev_box[0], te[0])
                        yA = max(prev_box[1], te[1])
                        xB = min(prev_box[2], te[2])
                        yB = min(prev_box[3], te[3])
                        interArea = max(0, xB - xA) * max(0, yB - yA)
                        boxAArea = (prev_box[2] - prev_box[0]) * (prev_box[3] - prev_box[1])
                        boxBArea = (te[2] - te[0]) * (te[3] - te[1])
                        iou = interArea / float(boxAArea + boxBArea - interArea)
                        if iou >= 0.6:  # 임계값
                            # 바운딩박스의 중심좌표구하기
                            label = f'{names[int(cls)]} {conf:.2f}'
                            
                            before = round(speed_list[idx_p][0], 2)
                            after = round(y_pred[idx_c][0], 2)
                            cha_sp = round(before - after, 2)
                        
                            if len(m_speed) == 0:
                                sp_list.append([cha_sp, False])
                                break
                            if abs(cha_sp) > abs(m_speed[0]) + 0.05 or abs(cha_sp) < abs(m_speed[0]) - 0.05:
                                cha_sp = round((cha_sp + m_speed[0]) / 2, 2)
                            
                            
                            speed = round(((m_speed[0] + cha_sp)) * 10, 2)
                            # sound.stop()
                            
                            # 위험할 때
                            # print(speed)
                            if after <= 5. or (speed * 10 > after):
                                sp_list.append([cha_sp, True])
                                
                                if active_side  == 'left' and x2 < 320:
                                    cnt_red += 2
                                    cnt_green -= 1
                                elif active_side  == 'right' and x1 > 320:
                                    cnt_red += 2
                                    cnt_green -= 1

                            # 위험하지 않을때
                            else :
                                sound.stop()
                                cnt_green +=1
                                cnt_red -=1
                                sp_list.append([cha_sp, False])

                prev_boxes = [i for i in zip(r_det[:, :4], sp_list)]
                speed_list = y_pred 
            
            points = np.array([[[269, 319], [108, 478], [524, 476], [361, 323]]])

            # 마스크 생성
            mask = np.zeros((480, 640), dtype=np.uint8)
            cv2.fillPoly(mask, [points], (255,))

            converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # 흰색 색상 범위
            white_mask = cv2.inRange(converted, np.array([0, 0, 120]), np.array([255, 80, 255]))
            yellow_mask = cv2.inRange(converted, np.array([18, 80, 80]), np.array([30, 255, 255]))

            combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
            white_yellow = cv2.bitwise_and(frame, frame, mask=combined_mask)
            median_blurred = cv2.medianBlur(white_yellow, 1)
            gray = cv2.cvtColor(median_blurred, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            masked_edges = cv2.bitwise_and(edges, edges, mask=mask)

            lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=20, minLineLength=50, maxLineGap=500)
            if lines is None: # and previous_lines is not None:
                lines = previous_lines
            else:
                previous_lines = lines

            # 화면에 그릴 두 직선을 위한 초기 설정
            left_line_x = []
            left_line_y = []
            right_line_x = []
            right_line_y = []
            
            # 차선 길이가 충분한지 확인하고 그려주기
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1)  # 기울기 계산
                if abs(slope) < 0.6:  # 수평에 가까운 선분은 무시
                    continue
                if slope <= 0:  # 왼쪽 차선
                    left_line_x.extend([x1, x2])
                    left_line_y.extend([y1, y2])
                else:  # 오른쪽 차선
                    right_line_x.extend([x1, x2])
                    right_line_y.extend([y1, y2])


            # 차선 길이가 충분한지 확인하고 그려주기
            if left_line_x:
                # 왼쪽 차선 다항식 계산
                poly_left = np.poly1d(np.polyfit(left_line_y, left_line_x, deg=1))
                under_left = int(poly_left(low_img))
                upper_left = int(poly_left(upper_img))


            if right_line_x:
                # 오른쪽 차선 다항식 계산
                poly_right = np.poly1d(np.polyfit(right_line_y, right_line_x, deg=1))
                under_right = int(poly_right(low_img))
                upper_right = int(poly_right(upper_img))


            A1 = upper_img - low_img
            B1 = under_left - upper_left
            C1 = A1 * under_left + B1 * low_img

            A2 = upper_img - low_img
            B2 = under_right - upper_right
            C2 = A2 * under_right + B2 * low_img

            # 행렬식 계산
            det = A1 * B2 - A2 * B1

            if det == 0:
                vanishing_point = None  
            else:
                # 소실점의 좌표 계산
                x = (B2 * C1 - B1 * C2) / det
                y = (A1 * C2 - A2 * C1) / det
                vanishing_point = [x, y]

            lane_distance = int(abs(under_left - under_right))
            soo=int(vanishing_point[0])
            chan=int(vanishing_point[1])

            left_triangle_points = np.array([
                ((under_left) - (lane_distance), low_img),
                (under_right, low_img),
                (soo, chan)
                ], dtype=np.int32)

            middle_triangle_points = np.array([
            (under_right, low_img),  # 오른쪽 차선의 하단 좌표
            (under_left, low_img),  # 왼쪽 차선의 하단 좌표 (이전에 정의되어 있어야 합니다)
            (soo, chan)                      # 소실점 좌표
            ], dtype=np.int32)

            right_triangle_points = np.array([
            (under_left, low_img),  # 오른쪽 차선의 하단 좌표
            ((under_left)+2*(lane_distance), low_img),  # 왼쪽 차선의 하단 좌표 (이전에 정의되어 있어야 합니다)
            (soo, chan)                      # 소실점 좌표
            ], dtype=np.int32)
            alpha = 0.5
            # overlay = np.zeros_like(im0s, dtype=np.uint8)

            if center !=None:
                center_x = int(center[0])
                center_y = int(center[1])

                # 변환된 좌표를 사용
                center = (center_x, center_y)


                # center 좌표가 삼각형 영역 안에 있는지 확인
                left_dist = cv2.pointPolygonTest(left_triangle_points, center, False)
                middle_dist=cv2.pointPolygonTest(middle_triangle_points, center, False)
                right_dist=cv2.pointPolygonTest(right_triangle_points, center, False)

                # dist 값이 0보다 크면 점은 내부에 있음, 0이면 경계 위에 있음, -1이면 외부에 있음
                left_inside_triangle = left_dist >= 0
                middle_inside_triangle = middle_dist >= 0
                right_inside_triangle = right_dist >= 0
                
                if left_inside_triangle and active_side=='left':
                    if cnt_red >4:
                        sound.play()                
                elif middle_inside_triangle:
                    pass              
                elif right_inside_triangle and active_side=='right':
                    if cnt_red >4:
                        sound.play()
            else:
                    pass

            x, y, w, h = 0, 0, frame.shape[1], int(low_img/1.55)
            cv2.imshow('frame', frame)
            if key == 27:
                cv2.destroyAllWindows()
                break
        cap.release()
        cv2.destroyAllWindows()

    def inference(self, img_path, conf=0.7, end2end=False):
        origin_img = cv2.imread(img_path)
        img, ratio = preproc(origin_img, self.imgsz, self.mean, self.std)
        data = self.infer(img)
        if end2end:
            num, final_boxes, final_scores, final_cls_inds = data
            final_boxes = np.reshape(final_boxes/ratio, (-1, 4))
            dets = np.concatenate([final_boxes[:num[0]], np.array(final_scores)[:num[0]].reshape(-1, 1), np.array(final_cls_inds)[:num[0]].reshape(-1, 1)], axis=-1)
        else:
            predictions = np.reshape(data, (1, -1, int(5+self.n_classes)))[0]
            dets = self.postprocess(predictions,ratio)

        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:,
                                                             :4], dets[:, 4], dets[:, 5]
            # origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
            #                  conf=conf, class_names=self.class_names)
        return origin_img

    @staticmethod
    def postprocess(predictions, ratio):
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy /= ratio
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.6, score_thr=0.1)
        return dets

    def get_fps(self):
        import time
        img = np.ones((1,3,self.imgsz[0], self.imgsz[1]))
        img = np.ascontiguousarray(img, dtype=np.float32)
        for _ in range(5):  # warmup
            _ = self.infer(img)

        t0 = time.perf_counter()
        for _ in range(100):  # calculate average time
            _ = self.infer(img)
        print(100/(time.perf_counter() - t0), 'FPS')


def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy"""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)


def preproc(image, input_size, mean, std, swap=(2, 0, 1)):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


def rainbow_fill(size=50):  # simpler way to generate rainbow color
    cmap = plt.get_cmap('jet')
    color_list = []

    for n in range(size):
        color = cmap(n/size)
        color_list.append(color[:3])  # might need rounding? (round(x, 3) for x in color)[:3]

    return np.array(color_list)


_COLORS = rainbow_fill(80).astype(np.float32).reshape(-1, 3)
