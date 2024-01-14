# KITTI의 라벨정보를 YOLOv7 포맷의 라벨 정보로 변경

# import os
# import glob
# from PIL import Image

# # 클래스 매핑
# class_mapping = {
#     'Truck': 0,
#     'Car': 0,
#     'Van': 0,
#     'Cyclist': 0,
#     'Pedestrian': 1,
#     'Person_sitting': 1
# }

# # 무시할 클래스
# ignore_classes = ['DontCare', 'Tram']

# # KITTI 라벨과 이미지 폴더 경로 설정 + 깊이 정보도 넣기
# kitti_labels_path = 'origin_data/label_2'  # KITTI 라벨 파일이 있는 폴더
# yolo_labels_path = 'labels_data'  # YOLO 라벨을 저장할 폴더

# # YOLO 라벨 폴더가 없다면 생성
# if not os.path.exists(yolo_labels_path):
#     os.makedirs(yolo_labels_path)

# # KITTI 이미지 폴더 경로 설정
# kitti_images_path = 'origin_data/data_object_image_2/training/image_2'

# # 라벨 파일 순회 및 변환
# for label_file in glob.glob(os.path.join(kitti_labels_path, '*.txt')):
    
#     # 이미지 파일 크기를 가져옴
#     image_filename = os.path.basename(label_file).replace('.txt', '.png')
#     image_file = os.path.join(kitti_images_path, image_filename)
    
#     with Image.open(image_file) as img:
#         image_width, image_height = img.size

#     with open(label_file, 'r') as file:
#         lines = file.readlines()

#     yolo_label_data = []
#     for line in lines:
#         parts = line.strip().split()
#         klass = parts[0]

#         # 무시할 클래스인 경우 건너뜀
#         if klass in ignore_classes:
#             continue

#         # 매핑할 클래스인 경우 변환 수행
#         if klass in class_mapping:
#             xmin = float(parts[4])
#             ymin = float(parts[5])
#             xmax = float(parts[6])
#             ymax = float(parts[7])
#             z = float(parts[13])  # 깊이 정보 추출
            
#             # 중심 좌표 및 크기 계산
#             x_center = ((xmin + xmax) / 2) / image_width
#             y_center = ((ymin + ymax) / 2) / image_height
#             width = (xmax - xmin) / image_width
#             height = (ymax - ymin) / image_height
            
#             # YOLO 포맷으로 변환된 데이터
#             yolo_label_data.append(f"{class_mapping[klass]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {z:.2f}\n")

#     # 변환된 YOLO 라벨 파일 저장
#     yolo_label_filename = os.path.basename(label_file)
#     yolo_label_filepath = os.path.join(yolo_labels_path, yolo_label_filename)
#     with open(yolo_label_filepath, 'w') as yolo_file:
#         yolo_file.writelines(yolo_label_data)

import os
import glob

# 클래스 매핑
class_mapping = {
    'Truck': 0,
    'Car': 0,
    'Van': 0,
    'Cyclist': 0,
    'Pedestrian': 1,
    'Person_sitting': 1
}

# 무시할 클래스
ignore_classes = ['DontCare', 'Tram']

# KITTI 라벨 폴더 경로 설정
kitti_labels_path = 'origin_data/label_2'  # KITTI 라벨 파일이 있는 폴더
yolo_labels_path = 'labels_data'  # YOLO 라벨을 저장할 폴더

# YOLO 라벨 폴더가 없다면 생성
if not os.path.exists(yolo_labels_path):
    os.makedirs(yolo_labels_path)

# 라벨 파일 순회 및 변환
for label_file in glob.glob(os.path.join(kitti_labels_path, '*.txt')):
    with open(label_file, 'r') as file:
        lines = file.readlines()

    yolo_label_data = []
    for line in lines:
        parts = line.strip().split()
        klass = parts[0]

        # 무시할 클래스인 경우 건너뜀
        if klass in ignore_classes:
            continue

        # 매핑할 클래스인 경우 변환 수행
        if klass in class_mapping:
            xmin = float(parts[4])
            ymin = float(parts[5])
            xmax = float(parts[6])
            ymax = float(parts[7])
            z = float(parts[13])  # 깊이 정보 추출
            
            # YOLO 포맷으로 변환된 데이터
            yolo_label_data.append(f"{class_mapping[klass]} {xmin} {ymin} {xmax} {ymax} {z}\n")

    # 변환된 YOLO 라벨 파일 저장
    yolo_label_filename = os.path.basename(label_file)
    yolo_label_filepath = os.path.join(yolo_labels_path, yolo_label_filename)
    with open(yolo_label_filepath, 'w') as yolo_file:
        yolo_file.writelines(yolo_label_data)
