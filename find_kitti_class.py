# kitti의 txt파일을 이용해서 클래스의 갯수 추출

import os
import glob
from collections import defaultdict

# KITTI 라벨 파일 폴더 경로 설정
kitti_labels_path = 'origin_data\label_2'

# 모든 고유한 클래스를 저장할 딕셔너리 초기화
unique_classes = defaultdict(int)

# 라벨 파일 순회
for label_file in glob.glob(os.path.join(kitti_labels_path, '*.txt')):
    with open(label_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            klass = parts[0]  # 클래스 이름
            unique_classes[klass] += 1

# 고유한 클래스 출력
for i, (klass, count) in enumerate(unique_classes.items()):
    print(f"Class {i}: {klass}, Count: {count}")

# 고유한 클래스를 YOLO 클래스 인덱스로 매핑
class_mapping = {klass: i for i, klass in enumerate(unique_classes.keys())}
print(f"YOLO Class Mapping: {class_mapping}")
