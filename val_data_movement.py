# 이미지데이터 train, val 데이터로 나누기

import os
import shutil
import random

# 파일 경로를 설정
images_dir = 'C:/Users/user/shlee/final_project/images_data'
labels_dir = 'C:/Users/user/shlee/final_project/labels_data'
images_val_dir = 'C:/Users/user/shlee/final_project/images_data/val'
labels_val_dir = 'C:/Users/user/shlee/final_project/labels_data/val'
images_train_dir = 'C:/Users/user/shlee/final_project/images_data/train'
labels_train_dir = 'C:/Users/user/shlee/final_project/labels_data/train'

# 필요한 디렉토리가 없으면 생성
os.makedirs(images_val_dir, exist_ok=True)
os.makedirs(labels_val_dir, exist_ok=True)
os.makedirs(images_train_dir, exist_ok=True)
os.makedirs(labels_train_dir, exist_ok=True)

# 이미지와 레이블 디렉토리에서 파일 목록을 가져옴
image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]

# 이미지와 레이블 파일의 기본 이름이 일치하는지 확인
# 확장자를 제외하고 파일 이름을 비교
matched_files = set([os.path.splitext(f)[0] for f in image_files]) & \
                set([os.path.splitext(f)[0] for f in label_files])


# 일치하는 파일들 중에서 랜덤하게 100개를 선택합니다.

# 이전에 쓰던 코드 3.9 이전 버전
#selected_files = random.sample(matched_files, 7442)

# 3.9 이상부터는 list 또는 tuple같이 순서가 있는 시퀀스를 쓰라고함
matched_files_list = list(matched_files)  # 집합을 리스트로 변환
selected_files = random.sample(matched_files_list, 1841)


# 선택된 파일들을 새로운 디렉토리로 이동합니다.
for base_name in matched_files:
    image_path = os.path.join(images_dir, f"{base_name}.png")
    label_path = os.path.join(labels_dir, f"{base_name}.txt")

    if base_name in selected_files:
        # 이미지 파일 이동
        shutil.move(image_path, os.path.join(images_val_dir, f"{base_name}.png"))
        # 레이블 파일 이동
        shutil.move(label_path, os.path.join(labels_val_dir, f"{base_name}.txt"))
    else:
        # 이동하지 않은 파일들을 트레인 디렉토리로 이동
        shutil.move(image_path, os.path.join(images_train_dir, f"{base_name}.png"))
        shutil.move(label_path, os.path.join(labels_train_dir, f"{base_name}.txt"))

print("Files have been moved to their respective directories.")
