# import os
# import shutil

# def copy_png_files(start_dir, target_dir):
#     if not os.path.exists(target_dir):
#         os.makedirs(target_dir)
    
#     for dirpath, dirnames, files in os.walk(start_dir):
#         for file_name in files:
#             if file_name.lower().endswith('.png'):
#                 shutil.copy(os.path.join(dirpath, file_name), target_dir)

# # 시작 디렉토리 설정
# start_dir = 'C:/Users/user/shlee/final_project/origin_data'

# # 대상 디렉토리 설정
# target_dir = 'C:/Users/user/shlee/final_project/images_data'

# # 함수 호출
# try:
#     copy_png_files(start_dir, target_dir)
#     print("복사 완료")
# except Exception as e:  # 예외 유형
#     print(f"복사 실패: {e}")

# 폴더 안 이미지 다른 폴더로 이동

import os
import shutil

def move_png_files(start_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    for dirpath, dirnames, files in os.walk(start_dir):
        for file_name in files:
            if file_name.lower().endswith('.png'):
                shutil.move(os.path.join(dirpath, file_name), target_dir)

# 시작 디렉토리 설정
start_dir = 'C:/Users/user/shlee/final_project/origin_data'

# 대상 디렉토리 설정
target_dir = 'C:/Users/user/shlee/final_project/images_data'

# 함수 호출
try:
    move_png_files(start_dir, target_dir)
    print("이동 완료")
except Exception as e:  # 예외의 유형을 출력
    print(f"이동 실패: {e}")
