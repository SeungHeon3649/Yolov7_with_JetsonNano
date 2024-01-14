# json 라벨 데이터를 txt 데이터로 변경

import json
import os

# 클래스 이름을 클래스 ID로 매핑
class_map = {'일반차량': '0', '목적차량': '0', '이륜차': '0', '보행자': '1'}

# 변환 함수를 정의합니다.
def convert_to_yolo_format(json_file, output_dir):
    with open(json_file, encoding = 'utf-8') as f:
        data = json.load(f)

    for item in data['row']:
        # YOLO 포맷으로 변환
        x_center = (int(item['points1'].split(',')[0]) + int(item['points2'].split(',')[0])) / 2
        y_center = (int(item['points1'].split(',')[1]) + int(item['points3'].split(',')[1])) / 2
        width = abs(int(item['points2'].split(',')[0]) - int(item['points1'].split(',')[0]))
        height = abs(int(item['points3'].split(',')[1]) - int(item['points1'].split(',')[1]))

        x_center /= int(item['width'])
        y_center /= int(item['height'])
        width /= int(item['width'])
        height /= int(item['height'])

        class_id = class_map[item['label']]

        # 변환된 데이터를 저장할 파일명을 설정
        base_filename = os.path.splitext(item['filename'])[0]
        txt_filename = f"{base_filename}.txt"

        # 결과를 저장합니다.
        with open(os.path.join(output_dir, txt_filename), 'a') as f_out:
            f_out.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

# 시작 디렉토리 설정
start_dir = 'C:/Users/user/shlee/final_project/origin_data'  # JSON 파일들이 있는 시작 디렉토리
output_dir = 'C:/Users/user/shlee/final_project/labels_data'  # 변환된 TXT 파일들을 저장할 디렉토리

# 시작 디렉토리부터 모든 하위 디렉토리를 순회
for dirpath, dirnames, files in os.walk(start_dir):
    for file_name in files:
        if file_name.endswith('.json'):
            # 각 JSON 파일에 대해 변환 함수를 호출
            try:
                convert_to_yolo_format(os.path.join(dirpath, file_name), output_dir)
                print(f"json to txt 완료: {file_name}")
            except Exception as e:  # 예외의 유형을 출력
                print(f"json to text 실패: {file_name}, Error: {e}")
