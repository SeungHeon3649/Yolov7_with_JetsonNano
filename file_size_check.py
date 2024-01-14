# 폴더의 사이즈를 체크해줌(파일의 갯수)

import os

# 폴더 경로를 지정합니다.
folder_path = 'origin_data\label_2'

# 해당 폴더 안에 있는 파일들의 리스트를 가져옵니다.
files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# 파일의 개수를 출력ㄴ합니다.
print(f"There are {len(files)} files in the directory.")
