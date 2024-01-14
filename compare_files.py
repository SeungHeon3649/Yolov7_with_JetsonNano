# txt 파일과 img 파일의 이름이 똑같지 않으면 제거

import os

# txt 파일이 있는 디렉토리
txt_dir = 'C:/Users/user/shlee/final_project/labels_data'
# png 파일이 있는 디렉토리
png_dir = 'C:/Users/user/shlee/final_project/images_data'

# txt 파일의 기본 이름을 모두 가져옴 (확장자 제외)
txt_files = {os.path.splitext(filename)[0] for filename in os.listdir(txt_dir) if filename.endswith('.txt')}

# png 파일의 기본 이름을 모두 가져옴 (확장자 제외)
png_files = {os.path.splitext(filename)[0] for filename in os.listdir(png_dir) if filename.endswith('.png')}

# txt 디렉토리에서 png 파일에 없는 항목을 삭제
for txt_file in txt_files - png_files:
    os.remove(os.path.join(txt_dir, txt_file + '.txt'))

# png 디렉토리에서 txt 파일에 없는 항목을 삭제
for png_file in png_files - txt_files:
    os.remove(os.path.join(png_dir, png_file + '.png'))

print("이름이 일치하지 않는 파일들을 삭제했습니다.")
