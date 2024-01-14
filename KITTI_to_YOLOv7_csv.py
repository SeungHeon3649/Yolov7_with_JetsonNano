import pandas as pd
import os

# Define the path to the directory containing the label files
label_dir = 'origin_data/label_2'

# Define the classes of interest and their new labels
class_mapping = {
    'Truck': 0,
    'Car': 0,
    'Van': 0,
    'Cyclist': 0,
    'Pedestrian': 1,
    'Person_sitting': 1
}

# Placeholder list for all bounding box information
bounding_boxes = []

# Check if the path exists
if not os.path.exists(label_dir):
    raise Exception(f"The provided directory path does not exist: {label_dir}")

# Loop over each file in the directory
for filename in os.listdir(label_dir):
    if filename.endswith('.txt'):
        # Construct the full file path
        file_path = os.path.join(label_dir, filename)
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split(' ')
                if parts[0] in class_mapping:
                    class_id = class_mapping[parts[0]]  # 클래스 번호
                    # 바운딩 박스 중심점 (x, y) 및 폭(width)과 높이(height) 계산
                    bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = map(float, parts[4:8])
                    bbox_x = (bbox_xmin + bbox_xmax) / 2.0
                    bbox_y = (bbox_ymin + bbox_ymax) / 2.0
                    bbox_width = bbox_xmax - bbox_xmin
                    bbox_height = bbox_ymax - bbox_ymin
                    # 깊이 정보 (Z 축 위치)
                    distance = float(parts[13])
                    bounding_boxes.append([class_id, bbox_x, bbox_y, bbox_width, bbox_height, distance])

# Create a DataFrame with the collected data
df = pd.DataFrame(bounding_boxes, columns=['Class', 'X', 'Y', 'Width', 'Height', 'Distance'])

# Save the DataFrame to a CSV file
csv_file_path = 'yj_babo_bounding_boxes.csv'
df.to_csv(csv_file_path, index=False)

csv_file_path