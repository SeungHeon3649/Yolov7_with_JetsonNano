import torch
from models.experimental import attempt_load

# 학습된 가중치와 모델 구조를 포함하는 파일을 로드합니다.
model = attempt_load('best.pt', map_location='cpu')  # 'cpu' 또는 'cuda'를 사용하세요

# 모델을 평가 모드로 설정합니다.
model.eval()

# ONNX로 변환할 때 필요한 더미 입력 데이터를 생성합니다.
dummy_input = torch.randn(1, 3, 640, 640)  # 이 값은 모델이 훈련된 입력 크기와 일치해야 합니다.

# 모델을 ONNX 형식으로 내보냅니다.
torch.onnx.export(model, dummy_input, "best.onnx", verbose=False, opset_version=12, input_names=['images'], output_names=['output'])

print("Model has been converted to ONNX format.")