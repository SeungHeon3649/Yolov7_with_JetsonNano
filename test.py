import cv2
import sys

src = cv2.VideoCapture(0)
if src.isOpened() == False:
    print("카메라 안켜짐")
    sys.exit()

# src.set(cv2.CAP_PROP_FRAME_WIDTH, 460)
# src.set(cv2.CAP_PROP_FRAME_HEIGHT, 460)

src.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
src.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = src.read()
    frame = cv2.resize(frame, (640, 640))
    if ret == False:
        print("동영상 출력 완료")
        break
    cv2.imshow("origin", frame)
    if cv2.waitKey(33) == 27:
        break

src.release()
cv2.destroyAllWindows()