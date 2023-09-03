from ultralytics import YOLO
import cv2
import cvzone
import math
import torch

# 查看版本
print(torch.__version__)
# 查看GPU是否可用
print(torch.cuda.is_available())
# 查看GPU型号
print(torch.cuda.get_device_name())

cap = cv2.VideoCapture("fire.mp4")
module = YOLO("fire_detector.pt")
classNames = ['fire', 'smoke']
myColor = (0, 255, 255)
confThreshold = 0.3
num = 0

while True:
    success, img = cap.read()
    results = module(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # 获取置信度
            conf = math.ceil(box.conf[0] * 100) / 100
            # print(conf)

            # 获取类别
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            # print(currentClass)
            print(f"{currentClass} {conf}")

            # 添加置信度判断，避免出现太多重复边框以及错误判断
            if conf >= confThreshold:
                # 在图像中显示
                # 火焰与烟雾显示效果不同
                if classNames[cls] == "fire":
                    myColor = (0, 0, 255)
                    cvzone.putTextRect(img, f"{currentClass} {conf}",
                                       (max(0, x1 + 10), max(35, y1 - 10)), 1, 1,
                                       (0, 255, 255), colorR=(0, 0, 0))
                    cvzone.cornerRect(img, (x1, y1, w, h), l=3, colorR=myColor, t=2, rt=2)
                else:
                    myColor = (0, 255, 255)
                    cvzone.putTextRect(img, f"{currentClass} {conf}",
                                       (max(0, x1 + 10), max(35, y1 - 10)), 1, 1,
                                       (0, 255, 255), colorR=(0, 0, 0))
                    cvzone.cornerRect(img, (x1, y1, w, h), l=30, colorR=myColor, t=5)

    cv2.imshow("Image", img)
    key = cv2.waitKey(0)
    if key == ord('a'):
        num += 1
        cv2.imwrite(f"fire_{num}.jpg", img)
