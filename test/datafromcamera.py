import cv2

cap = cv2.VideoCapture(0)  # 计算机自带的摄像头为0，外部设备为1
i = 0
while (1):
    ret, frame = cap.read()  # ret:True/False,代表有没有读到图片  frame:当前截取一帧的图片
    cv2.imshow("capture", frame)

    if (cv2.waitKey(1) & 0xFF) == ord('s'):  # 不断刷新图像，这里是1ms 返回值为当前键盘按键值
        RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # RGB图像转为单通道的灰度图像
        RGB = cv2.resize(RGB, (640, 480))  # 图像大小为320*240
        cv2.imwrite('/camera/%d.jpg' % i, RGB)
        i += 1
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
