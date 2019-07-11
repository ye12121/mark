import os
import cv2
from PIL import Image
import numpy as np

fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
size = (640, 360)
vw = cv2.VideoWriter('file.avi', fourcc=fourcc, fps=1.0, frameSize=size)

os.chdir('/home/ye/Downloads/object_detection/models/research/slim/slim-demo/test')
path = "/image"
path = os.path.join('/home/ye/Downloads/object_detection/models/research/slim/slim-demo/test', 'image')
filelist = os.listdir(path)
index = 0
for f in filelist:
    f_read = cv2.imread(os.path.join(path, f))
    f_img = Image.fromarray(f_read)
    f_rs = f_img.resize([640, 360], resample=Image.NONE)
    f_out = np.array(f_rs)
    # cv2.imwrite("file"+str(index)+".jpg",f_out)
    # index+=1
    vw.write(f_out)
vw.release()

cap = cv2.VideoCapture(os.path.join('/home/ye/Downloads/object_detection/models/research/slim/slim-demo/test', 'image', 'file.avi'))
while(1):
    # get a frame
    ret, frame = cap.read()
    # show a frame
    cv2.imshow("capture", frame)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()