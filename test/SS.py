import selectivesearch.selectivesearch as ss
import cv2

img = cv2.imread("./image/1.jpg")
img_lbl, regions = ss.selective_search(img, scale=1000, sigma=0.8, min_size=50)
for r in regions:
    x, y, h, w = r['rect']
    cv2.rectangle(img, (x, y), (x + h, y + w), (0, 255, 0), 2)
cv2.imshow("img", img)
print("OK")
cv2.waitKey(0)
