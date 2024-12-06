import time

import cv2
img_path = 'person_xiushi.png'
input_img1 = cv2.imread(img_path)
input_img = cv2.cvtColor(input_img1, cv2.COLOR_BGR2GRAY)
input_img = cv2.resize(input_img, (224, 224))

sobel_start = time.time()
input_img = cv2.GaussianBlur(input_img, (3, 3), 0)  # 高斯滤波处理原图像降噪
x = cv2.Sobel(input_img, cv2.CV_16S, 1, 0)  # Sobel函数求完导数后会有负值，还有会大于255的值
y = cv2.Sobel(input_img, cv2.CV_16S, 0, 1)  # 使用16位有符号的数据类型，即cv2.CV_16S
Scale_absX = cv2.convertScaleAbs(x)  # 转回uint8
Scale_absY = cv2.convertScaleAbs(y)
sobel_image = cv2.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)
print(time.time()-sobel_start)