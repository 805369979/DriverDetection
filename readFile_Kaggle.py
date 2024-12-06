import pandas as pd
import cv2
import os
import numpy as np
from skimage.feature import hog

ONE_HOT_ENCODING = True

OUTPUT_FOLDER_NAME = "D:\\深度学习项目代码\\file\\code\\firstCode\\driver_feature_kaggle"
data_path = 'Train_data_list.csv'

data_path_abs = 'D:\\dataset\\state-farm-distracted-driver-detection\\imgs\\train'

SELECTED_LABELS = [i for i in range(0,10)]
# SELECTED_LABELS = [0,1,2,3,4,5,6,7,8,9]

new_labels= SELECTED_LABELS


def get_new_label(label, one_hot_encoding=False):
    if one_hot_encoding:
        new_label = new_labels.index(label)
        label = list(np.zeros(len(new_labels), 'uint8'))
        label[new_label] = 1
        return label
    else:
        return new_labels.index(label)


img_data_list_train=[]
img_data_list_test=[]

labels_list_train = []
labels_list_test = []


img_list_all = os.listdir(data_path_abs)
count=0
for key,v in enumerate(img_list_all):
    print(key)
    for key1,v1 in enumerate(os.listdir(data_path_abs+"/"+v)):
        print(data_path_abs+"/"+v+"/"+v1)
        input_img = cv2.imread(data_path_abs+"/"+v+"/"+v1)
    #         cv2.imshow('Segmentation', input_img)
    #         cv2.waitKey(1500)
    #         cv2.destroyAllWindows()
    #     input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(input_img, (224, 224))
        # img_data_list_test.append(image)
        # print(get_new_label(key, one_hot_encoding=ONE_HOT_ENCODING))

        # labels_list_test.append(get_new_label(key, one_hot_encoding=ONE_HOT_ENCODING))
        blur = cv2.GaussianBlur(image, (3, 3), 0)  # 高斯滤波处理原图像降噪
        #
        x = cv2.Sobel(blur, cv2.CV_16S, 1, 0,ksize=3)  # Sobel函数求完导数后会有负值，还有会大于255的值
        y = cv2.Sobel(blur, cv2.CV_16S, 0, 1,ksize=3)  # 使用16位有符号的数据类型，即cv2.CV_16S
        Scale_absX = cv2.convertScaleAbs(x)  # 转回uint8
        Scale_absY = cv2.convertScaleAbs(y)
        sobel_image = cv2.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)
        sobel_image = cv2.cvtColor(sobel_image, cv2.COLOR_BGR2GRAY)

        cv2.imshow('Segmentation', sobel_image)
        cv2.waitKey(1500)
        cv2.destroyAllWindows()
        # img_data_list_test.append(sobel_image)

        count+=1
    #         break
    print(count)



print(len(img_data_list_test))
print(len(labels_list_test))
save_n = 'test'
np.save(OUTPUT_FOLDER_NAME + '/' + save_n + '/images.npy', img_data_list_test)

if ONE_HOT_ENCODING:
    np.save(OUTPUT_FOLDER_NAME + '/' + save_n + '/labels.npy', labels_list_test)
else:
    np.save(OUTPUT_FOLDER_NAME + '/' + save_n + '/labels.npy', labels_list_test)


img_data_list_test = []
labels_list_test = []
hog_features_test = []

