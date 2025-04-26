import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os
import numpy as np
from skimage import io

from skimage.feature import hog

ONE_HOT_ENCODING = True
GET_HOG_WINDOWS_FEATURES = True
GET_HOG_FEATURES = True


OUTPUT_FOLDER_NAME = "C:\\Users\\Administrator\\Desktop\\DriverDetection\\driver_feature_small_Sobel"
data_path = 'D:\\dataset\\auc.distracted.driver.dataset_v2\\v1_cam1_no_split2\\'

data_path_abs = 'D:\\dataset\\auc.distracted.driver.dataset_v2\\v1_cam1_no_split2\\'


original_labels = [0,1,2,3,4,5,6,7,8,9]
SELECTED_LABELS = [0,1,2,3,4,5,6,7,8,9]

new_labels= SELECTED_LABELS
print(new_labels)


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

landmarks_train = []
landmarks_test = []

hog_features_train = []
hog_features_test = []

hog_images_train = []
hog_images_test = []

# for path in ['Test_data_list.csv','Train_data_list.csv']:
from sklearn.svm import SVC

for path in ['Test_data_list.csv']:
    path = data_path+path
    print(path)
    train = pd.read_csv(path)

    train_data = train['Image']
    train_label = train['Label']

    count = 0
    for k, i in enumerate(train_data):
        if k%100==0:
            print(k, i)

        name = i.split("/")[-1]
        class_data = i.split("/")[-2]
        # print(class_data, name, train_label[k])

        img_list_all = os.listdir(data_path_abs + '/' + class_data)
        # print(img_list_all)

        for d in img_list_all:
            if name == d:
                # print(data_path + class_data+"\\"+name)
                input_img1 = cv2.imread(data_path + class_data + "\\" + name)
                input_img = cv2.cvtColor(input_img1, cv2.COLOR_BGR2GRAY)

                input_img = cv2.resize(input_img, (224, 224))
                # input_img1 = cv2.resize(input_img1, (448, 448))
                # print(get_new_label(key, one_hot_encoding=ONE_HOT_ENCODING))
                labels_list_test.append(get_new_label(train_label[k], one_hot_encoding=ONE_HOT_ENCODING))
                #

                input_img = cv2.GaussianBlur(input_img, (5, 5), 0)  # 高斯滤波处理原图像降噪
                x = cv2.Sobel(input_img, cv2.CV_16S, 1, 0)  # Sobel函数求完导数后会有负值，还有会大于255的值
                y = cv2.Sobel(input_img, cv2.CV_16S, 0, 1)  # 使用16位有符号的数据类型，即cv2.CV_16S
                Scale_absX = cv2.convertScaleAbs(x)  # 转回uint8
                Scale_absY = cv2.convertScaleAbs(y)
                sobel_image = cv2.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)
                img_data_list_test.append(sobel_image)
                # cv2.imshow('jijia_xy_Scharr_full', input_img1)
                # cv2.imshow('jijia_xy_Scharr_full', sobel_image)
                # plt.imsave("demo1.jpg", input_img1)  # 将改变后的图像保存
                # plt.imsave("demo1.eps", input_img1)  # 将改变后的图像保存
                # plt.imsave("demo1.svg", input_img1)  # 将改变后的图像保存

                # plt.imsave("demo.png", sobel_image)
                # plt.imsave("demo.svg", sobel_image)
                # plt.imsave("demo.eps", sobel_image)
                # cv2.waitKey(3000)
                # cv2.destroyAllWindows()
                count+=1
                break
    print(count)
    # if path.split('\\')[-1] == 'Train_data_list.csv':
    #     save_n = 'train'
    # else:
    #     save_n = 'test'
    save_n = 'test'
    # np.random.seed(2024)
    # np.random.shuffle(img_data_list_test)
    # np.random.seed(2024)
    # np.random.shuffle(labels_list_test)
    # np.random.seed(2019)
    # np.random.shuffle(hog_features_test)

    print(len(img_data_list_test))
    print(len(labels_list_test))
    # print(len(hog_features_test))

    np.random.seed(2025)
    np.random.shuffle(img_data_list_test)
    np.random.seed(2025)
    np.random.shuffle(labels_list_test)


    np.save(OUTPUT_FOLDER_NAME + '/' + save_n + '/images.npy', img_data_list_test)

    if ONE_HOT_ENCODING:
        np.save(OUTPUT_FOLDER_NAME + '/' + save_n + '/labels.npy', labels_list_test)
    else:
        np.save(OUTPUT_FOLDER_NAME + '/' + save_n + '/labels.npy', labels_list_test)
    # np.save(OUTPUT_FOLDER_NAME + '/' + save_n + '/hog_features.npy', hog_features_test)


    img_data_list_test = []
    labels_list_test = []
    # hog_features_test = []

