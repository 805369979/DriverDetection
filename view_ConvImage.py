# coding: utf-8

from keras.models import Model
import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D,MaxPooling2D,SeparableConv2D
from keras.layers import Activation
from pylab import *
import keras
from keras.preprocessing import image


def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row,col

def visualize_feature_map(img_batch):
    feature_map = np.squeeze(img_batch,axis=0)
    print(feature_map.shape)

    feature_map_combination=[]
    plt.figure()

    num_pic = feature_map.shape[2]
    row,col = get_row_col(num_pic)

    for i in range(0,num_pic):
        feature_map_split=feature_map[:,:,i]
        # cv2.imwrite("feature_map_split.jpg",feature_map_split)
        feature_map_combination.append(feature_map_split)
        # plt.subplot(row,col,i+1)
        # plt.imshow(feature_map_split)
        # axis('off')
        # title('feature_map_{}'.format(i))


    # plt.savefig('feature_map.jpg')
    # plt.show()

    # 各个特征图按1：1 叠加
    feature_map_sum = sum(ele for ele in feature_map_combination)


    plt.savefig("img.eps")
    plt.savefig("img.jpg")

def create_model(img):
    model = Sequential()

    # 第一层CNN
    # 第一个参数是卷积核的数量，第二三个参数是卷积核的大小
    model.add(Convolution2D(64, 3, strides=(1, 1),padding="same", input_shape=img.shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 第二层CNN
    # model.add(Convolution2D(128, 3, strides=(1, 1),padding="same"))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # # # # # # # # #
    # # # # # # # 第三层CNN
    # model.add(Convolution2D(256, 3, strides=(1, 1),padding="same"))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # # # # # # # # #
    # # # # # # # # # 第四层CNN
    # model.add(Convolution2D(512, 3, strides=(1,1),padding="same"))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # # model.add(Convolution2D(256, 3, 3, input_shape=img.shape))
    # # model.add(Activation('relu'))
    # # model.add(MaxPooling2D(pool_size=(2, 2)))

    return model


if __name__ == "__main__":

    img = cv2.imread('person_xiushi.png')
    # img = image.load_img('person_xiushi.png', target_size=(512, 512))

    model = create_model(img)
    img_batch = np.expand_dims(img, axis=0)
    conv_img = model.predict(img_batch)  # conv_img 卷积结果
    visualize_feature_map(conv_img)