import csv

import cv2
import keras
from keras.utils.np_utils import to_categorical
from sklearn.metrics import f1_score, recall_score, precision_score
from tensorflow.keras.layers import Conv2D, MaxPooling2D, DepthwiseConv2D, GlobalAveragePooling2D, Dense, PReLU, Input, \
    BatchNormalization, GlobalMaxPooling2D, SeparableConv2D, LeakyReLU, Concatenate,Lambda,Flatten,SeparableConvolution2D
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers, layers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.layers import Activation, Dropout, Flatten, AveragePooling2D, add
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import regularizers
import os
import numpy as np
import warnings
from keras import backend as K
import time

import tensorflow.keras.layers

import cbam

import uuid
# 用于避免卷积层同名报错
from train_auc_model_new_cam import make_gradcam_heatmap, save_and_display_gradcam

unique_random_number = uuid.uuid4()

class FerModel(object):

    def __init__(self):
        self.x_shape = (224, 224, 3)
        self.epoch = 100
        self.batchsize = 32
        self.weight_decay = 0.0005
        self.classes = 10
        self.model = self.build_model()
        self.call_backs = self.get_call_backs()
        start = time.time()
        self.history = self.train()
        end = time.time()

        print(f'训练共耗时{round(end - start, 2)}s')

        # self.show_history()

    @staticmethod
    def get_call_backs():
        call_backs = [
            # ModelCheckpoint('./logs/' + 'best.h5',
            #                 save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2),
            TensorBoard('./logs/balanced_data_model'),
            # EarlyStopping(monitor='val_loss', patience=40)
        ]

        print("调用回调函数!!!")
        return call_backs


    def build_model(self):
        # base_model = tensorflow.keras.applications.resnet50.ResNet50(include_top=True, classes=10, weights=None, input_shape=(224, 224, 3))
        base_model = tensorflow.keras.applications.NASNetMobile(include_top=True,  weights=None,classes=10,
                                                                input_shape=(224, 224, 3))

        x = base_model.output
        model = Model(inputs=base_model.input, outputs=x)
        model.summary()
        return model

    def train(self):
        X_train = np.load('./driver_feature_RGB_224/train/images.npy')
        X_train = X_train.astype('float32')
        X_train = X_train.reshape([-1, 224,224, 3])
        np.random.seed(2025)
        np.random.shuffle(X_train)

        y_train = np.load('./driver_feature_RGB_224/train/labels.npy')
        np.random.seed(2025)
        np.random.shuffle(y_train)

        X_valid = np.load('./driver_feature_RGB_224/test/images.npy')
        X_valid = X_valid.astype('float32')
        X_valid = X_valid.reshape([-1, 224,224, 3])
        np.random.seed(2025)
        np.random.shuffle(X_valid)

        y_valid = np.load('./driver_feature_RGB_224/test/labels.npy')
        np.random.seed(2025)
        np.random.shuffle(y_valid)

        print(X_train.shape)
        print(y_train.shape)
        print(X_valid.shape)
        print(y_valid.shape)

        # X_train = np.load('./driver_feature_small_Sobel_driver100_rgb/test/images.npy')
        # X_train = X_train.reshape([-1, 224, 224, 1])
        # X_train = X_train - np.mean(X_train, axis=0)
        #
        # np.random.seed(6666)
        # np.random.shuffle(X_train)
        #
        # y_train = np.load('./driver_feature_small_Sobel_driver100_rgb/test/labels.npy')
        # np.random.seed(6666)
        # np.random.shuffle(y_train)
        #
        # X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        #
        # print(X_train.shape)
        # print(y_train.shape)
        # print(X_valid.shape)
        # print(y_valid.shape)


        momentum_optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.95)

        self.model.compile(optimizer=momentum_optimizer,
                           loss='categorical_crossentropy',  # 损失函数
                           metrics=['accuracy'])  # 指标
        # training the model
        history = self.model.fit(X_train, y_train,
                                 batch_size=self.batchsize,
                                 epochs=self.epoch,
                                 verbose=1,
                                 # shuffle=True,
                                 validation_data=(X_valid, y_valid),
                                 callbacks=[self.call_backs]
                                 )

        yPred = []
        y = []

        conficent = []
        temp = []

        for i in range(len(X_valid)):
            image = np.expand_dims(X_valid[i], axis=0)
            predictions = self.model.predict(image)
            top_class_index = tf.argmax(predictions, axis=-1)
            top_class_probability = predictions[0][top_class_index]
            x_real = [k for k, v in enumerate(y_valid[i]) if v == 1]
            real_class_probability = predictions[0][x_real[0]]
            yPred.append([int(top_class_index.numpy())])
            temp.append(int(top_class_index.numpy()))
            temp.append(top_class_probability)
            temp.append(x_real[0])
            temp.append(real_class_probability)
            conficent.append(temp)
            temp = []
            y.append(x_real)

        with open("confidenceFile/NASNetMobile.csv", "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["top_class_index", "top_class_probability", "x_real", "real_class_probability"])
            for r in conficent:
                writer.writerow(r)
        draw_confu(y, yPred, name='test')
        precision = precision_score(y, yPred, average='micro')
        _val_f1 = f1_score(y, yPred, average='micro')
        _val_recall = recall_score(y, yPred, average='micro')

        print("draw compliment" + str(_val_f1) + "*---" + str(_val_recall)+"*---" + str(precision))


        data_path_abs = 'C:\\Users\\Administrator\\Desktop\\DriverDetection\\auc'
        img_list_all = os.listdir(data_path_abs)
        for key, v in enumerate(img_list_all):
            input_img1 = cv2.imread(data_path_abs + "/" + v)
            input_img1 = cv2.resize(input_img1, (224, 224))
            sobel_image = np.expand_dims(input_img1, axis=0)
            #     # 获取最后一层卷积层的输出
            last_conv_layer = self.model.get_layer('activation_187')
            last_conv_layer_output = last_conv_layer.output
            model_with_last_conv = Model(inputs=self.model.input, outputs=last_conv_layer_output)
            feature_map = model_with_last_conv.predict(sobel_image)
            aa = visualize_feature_map(feature_map)

            input_img1 = cv2.imread(data_path_abs + "/" + v)
            input_img1 = cv2.resize(input_img1, (224, 224))

            feature_map_sum1 = cv2.resize(aa, (224, 224))
            # 将热力图转换为RGB格式
            feature_map_sum = np.uint8(255 * feature_map_sum1)
            feature_map_sum[feature_map_sum < 80] = 0
            # 将热利用应用于原始图像
            feature_map_sum = cv2.applyColorMap(feature_map_sum, cv2.COLORMAP_JET)
            # feature_map_sum = cv2.cvtColor(feature_map_sum, cv2.COLOR_BGR2RGB)  # 转换颜色映射
            # 　这里的热力图因子是０.４
            superimposed_img = feature_map_sum * 0.4 + input_img1
            cv2.imwrite("MobileNet1_{}".format(v), superimposed_img)

            #     # 计算类别的梯度
            grad_model = Model([self.model.inputs], [last_conv_layer.output, self.model.output])
            with tf.GradientTape() as tape:
                conv_layer_output, preds = grad_model(sobel_image)
                class_channel = preds[0][np.argmax(preds[0])]
            print(preds[0], end="------------np.argmax(preds[0]):")
            print(np.argmax(preds[0]), end="------------class_channel:")
            # class_channel = preds[0][np.argmax(preds[0])]
            print(class_channel)
            print("==================================================")


        # 靠谱方式
        data_path_abs = 'C:\\Users\\Administrator\\Desktop\\DriverDetection\\auc'
        img_list_all = os.listdir(data_path_abs)
        for key, v in enumerate(img_list_all):
            input_img1 = cv2.imread(data_path_abs + "/" + v)
            input_img1 = cv2.resize(input_img1, (224, 224))
            sobel_image = np.expand_dims(input_img1, axis=0)
            predictions = self.model.predict(sobel_image)

            # cv2.imshow('Segmentation', sobel_image)
            # cv2.waitKey(1500)
            # cv2.destroyAllWindows()

            #     # 获取最后一层卷积层的输出
            last_conv_layer = self.model.get_layer('activation_187')
            grad_model = Model([self.model.inputs], [last_conv_layer.output, self.model.output])

            #     # 计算类别的梯度
            with tf.GradientTape() as tape:
                conv_layer_output, preds = grad_model(sobel_image)
                class_channel = preds[0][np.argmax(preds[0])]
            # 计算梯度
            grads = tape.gradient(class_channel, conv_layer_output)
            # 计算权重
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            # pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
            # heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_layer_output), axis=-1)
            heatmap = conv_layer_output @ pooled_grads[..., tf.newaxis]

            # feature_map_sum = sum(ele for ele in feature_map_combination)
            heatmap = tf.maximum(heatmap, 0)
            heatmap /= tf.reduce_max(heatmap)

            # 重塑热力图并将其缩放到与原始图像相同的大小
            heatmap = np.squeeze(heatmap)
            #     heatmap = cv2.resize(heatmap, (224, 224))
            #     heatmap = np.uint8(255 * heatmap)
            #     heatmap = np.clip(heatmap, 0, 1)
            # 颜色映射
            gbkInput = cv2.imread(data_path_abs + "/" + v)
            gbkInput = cv2.resize(gbkInput, (224, 224))

            feature_map_sum1 = cv2.resize(heatmap, (224, 224))
            # 将热力图转换为RGB格式
            feature_map_sum = np.uint8(255 * feature_map_sum1)
            feature_map_sum[feature_map_sum < 80] = 0
            # 将热利用应用于原始图像
            feature_map_sum = cv2.applyColorMap(feature_map_sum, cv2.COLORMAP_JET)
            # 　这里的热力图因子是０.４
            superimposed_img = feature_map_sum * 0.4 + gbkInput
            cv2.imwrite("MobileNet2_{}".format(v), superimposed_img)

def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col


def output_heatmap(model, last_conv_layer, img):
    """Get the heatmap for image.

    Args:
           model: keras model.
           last_conv_layer: name of last conv layer in the model.
           img: processed input image.

    Returns:
           heatmap: heatmap.
    """
    # predict the image class

    preds = model.predict(img)
    # find the class index
    index = np.argmax(preds[0])
    print('index: %s' % index)
    # This is the entry in the prediction vector
    target_output = model.output[:, index]

    # get the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer)

    # compute the gradient of the output feature map with this target class
    grads = K.gradients(target_output, last_conv_layer.output)[0]

    # mean the gradient over a specific feature map channel
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    # this function returns the output of last_conv_layer and grads
    # given the input picture
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([img])

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the target class

    for i in range(conv_layer_output_value.shape[-1]):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(conv_layer_output_value, axis=-1)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[2]), cv2.INTER_LINEAR)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    print(heatmap.shape)
    return heatmap, index


def visualize_feature_map(img_batch):
    feature_map = np.squeeze(img_batch, axis=0)
    # print(feature_map.shape)
    feature_map_combination = []
    plt.figure()

    num_pic = feature_map.shape[2]
    row, col = get_row_col(num_pic)

    for i in range(0, num_pic):
        feature_map_split = feature_map[:, :, i]
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
    feature_map_sum = np.maximum(feature_map_sum, 0)
    feature_map_sum /= np.max(feature_map_sum)

    # print(feature_map_sum)

    plt.imshow(feature_map_sum)
    # plt.savefig("{}".format("feature_map_sum"+str(uuid.uuid4())+".eps"))
    # plt.savefig("{}".format("feature_map_sum"+str(uuid.uuid4())+".jpg"))
    return feature_map_sum

    # cv2.imshow('Segmentation', feature_map_sum)
    # cv2.waitKey(4000)
    # cv2.destroyAllWindows()

from tensorflow.keras.models import Model

def grad_cam(model, img_array, layer_name, class_idx=None):
    # 创建梯度计算模型
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if class_idx is None:
            class_idx = np.argmax(predictions)
        loss = predictions[:, class_idx]

    # 计算梯度
    grads = tape.gradient(loss, conv_outputs)

    # 计算权重（全局平均池化梯度）
    weights = tf.reduce_mean(grads, axis=(0, 1))

    # 生成热力图
    cam = np.dot(conv_outputs, weights)
    cam = np.maximum(cam, 0)  # ReLU
    cam = cv2.resize(cam, (img_array.shape, img_array.shape))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # 归一化

    return cam

import sklearn
import seaborn as sns
def draw_confu(y, y_pred, name=''):
    sns.set(font_scale=3)
    confusion_matrix = sklearn.metrics.confusion_matrix(y, y_pred)
    plt.xticks(fontsize=10)  # 设置x轴刻度字体大小为12
    plt.yticks(fontsize=10)
    plt.figure(figsize=(16, 14))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", annot_kws={"size": 20});
    plt.title("Confusion matrix", fontsize=32)
    plt.ylabel('Actual Label', fontsize=28)
    plt.xlabel('Predicted Label', fontsize=28)
    plt.savefig('./result_%s.eps' % (name))
    plt.savefig('./result_%s.svg' % (name))
    plt.savefig('./result_%s.jpg' % (name))

if __name__ == '__main__':
    import tensorflow as tf
    # 启用兼容模式
    fer_model = FerModel()
    print(tf.test.is_gpu_available())

