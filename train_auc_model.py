import csv

import cv2
import keras
import sklearn
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
unique_random_number = uuid.uuid4()
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras

# Display
import matplotlib as mpl
import matplotlib.pyplot as plt

class FerModel(object):

    def __init__(self):
        self.x_shape = (224, 224, 1)
        self.epoch = 250
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

    def channel_shuffle(self, x, groups=1):
        height, width, in_channels = x.shape.as_list()[1:]
        channels_per_group = in_channels // groups

        x = K.reshape(x, [-1, height, width, groups, channels_per_group])
        x = K.permute_dimensions(x, (0, 1, 2, 4, 3))  # transpose
        x = K.reshape(x, [-1, height, width, in_channels])

        print(K.print_tensor(x, 'shuffled'))

        return x

    def gp_3x3(self, inputs, pre_layer):
        interval = int(pre_layer / 2)

        normalized1_1 = Lambda(lambda x: x[:, :, :, :interval],name=str(uuid.uuid4()))(inputs)
        normalized1_2 = Lambda(lambda x: x[:, :, :, interval:],name=str(uuid.uuid4()))(inputs)

        # 第一路
        tower_1 = DepthwiseConv2D((3, 3), padding='same',
                                  input_shape=self.x_shape, kernel_initializer='uniform',kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(normalized1_1)

        # 第二路
        tower_2 = DepthwiseConv2D((3, 3), padding='same',
                                  input_shape=self.x_shape,kernel_initializer='uniform', kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(normalized1_2)

        output = keras.layers.concatenate([tower_1, tower_2], axis=3,name=str(uuid.uuid4()))
        # output = self.channel_shuffle(output, 2)

        output = Conv2D(pre_layer // 4, 1, kernel_regularizer=regularizers.l2(self.weight_decay), activation='relu')(
            output)
        output = BatchNormalization()(output)

        return output

    def gp_5x5(self, inputs, pre_layer):
        interval = int(pre_layer / 4)

        normalized1_1 = Lambda(lambda x: x[:, :, :, :interval],name=str(uuid.uuid4()))(inputs)
        normalized1_2 = Lambda(lambda x: x[:, :, :, interval:2 * interval],name=str(uuid.uuid4()))(inputs)
        normalized1_3 = Lambda(lambda x: x[:, :, :, 2 * interval:3 * interval],name=str(uuid.uuid4()))(inputs)
        normalized1_4 = Lambda(lambda x: x[:, :, :, 3 * interval:],name=str(uuid.uuid4()))(inputs)

        # 第一路
        tower_1 = DepthwiseConv2D((5, 1), padding='same',
                                  input_shape=self.x_shape, kernel_initializer='uniform',
                                  kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(normalized1_1)
        tower_1 = DepthwiseConv2D((1, 5), padding='same',
                                  input_shape=self.x_shape,kernel_initializer='uniform', kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(tower_1)

        # 第二路
        tower_2 = DepthwiseConv2D((5, 1), padding='same',
                                  input_shape=self.x_shape, kernel_initializer='uniform',kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(normalized1_2)
        tower_2 = DepthwiseConv2D((1, 5), padding='same',
                                  input_shape=self.x_shape,kernel_initializer='uniform', kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(tower_2)

        # 第三路
        tower_3 = DepthwiseConv2D((5, 1), padding='same',
                                  input_shape=self.x_shape,kernel_initializer='uniform', kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(normalized1_3)
        tower_3 = DepthwiseConv2D((1, 5), padding='same',
                                  input_shape=self.x_shape,kernel_initializer='uniform', kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(tower_3)

        # 第四路
        tower_4 = DepthwiseConv2D((5, 1), padding='same',
                                  input_shape=self.x_shape, kernel_initializer='uniform',kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(normalized1_4)
        tower_4 = DepthwiseConv2D((1, 5), padding='same',
                                  input_shape=self.x_shape, kernel_initializer='uniform',kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(tower_4)

        output = keras.layers.concatenate([tower_1, tower_2, tower_3, tower_4], axis=3,name=str(uuid.uuid4()))
        # output = self.channel_shuffle(output, 4)

        output = Conv2D(pre_layer // 4, 1,kernel_initializer='uniform', kernel_regularizer=regularizers.l2(self.weight_decay), activation='relu')(
            output)
        output = BatchNormalization()(output)

        return output

    def gp_7x7(self, inputs, pre_layer):
        interval = int(pre_layer / 4)

        normalized1_1 = Lambda(lambda x: x[:, :, :, :interval],name=str(uuid.uuid4()))(inputs)
        normalized1_2 = Lambda(lambda x: x[:, :, :, interval:2 * interval],name=str(uuid.uuid4()))(inputs)
        normalized1_3 = Lambda(lambda x: x[:, :, :, 2 * interval:3 * interval],name=str(uuid.uuid4()))(inputs)
        normalized1_4 = Lambda(lambda x: x[:, :, :, 3 * interval:],name=str(uuid.uuid4()))(inputs)

        # 第一路
        tower_1 = SeparableConv2D(interval // 4, (7, 1), padding='same',
                                  input_shape=self.x_shape,kernel_initializer='uniform', kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(normalized1_1)
        tower_1 = SeparableConv2D(interval // 4, (1, 7),kernel_initializer='uniform', padding='same',
                                  input_shape=self.x_shape, kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(tower_1)

        # 第二路
        tower_2 = SeparableConv2D(interval // 4, (7, 1),kernel_initializer='uniform', padding='same',
                                  input_shape=self.x_shape, kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(normalized1_2)
        tower_2 = SeparableConv2D(interval // 4, (1, 7), padding='same',kernel_initializer='uniform',
                                  input_shape=self.x_shape, kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(tower_2)

        # 第三路
        tower_3 = SeparableConv2D(interval // 4, (7, 1), padding='same',kernel_initializer='uniform',
                                  input_shape=self.x_shape, kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(normalized1_3)
        tower_3 = SeparableConv2D(interval // 4, (1, 7), padding='same',kernel_initializer='uniform',
                                  input_shape=self.x_shape, kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(tower_3)
        # 第四路
        tower_4 = SeparableConv2D(interval // 4, (7, 1), padding='same',kernel_initializer='uniform',
                                  input_shape=self.x_shape, kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(normalized1_4)
        tower_4 = SeparableConv2D(interval // 4, (1, 7), padding='same',kernel_initializer='uniform',
                                  input_shape=self.x_shape, kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(tower_4)

        output = keras.layers.concatenate([tower_1, tower_2, tower_3, tower_4], axis=3,name=str(uuid.uuid4()))
        output = BatchNormalization()(output)

        return output

    def channel_attention(self,inputs):
        # 定义可训练变量，反向传播可更新
        gama = tf.Variable(tf.ones(1))  # 初始化1

        # 获取输入特征图的shape
        b, h, w, c = inputs.shape

        # 重新排序维度[b,h,w,c]==>[b,c,h,w]
        x = tf.transpose(inputs, perm=[0, 3, 1, 2])  # perm代表重新排序的轴
        # 重塑特征图尺寸[b,c,h,w]==>[b,c,h*w]
        x_reshape = tf.reshape(x, shape=[-1, c, h * w])

        # 重新排序维度[b,c,h*w]==>[b,h*w,c]
        x_reshape_trans = tf.transpose(x_reshape, perm=[0, 2, 1])  # 指定需要交换的轴
        # 矩阵相乘
        x_mutmul = x_reshape_trans @ x_reshape
        # 经过softmax归一化权重
        x_mutmul = tf.nn.softmax(x_mutmul)

        # reshape后的特征图与归一化权重矩阵相乘[b,x,h*w]
        x = x_reshape @ x_mutmul
        # 重塑形状[b,c,h*w]==>[b,c,h,w]
        x = tf.reshape(x, shape=[-1, c, h, w])
        # 重新排序维度[b,c,h,w]==>[b,h,w,c]
        x = tf.transpose(x, perm=[0, 2, 3, 1])
        # 结果乘以可训练变量
        x = x * gama

        # 输入和输出特征图叠加
        x = add([x, inputs])

        return x

    # （2）位置注意力
    def position_attention(self,inputs):
        # 定义可训练变量，反向传播可更新
        gama = tf.Variable(tf.ones(1))  # 初始化1

        # 获取输入特征图的shape
        b, h, w, c = inputs.shape

        # 深度可分离卷积[b,h,w,c]==>[b,h,w,c//8]
        x1 = SeparableConv2D(filters=c // 8, kernel_size=(1, 1), strides=1, padding='same')(inputs)
        # 调整维度排序[b,h,w,c//8]==>[b,c//8,h,w]
        x1_trans = tf.transpose(x1, perm=[0, 3, 1, 2])
        # 重塑特征图尺寸[b,c//8,h,w]==>[b,c//8,h*w]
        x1_trans_reshape = tf.reshape(x1_trans, shape=[-1, c // 8, h * w])
        # 调整维度排序[b,c//8,h*w]==>[b,h*w,c//8]
        x1_trans_reshape_trans = tf.transpose(x1_trans_reshape, perm=[0, 2, 1])
        # 矩阵相乘
        x1_mutmul = x1_trans_reshape_trans @ x1_trans_reshape
        # 经过softmax归一化权重
        x1_mutmul = tf.nn.softmax(x1_mutmul)

        # 深度可分离卷积[b,h,w,c]==>[b,h,w,c]
        x2 = SeparableConv2D(filters=c, kernel_size=(1, 1), strides=1, padding='same')(inputs)
        # 调整维度排序[b,h,w,c]==>[b,c,h,w]
        x2_trans = tf.transpose(x2, perm=[0, 3, 1, 2])
        # 重塑尺寸[b,c,h,w]==>[b,c,h*w]
        x2_trans_reshape = tf.reshape(x2_trans, shape=[-1, c, h * w])

        # 调整x1_mutmul的轴，和x2矩阵相乘
        x1_mutmul_trans = tf.transpose(x1_mutmul, perm=[0, 2, 1])
        x2_mutmul = x2_trans_reshape @ x1_mutmul_trans

        # 重塑尺寸[b,c,h*w]==>[b,c,h,w]
        x2_mutmul = tf.reshape(x2_mutmul, shape=[-1, c, h, w])
        # 轴变换[b,c,h,w]==>[b,h,w,c]
        x2_mutmul = tf.transpose(x2_mutmul, perm=[0, 2, 3, 1])
        # 结果乘以可训练变量
        x2_mutmul = x2_mutmul * gama

        # 输入和输出叠加
        x = add([x2_mutmul, inputs])
        return x

    # （3）DANet网络架构
    def danet(self,inputs):
        # 输入分为两个分支
        x1 = self.channel_attention(inputs)  # 通道注意力
        x2 = self.position_attention(inputs)  # 位置注意力

        # 叠加两个注意力的结果
        x = add([x1, x2])
        return x

    def gp_block(self, inputs, in_filters):
        # 生成分组卷积模块

        # 第一路
        tower_1 = Conv2D(in_filters // 4, (1, 1), padding='same',name=str(uuid.uuid4()),kernel_initializer='uniform',
                         input_shape=self.x_shape, kernel_regularizer=regularizers.l2(self.weight_decay),
                         activation='relu')(inputs)
        tower_1 = BatchNormalization()(tower_1)

        # 第二路
        tower_2 = self.gp_3x3(inputs, in_filters)

        # 第三路
        tower_3 = self.gp_5x5(inputs, in_filters)

        # 第四路
        tower_4 = self.gp_7x7(inputs, in_filters)

        # 第四路
        # tower_5 = self.gp_9x9(inputs, in_filters)

        output = tf.keras.layers.concatenate([tower_1, tower_2, tower_3, tower_4], axis=3,name=str(uuid.uuid4()))

        return output

    def py_block(self, inputs, in_filters, out_filters,name):
        # 生成金字塔卷积模块

        weight_decay = 0.0005

        x = Conv2D(in_filters, 1,kernel_initializer='uniform', padding='same', kernel_regularizer=regularizers.l2(self.weight_decay), name=str(uuid.uuid4()), activation='relu')(inputs)

        x = BatchNormalization()(x)

        x = self.gp_block(x, in_filters)

        x = cbam.cbam_module(x)

        x = Conv2D(out_filters, 1, padding='same',kernel_initializer='uniform', kernel_regularizer=regularizers.l2(self.weight_decay),name=name)(x)
        x = BatchNormalization()(x)

        inputs = Conv2D(out_filters, 1, padding='same',kernel_initializer='uniform', kernel_regularizer=regularizers.l2(self.weight_decay),name=str(uuid.uuid4()))(inputs)
        inputs = BatchNormalization()(inputs)

        x = keras.layers.add([inputs, x])
        x = Activation('relu')(x)

        return x


    def build_model(self):

        inputs = Input(shape=self.x_shape)
        x1 = Conv2D(64, 3, padding='same', activation="relu",
                                    # depthwise_regularizer=regularizers.l2(self.weight_decay),
                                    # pointwise_regularizer=regularizers.l2(self.weight_decay),
                    kernel_initializer='he_normal', name='conv2d_1_1')(inputs)
        x = MaxPooling2D()(x1)
        x = BatchNormalization()(x)
        x2 = SeparableConv2D(128, 7, padding='same', activation="relu",
                                    # depthwise_regularizer=regularizers.l2(self.weight_decay),
                                    # pointwise_regularizer=regularizers.l2(self.weight_decay),
                                    kernel_initializer='he_normal',name='conv2d_1_2')(x)
        x = MaxPooling2D()(x2)
        x = BatchNormalization()(x)
        x3 = SeparableConv2D(256,11, padding='same', activation="relu",
                                   # depthwise_regularizer=regularizers.l2(self.weight_decay),
                                   # pointwise_regularizer=regularizers.l2(self.weight_decay),
                                    kernel_initializer='he_normal', name='conv2d_1_3')(x)
        x = MaxPooling2D()(x3)
        x = BatchNormalization()(x)
        x4 = SeparableConv2D(512, 15, padding='same', activation="relu",
                                    # depthwise_regularizer=regularizers.l2(self.weight_decay),
                                    # pointwise_regularizer=regularizers.l2(self.weight_decay),
                                    kernel_initializer='he_normal',name='conv2d_1_4')(x)
        x = MaxPooling2D()(x4)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        # x3 = self.danet(x)
        x8 = self.py_block(x3, 16, 64,"1")
        x5 = self.py_block(x, 32, 256,"2")

        # x3 = GlobalMaxPooling2D(name=str(uuid.uuid4()))(x3)
        x5 = GlobalMaxPooling2D(name=str(uuid.uuid4()))(x5)
        x4 = GlobalMaxPooling2D(name=str(uuid.uuid4()))(x4)
        x3 = GlobalMaxPooling2D(name=str(uuid.uuid4()))(x8)

        x = tf.keras.layers.concatenate([x4,x5,x3], name="concatenate")
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        # x = Dense(128, activation='relu')(x)
        outputs = Dense(self.classes, activation="softmax",name='softmax',kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(self.weight_decay))(x)
        model_ = Model(inputs=inputs, outputs=outputs)
        model_.summary()
        return model_

    def train(self):
        import numpy as np
        X_train = np.load('./driver_feature_small_Sobel/train/images.npy')
        X_train = X_train.reshape([-1, 224,224, 1])
        X_train = X_train - np.mean(X_train, axis=0)
        #
        np.random.seed(2025)
        np.random.shuffle(X_train)

        y_train = np.load('./driver_feature_small_Sobel/train/labels.npy')
        np.random.seed(2025)
        np.random.shuffle(y_train)

        X_valid = np.load('./driver_feature_small_Sobel/test/images.npy')
        X_valid = X_valid.reshape([-1, 224, 224, 1])
        X_valid = X_valid - np.mean(X_valid, axis=0)
        np.random.seed(2025)
        np.random.shuffle(X_valid)
        # X_train = X_train / 255.0
        # X_train /= np.std(X_train, axis=0)
        y_valid = np.load('./driver_feature_small_Sobel/test/labels.npy')
        np.random.seed(2025)
        np.random.shuffle(y_valid)

        momentum_optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.95)

        self.model.compile(optimizer=momentum_optimizer,
                           loss='categorical_crossentropy', # 损失函数
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

        with open("confidenceFile/auc.csv", "a", newline='') as f:
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
            input_img = cv2.cvtColor(input_img1, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(input_img, (3, 3), 0)  # 高斯滤波处理原图像降噪
            x = cv2.Sobel(blur, cv2.CV_16S, 1, 0, ksize=3)  # Sobel函数求完导数后会有负值，还有会大于255的值
            y = cv2.Sobel(blur, cv2.CV_16S, 0, 1, ksize=3)  # 使用16位有符号的数据类型，即cv2.CV_16S
            Scale_absX = cv2.convertScaleAbs(x)  # 转回uint8
            Scale_absY = cv2.convertScaleAbs(y)
            sobel_image = cv2.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)
            sobel_image = np.expand_dims(sobel_image, axis=0)

            #     # 获取最后一层卷积层的输出
            last_conv_layer = self.model.get_layer('2')
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
            cv2.imwrite("my{}".format(v), superimposed_img)

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
            input_img = cv2.cvtColor(input_img1, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(input_img, (3, 3), 0)  # 高斯滤波处理原图像降噪
            x = cv2.Sobel(blur, cv2.CV_16S, 1, 0, ksize=3)  # Sobel函数求完导数后会有负值，还有会大于255的值
            y = cv2.Sobel(blur, cv2.CV_16S, 0, 1, ksize=3)  # 使用16位有符号的数据类型，即cv2.CV_16S
            Scale_absX = cv2.convertScaleAbs(x)  # 转回uint8
            Scale_absY = cv2.convertScaleAbs(y)
            sobel_image = cv2.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)
            sobel_image = np.expand_dims(sobel_image, axis=0)
            # sobel_image = sobel_image - np.mean(sobel_image, axis=0)
            predictions = self.model.predict(sobel_image)

            # cv2.imshow('Segmentation', sobel_image)
            # cv2.waitKey(1500)
            # cv2.destroyAllWindows()

            #     # 获取最后一层卷积层的输出
            last_conv_layer = self.model.get_layer('2')
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
            cv2.imwrite("my2{}".format(v), superimposed_img)


            #     # 获取最后一层卷积层的输出
            last_conv_layer = self.model.get_layer('conv2d_1_4')
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
            cv2.imwrite("my3{}".format(v), superimposed_img)


def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row,col
def visualize_feature_map(img_batch):
    feature_map = np.squeeze(img_batch,axis=0)
    # print(feature_map.shape)

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

import numpy as np
import matplotlib.pyplot as plt
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
    # 不使用gpu则开启这一行代码
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print(tf.test.is_gpu_available())
    fer_model = FerModel()
    print(tf.test.is_gpu_available())


