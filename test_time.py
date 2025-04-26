import keras
from keras.utils.np_utils import to_categorical
from sklearn.metrics import f1_score, recall_score
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
from keras.applications import vgg16,mobilenet_v3,densenet,nasnet
import tensorflow.keras.layers
from tensorflow.python.keras.applications.densenet import DenseNet201

import cbam
import numpy as np
import uuid
# 用于避免卷积层同名报错
unique_random_number = uuid.uuid4()
import sklearn
import seaborn as sns
class FerModel(object):
    def __init__(self):
        self.x_shape = (224, 224,3)
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

    def py_block(self, inputs, in_filters, out_filters):
        # 生成金字塔卷积模块

        weight_decay = 0.0005

        x = Conv2D(in_filters, 1,kernel_initializer='uniform', padding='same', kernel_regularizer=regularizers.l2(self.weight_decay), name=str(uuid.uuid4()), activation='relu')(inputs)

        x = BatchNormalization()(x)

        x = self.gp_block(x, in_filters)

        x = cbam.cbam_module(x)

        x = Conv2D(out_filters, 1, padding='same',kernel_initializer='uniform', kernel_regularizer=regularizers.l2(self.weight_decay),name=str(uuid.uuid4()))(x)
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
        # x = Dropout(0.3)(x)
        x2 = SeparableConv2D(128, 5, padding='same', activation="relu",
                                    # depthwise_regularizer=regularizers.l2(self.weight_decay),
                                    # pointwise_regularizer=regularizers.l2(self.weight_decay),
                                    kernel_initializer='he_normal',name='conv2d_1_2')(x)
        x = MaxPooling2D()(x2)
        x = BatchNormalization()(x)
        # x = Dropout(0.3)(x)
        x3 = SeparableConv2D(256,7, padding='same', activation="relu",
                                   # depthwise_regularizer=regularizers.l2(self.weight_decay),
                                   # pointwise_regularizer=regularizers.l2(self.weight_decay),
                                    kernel_initializer='he_normal', name='conv2d_1_3')(x)
        x = MaxPooling2D()(x3)
        x = BatchNormalization()(x)
        # x = Dropout(0.3)(x)
        x4 = SeparableConv2D(512, 9, padding='same', activation="relu",
                                    # depthwise_regularizer=regularizers.l2(self.weight_decay),
                                    # pointwise_regularizer=regularizers.l2(self.weight_decay),
                                    kernel_initializer='he_normal',name='conv2d_1_4')(x)
        x = MaxPooling2D()(x4)
        x = BatchNormalization()(x)
        # x = Dropout(0.3)(x)
        # x3 = self.danet(x)
        x8 = self.py_block(x3, 16, 64)
        x5 = self.py_block(x, 32, 256)

        # x3 = GlobalMaxPooling2D(name=str(uuid.uuid4()))(x3)
        x11 = GlobalMaxPooling2D(name=str(uuid.uuid4()))(x5)
        x12 = GlobalMaxPooling2D(name=str(uuid.uuid4()))(x4)
        x13 = GlobalMaxPooling2D(name=str(uuid.uuid4()))(x8)

        x = tf.keras.layers.concatenate([x11,x12,x13], name=str(uuid.uuid4()))
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        # x = Dense(128, activation='relu')(x)
        outputs = Dense(self.classes, activation="softmax",kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(self.weight_decay))(x)
        model_ = Model(inputs=inputs, outputs=outputs)
        # model_.summary()
        return model_

    def train(self):
        import numpy as np
        x = tf.constant(np.random.randn(32, 224, 224, 3))
        start = time.time()
        for i in range(500):
            features = self.model.predict(x)
        end = time.time()-start
        print(end/i)
        x = tf.constant(np.random.randn(32, 224, 224, 3))
        print(get_flops(self.model, [x]))



def get_flops(model, model_inputs) -> float:
    """
    Calculate FLOPS [GFLOPs] for a tf.keras.Model or tf.keras.Sequential model
    in inference mode. It uses tf.compat.v1.profiler under the hood.
    """
    # if not hasattr(model, "model"):
    #     raise wandb.Error("self.model must be set before using this method.")

    if not isinstance(
            model, (tf.keras.models.Sequential, tf.keras.models.Model)
    ):
        raise ValueError(
            "Calculating FLOPS is only supported for "
            "`tf.keras.Model` and `tf.keras.Sequential` instances."
        )

    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2_as_graph,
    )

    # Compute FLOPs for one sample
    batch_size = 1
    inputs = [
        tf.TensorSpec([batch_size] + inp.shape[1:], inp.dtype)
        for inp in model_inputs
    ]

    # convert tf.keras model into frozen graph to count FLOPs about operations used at inference
    real_model = tf.function(model).get_concrete_function(inputs)
    frozen_func, _ = convert_variables_to_constants_v2_as_graph(real_model)

    # Calculate FLOPs with tf.profiler
    run_meta = tf.compat.v1.RunMetadata()
    opts = (
        tf.compat.v1.profiler.ProfileOptionBuilder(
            tf.compat.v1.profiler.ProfileOptionBuilder().float_operation()
        )
            .with_empty_output()
            .build()
    )

    flops = tf.compat.v1.profiler.profile(
        graph=frozen_func.graph, run_meta=run_meta, cmd="scope", options=opts
    )

    tf.compat.v1.reset_default_graph()

    # convert to GFLOPs
    return (flops.total_float_ops / 1e9) / 2

if __name__ == '__main__':

    # 测试自己的模型时间
    import tensorflow as tf
    # # 使用gpu则开启这一行代码
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # # 使用cpu则开启这一行代码
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print(tf.test.is_gpu_available())
    fer_model = FerModel()
    print(tf.test.is_gpu_available())

    # 测试模型时间
    # import tensorflow as tf
    # # 使用gpu则开启这一行代码
    # # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # # 使用cpu则开启这一行代码
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # print(tf.test.is_gpu_available())
    # import numpy as np
    # model = tensorflow.keras.applications.efficientnet.EfficientNetB7(input_shape=(224,224,3),weights=None)
    # base_model = DenseNet201(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    # base_model = tensorflow.keras.applications.resnet.ResNet152(include_top=False, weights='imagenet',input_shape=(224, 224, 3))
    # base_model = tensorflow.keras.applications.resnet.ResNet50(include_top=False, weights='imagenet',input_shape=(224, 224, 3))
    # base_model = tensorflow.keras.applications.mobilenet.MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    # base_model = tensorflow.keras.applications.NASNetMobile(weights='imagenet', include_top=False,input_shape=(224, 224, 3))
    # base_model = tensorflow.keras.applications.Xception(weights='imagenet', include_top=False,input_shape=(224, 224, 3))
    # x = base_model.output
    # x = GlobalAveragePooling2D()(x)
    # predictions = Dense(10, activation='softmax')(x)
    # print(base_model.summary())
    # model = Model(inputs=base_model.input, outputs=predictions)
    # x = tf.constant(np.random.randn(1, 224, 224, 3))
    # start = time.time()
    # cpu 50 Gpu 500
    # for i in range(500):
    #     features = model.predict(x)
    # end = time.time() - start
    # print(end / i)
    # x = tf.constant(np.random.randn(1, 224, 224, 3))
    # print(get_flops(model, [x]))


