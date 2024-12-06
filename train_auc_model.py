import keras
from sklearn.metrics import f1_score, recall_score
from tensorflow.keras.layers import Conv2D, MaxPooling2D, DepthwiseConv2D, GlobalAveragePooling2D, Dense, PReLU, Input, \
    BatchNormalization, GlobalMaxPooling2D, SeparableConv2D, LeakyReLU, Concatenate,Lambda,Flatten,SeparableConvolution2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, EarlyStopping
from keras.layers import Activation, Dropout, Flatten, AveragePooling2D, add
from matplotlib import pyplot as plt
from tensorflow.keras import regularizers
import os
from keras import backend as K
import time
import cbam
import uuid
import sklearn
import seaborn as sns
# 用于避免卷积层同名报错
unique_random_number = uuid.uuid4()


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
        x = Dropout(0.3)(x)
        x2 = SeparableConv2D(128, 5, padding='same', activation="relu",
                                    # depthwise_regularizer=regularizers.l2(self.weight_decay),
                                    # pointwise_regularizer=regularizers.l2(self.weight_decay),
                                    kernel_initializer='he_normal',name='conv2d_1_2')(x)
        x = MaxPooling2D()(x2)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x3 = SeparableConv2D(256,7, padding='same', activation="relu",
                                   # depthwise_regularizer=regularizers.l2(self.weight_decay),
                                   # pointwise_regularizer=regularizers.l2(self.weight_decay),
                                    kernel_initializer='he_normal', name='conv2d_1_3')(x)
        x = MaxPooling2D()(x3)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x4 = SeparableConv2D(512, 9, padding='same', activation="relu",
                                    # depthwise_regularizer=regularizers.l2(self.weight_decay),
                                    # pointwise_regularizer=regularizers.l2(self.weight_decay),
                                    kernel_initializer='he_normal',name='conv2d_1_4')(x)
        x = MaxPooling2D()(x4)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x8 = self.py_block(x3, 16, 64)
        x5 = self.py_block(x, 32, 256)

        x11 = GlobalMaxPooling2D(name=str(uuid.uuid4()))(x5)
        x12 = GlobalMaxPooling2D(name=str(uuid.uuid4()))(x4)
        x13 = GlobalMaxPooling2D(name=str(uuid.uuid4()))(x8)

        x = tf.keras.layers.concatenate([x11,x12,x13], name=str(uuid.uuid4()))
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(128, activation='relu')(x)
        outputs = Dense(self.classes, activation="softmax",kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(self.weight_decay))(x)
        model_ = Model(inputs=inputs, outputs=outputs)
        # model_.summary()
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
        for i in range(len(X_valid)):
            image = np.expand_dims(X_valid[i], axis=0)
            predictions = self.model.predict(image)
            top_class_index = tf.argmax(predictions, axis=-1)
            # top_class_probability = predictions[0][top_class_index]
            yPred.append([int(top_class_index.numpy())])
            x_real = [k for k, v in enumerate(y_valid[i]) if v == 1]
            y.append(x_real)
        draw_confu(y, yPred, name='test')
        _val_f1 = f1_score(y, yPred, average='micro')
        _val_recall = recall_score(y, yPred, average='micro')
        print("draw compliment" + str(_val_f1) + "*---" + str(_val_recall))

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

