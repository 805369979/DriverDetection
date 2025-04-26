import matplotlib.pyplot as plt

# file_path = 'C:\\Users\\Administrator\\Desktop\\实验数据结果\\分心\\sdf73\\plot.txt'
file_path = 'C:\\Users\\Administrator\\Desktop\\论文投稿\\返修才来哦\\kaggle\\99.87.txt'
# file_path = 'C:\\Users\\Administrator\\Desktop\\论文投稿\\实验数据结果\\分心\\AUCD2_95.66\\plot.txt'
file = open(file_path, 'r')
accData = []
accValData = []
accLoss = []
valLossData = []
while True:
    line = file.readline()
    if not line :
        break
    if line.__contains__("Epoch"):
        continue
    print(line, end='')
    accIndex = line.index("accuracy")
    acc = line[accIndex+9:accIndex+16]
    accData.append(float(acc))
    print(acc)

    lossIndex = line.index("loss")
    loss = line[lossIndex + 5:lossIndex + 12]
    accLoss.append(float(loss))
    print(loss)

    accValIndex = line.index("val_accuracy")
    accVal = line[accValIndex + 13:accValIndex + 20]
    accValData.append(float(accVal))
    print(accVal)

    lossValIndex = line.index("val_loss")
    lossVal = line[lossValIndex + 10:lossValIndex + 16]
    valLossData.append(float(lossVal))
    print(lossVal)

file.close()

x= range(1,201) #创建等差数列(0,2)分成100份
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']#设置字体为SimHei显示中文
plt.plot(x,accData,label='train_accuracy')#（x,x平方）坐标画图
plt.plot(x,accValData,label='test_accuracy ( Best Acc=0.9987 )')#（x,x三次方）坐标画图
plt.xlabel('epoch')#x坐标轴名
plt.ylabel('accuracy')#y坐标轴名
plt.title('Train and Test accuracy')
plt.legend()#加上图例
plt.grid()
plt.savefig('acc.png')
plt.savefig('acc.eps')
plt.savefig('acc.pdf')

plt.show()#显示图像

x= range(1,201) #创建等差数列(0,2)分成100份
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']#设置字体为SimHei显示中文
plt.plot(x,accLoss,label='train_loss')#（x,x平方）坐标画图
plt.plot(x,valLossData,label='test_loss')#（x,x三次方）坐标画图
plt.xlabel('epoch')#x坐标轴名
plt.ylabel('loss')#y坐标轴名
plt.title('Train and Test loss')
plt.legend()#加上图例
plt.grid()
plt.savefig('loss.png')
plt.savefig('loss.eps')
plt.savefig('loss.pdf')
plt.show()#显示图像
