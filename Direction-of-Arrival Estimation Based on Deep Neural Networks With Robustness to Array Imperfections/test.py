import numpy as np
import torch
from Autoencoder import DNN_Autoencoder
from multilayer import DNN_multilayer
from dataset_classifier import Data_Loader

import matplotlib.pyplot as plt


def plot_vector_as_curve(vector):
    # 创建一个新的图表
    plt.figure()
    # 生成x轴的索引值（从0开始）
    x = range(len(vector))
    # 绘制向量的曲线图
    plt.plot(x, vector, marker='o', linestyle='-')
    # 添加网格线
    plt.grid()
    # 添加标题和坐标轴标签
    plt.title('Vector as Curve')
    plt.xlabel('Index')
    plt.ylabel('Value')
    # 显示图表
    plt.show()


def find_max_and_second_max(arr):
    if len(arr) < 2:
        raise ValueError("数组长度必须至少为2")

    max_val = second_max = float('-inf')

    for num in arr:
        if num > max_val:
            second_max = max_val
            max_val = num
        elif second_max < num < max_val:
            second_max = num

    if second_max == float('-inf'):
        raise ValueError("未找到次大值")

    index_max = np.where(arr == max_val)
    index_second = np.where(arr == second_max)

    return index_max[0][0], index_second[0][0]


batch_size = 64
device = torch.device("cuda")
model1 = torch.load("Autoencoder_204999.pth")
model2 = torch.load("example_classifier20300.pth")
data_dataset = Data_Loader('./data/Classifier.mat')
train_dataloader = torch.utils.data.DataLoader(dataset=data_dataset, shuffle=True, batch_size=batch_size,
                                               drop_last=False)
true = 0
k = 0
for data in train_dataloader:
    imgs, targets = data
    imgs = imgs.to(device=device, dtype=torch.float32)
    targets = targets.to(device=device, dtype=torch.float32)
    output = model1(imgs)
    a = output[0, :].tolist()
    b = model2(output)
    c = b[0, :].tolist()
    output = targets[0, :].tolist()
    q = imgs[0, :].tolist()
    plot_vector_as_curve(c)
    plot_vector_as_curve(output)
    plot_vector_as_curve(a)
    plot_vector_as_curve(q)
    # for i in range(0, batch_size):  # 迭代0到32之间的数字，即0到31
    #     k += 1
    #     reg_c = b[i, ::].cpu().detach().numpy()  # 因为output_c的batchsize是32，所以说包含着32个一维向量，相当于包含着32个方位普
    #     reg_label = targets[i, ::].cpu().detach().numpy()  # 同理，标签也是batchsize是32，那么包含着32个一维向量
    #     # max_c = max(reg_c)# 取最大值
    #     # index_c = reg_c.index(max_c)# 找到最大值所在位置
    #     c_max, c_second = find_max_and_second_max(reg_c)
    #     if c_second > c_max:
    #         c = c_second
    #         c_second = c_max
    #         c_max = c
    #     position_max = c_max - 60  # C_max是方位谱峰值在数组中的位置，数组的索引从0开始，到120结束，索引的向量位置减去60
    #     # 得到的就是目标粗略的角度，加上误差就是得到准确的误差
    #     position_second = c_second - 60
    #     # max_label = max(reg_label)
    #     # index_label = reg_label.index(max_label)
    #     label_max = np.array(np.where(reg_label == 1))  # 因为标签中目标所在网格都是用1标记，所以检索最大值最小值的时候会紊乱，不如直接检索元素为1的位置
    #     if label_max[0, 1] > label_max[0, 0]:
    #         c1 = label_max[0,1]
    #         label_max[0,1] = label_max[0,0]
    #         label_max[0,0] = c1
    #     mask_max = label_max[0,0] - 60  # 减60而不是61的原因是python首字母第一个元素位置为0，而不是1
    #     mask_second = label_max[0,1] - 60
    #     # print("当前正确率为{}%".format(accuracy*100))
    #     if (abs(position_max - mask_max) <= 1) & (abs(position_second - mask_second) <= 1):
    #         true += 1
    #         accuracy = true / k
    #         print("当前正确率为{}%".format(accuracy * 100))
    #     else:
    #         print("误差值大于1，当前误差为{}".format(max(abs(position_max - mask_max), abs(position_second - mask_second))))
