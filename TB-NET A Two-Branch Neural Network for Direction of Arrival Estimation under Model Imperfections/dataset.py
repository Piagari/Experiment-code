import numpy as np
import torch
import hdf5storage
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

np.set_printoptions(suppress=True)  # 将带e的数据转化为正常数据


class Data_Loader_train(Dataset):
    def __init__(self, data_path):
        super().__init__()
        data = hdf5storage.loadmat(data_path)  # 初始化函数，读取mat文件
        self.image_reg = data['reg_train']  # 提取mat文件中的变量
        self.label_c = data['reg_train_c']
        self.label_r = data['reg_train_r']

    def __getitem__(self, index):
        image_reg = self.image_reg
        label_c = self.label_c
        label_r = self.label_r
        data = image_reg[index]
        data_label_c = label_c[index].reshape(1,121).T
        data_label_r = label_r[index].reshape(1,121).T
        return data, data_label_c, data_label_r

    def __len__(self):
        return len(self.image_reg)


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


# 测试模块是否有问题
if __name__ == "__main__":
    dataset = Data_Loader_train('./数据集/train.mat')
    train_dataloader = torch.utils.data.DataLoader(dataset=dataset, shuffle=True,
                                                   batch_size=32)

    for image, label_c1, label_r1 in train_dataloader:
        print(image.shape)
        print(label_c1.shape)
        print(label_r1.shape)
        print(image[0, 0, :].shape)
        a = label_c1[0,  :].tolist()
        b = label_r1[0,  :].tolist()
        print(b)
        plot_vector_as_curve(a)
        plot_vector_as_curve(b)

