import numpy as np
import torch
import hdf5storage
from torch.utils.data import Dataset


class Data_Loader(Dataset):
    def __init__(self, data_path):
        super().__init__()
        data = hdf5storage.loadmat(data_path)  # 初始化函数，读取mat文件，包含reg（81450个3通道数据），labelreg（81450个180*1的标签）
        self.image_reg = data['reg_r']  # 将matlab中的3通道矩阵数据赋值给python变量
        self.label_reg = data['reg']  # 将matlab中的标签数据赋值给python变量
        np.set_printoptions(suppress=True)  # 将带e的数据转化为正常数据

    def __getitem__(self, index):
        image_reg = self.image_reg
        label_reg = self.label_reg
        data = np.array(image_reg[index])
        data_label = np.array(label_reg[index])
        # print(data)
        # print(data_label)
        return data, data_label

    def __len__(self):
        return len(self.image_reg)


# 测试模块是否有问题
if __name__ == "__main__":
    dataset = Data_Loader('./data/Autoencoder.mat')
    train_dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                   batch_size=1)

    for image, label in train_dataloader:
        print(image.shape)
        print(label.shape)



