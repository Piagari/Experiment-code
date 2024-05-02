import numpy as np
import torch
import hdf5storage
from torch.utils.data import Dataset

np.set_printoptions(suppress=True)  # 将带e的数据转化为正常数据


class Data_Loader_verify(Dataset):
    def __init__(self, data_path):
        super().__init__()
        data = hdf5storage.loadmat(data_path)  # 初始化函数，读取mat文件
        self.image_reg = data['reg_verify']  # 提取mat文件中的变量
        self.label_c = data['reg_verify_c']
        self.label_r = data['reg_verify_r']

    def __getitem__(self, index):
        image_reg = self.image_reg
        label_c = self.label_c
        label_r = self.label_r
        data = image_reg[index]
        data_label_c = label_c[index]
        data_label_r = label_r[index]
        # print(type(label_c))
        # print(type(data_label_c))
        return data, data_label_c, data_label_r

    def __len__(self):
        return len(self.image_reg)


# 测试模块是否有问题
if __name__ == "__main__":
    dataset = Data_Loader_verify('./数据集/tb_net_M1.mat')
    train_dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                   batch_size=1)

    for image, label_c1, label_r1 in train_dataloader:
        print(image.shape)
        print(label_c1.shape)
        print(label_r1.shape)
