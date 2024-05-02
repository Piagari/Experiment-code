import numpy as np
import torch
import hdf5storage
from torch.utils.data import Dataset

np.set_printoptions(suppress=True)  # 将带e的数据转化为正常数据


class Data_Loader(Dataset):
    def __init__(self, data_path):
        super().__init__()
        data = hdf5storage.loadmat(data_path)
        self.image_reg = data['reg_test']
        self.label_reg = data['labelreg_test']

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
    dataset = Data_Loader('./data/matlab_M20_SNR20.mat')
    train_dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                   batch_size=1)

    for image, label in train_dataloader:
        print(image.shape)
        print(label.shape)
