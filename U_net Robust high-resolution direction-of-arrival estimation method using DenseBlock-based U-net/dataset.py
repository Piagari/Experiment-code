import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
from torchvision.transforms import transforms

# import random
# 将原始的PILImage格式或者numpy.array格式的数据格式化为可被pytorch快速处理的张量类型。
tensor_trans = transforms.ToTensor()


class ISBI_Loader(Dataset):
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        # self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'images/*.png'))
        self.mask_path = glob.glob(os.path.join(data_path, 'mask/*.png'))

    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip

    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        # 根据image_path生成label_path
        label_path = self.mask_path[index]
        # 读取训练图片和标签图片
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
        # 将数据转为单通道的图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        # image = tensor_trans(image)
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        # 处理标签，将像素值为255的改为1
        label_max = label.max();
        if label_max > 1:
            label = label / 255.0
            label[label > 0] = 1.0

        return image, label

    # # 随机进行数据增强，为2时不做处理  PS:数据少，只有几十张图片的时候，的时候才用
    # flipCode = random.choice([-1, 0, 1, 2])
    # if flipCode != 2:
    #     image = self.augment(image, flipCode)
    #     label = self.augment(label, flipCode)
    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)


# 测试模块是否有问题
if __name__ == "__main__":
    isbi_dataset = ISBI_Loader("DRIVE/training/")
    print("数据个数：", len(isbi_dataset))
    train_dataloader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                                   batch_size=1)
    for image, label in train_dataloader:
        print(image.shape)
