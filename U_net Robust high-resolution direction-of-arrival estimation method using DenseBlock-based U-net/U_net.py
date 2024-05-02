import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, ReLU, ConvTranspose2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import ISBI_Loader

# test_data =
# 定义参与训练的设备（cpu or gpu）
device = torch.device("cuda")

# 准备数据集

if __name__ == "__main__":
    isbi_dataset = ISBI_Loader("DRIVE/training/")
    isbi_dataset1 = ISBI_Loader("DRIVE/test/")
    train_dataloader = torch.utils.data.DataLoader(dataset=isbi_dataset, batch_size=1)
    test_dataloader = torch.utils.data.DataLoader(dataset=isbi_dataset1, batch_size=1)

# length数据的长度
train_data_size = len(train_dataloader)
test_data_size = len(test_dataloader)
# 输出数据集长度，检验数据集是否正确导入
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))


# 利用 dataloader来加载数据集
# 导入的数据集，batch_size每页多少个文件默认为1，shuffle是否打乱顺序默认为否
# 由于1050ti自身的原因，当batch_size超过2以后就会报错
# Function 模块到模块的复合运算函数
class Function(nn.Module):
    def __init__(self, num_input_channels):
        super().__init__()
        self.module1 = Sequential(
            nn.BatchNorm2d(num_input_channels),  # 进行数据的归一化处理,在进行Relu之前不会因为数据过大而导致网络性能的不稳定
            ReLU(),  # 如果没有非线性激活每一层的输出都是上层输入的线性函数,通过非线性激活后输出不再是输入的线性组合，可以逼近任意函数
            Conv2d(num_input_channels, 16, 3, padding=1, bias=False),  # 3×3卷积核，进行卷积，得到通道数为16
        )

    def forward(self, x):
        x1 = self.module1(x)
        # 在通道维上将输入和输出连结
        return torch.cat([x, x1], 1)  # （N, C, H, W） 分别对应0 1 2 3共4个维度，torch.cat将数据从第二个维度进行连接，于是就有了每经过一个层，通道数都会增加


# Denseblock模块的实现，最后add_moduletorch.cat
# 将不同参数的块间运算Function顺序连接，最终把所有的Xi串联，实现密集连接
class DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_channels, growth_rate):
        super().__init__()
        for i in range(num_layers):  # 从0开始 0 1
            layer = Function(num_input_channels + i * growth_rate)  # 每经过一层，通道数都要增加
            self.add_module("Function%d" % (i + 1), layer)


# 收缩路径重复部分 denseblock+MaxPool2d
class down(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super().__init__()
        self.module1 = Sequential(
            MaxPool2d(2),  # 尺寸减少一半
            DenseBlock(num_layers, in_channels, growth_rate),  # 仅对通道数进行操作，尺寸不变
        )

    def forward(self, x):
        x1 = self.module1(x)
        return x1


# 扩张路径重复部分 denseblock+MaxPool2d
class up(nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, growth_rate, cat_size):
        super().__init__()
        self.conv = ConvTranspose2d(in_channels, out_channels, 2, 2)
        self.dense = DenseBlock(num_layers, cat_size, growth_rate)

    def forward(self, x1, x2):
        x1 = self.conv(x1)  # 反卷积
        x = torch.cat([x2, x1], dim=1)  # 与收缩路径模块进行连接
        x = self.dense(x)
        return x


# -------------搭建DenseNet网络-----------------------------
class unet(nn.Module):
    def __init__(self, in_channels: int = 1,
                 growth_rate: int = 16,
                 num_layers: int = 2,
                 basic: int = 64):
        super().__init__()
        self.first_conv = Conv2d(in_channels, basic, kernel_size=1)
        self.Dense = DenseBlock(num_layers, basic, growth_rate)
        self.down1 = down(num_layers, basic + num_layers * growth_rate, growth_rate)
        self.down2 = down(num_layers, basic + 2 * num_layers * growth_rate, growth_rate)
        self.down3 = down(num_layers, basic + 3 * num_layers * growth_rate, growth_rate)
        self.up1 = up(num_layers, basic + 4 * num_layers * growth_rate, basic + 3 * num_layers * growth_rate,
                      growth_rate, 2 * basic + 6 * num_layers * growth_rate)
        self.up2 = up(num_layers, 2 * basic + 7 * num_layers * growth_rate, basic + 2 * num_layers * growth_rate,
                      growth_rate, 2 * basic + 4 * num_layers * growth_rate)
        self.up3 = up(num_layers, 2 * basic + 5 * num_layers * growth_rate, basic + num_layers * growth_rate,
                      growth_rate, 2 * basic + 2 * num_layers * growth_rate)
        self.last_conv = Conv2d(2 * basic + 3 * num_layers * growth_rate, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.first_conv(x)
        x2 = self.Dense(x1)
        x3 = self.down1(x2)
        x4 = self.down2(x3)
        x5 = self.down3(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.last_conv(x)
        x = self.sigmoid(x)
        return x


# 创建模型
example = unet()
example = example.to(device)

# 损失函数，均方误差
loss_fn = nn.MSELoss()
# loss_fn = nn.BCELoss()
# loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 优化器,利用优化器可以从反向传播后的梯度入手，对数据进行优化
# 举例： learning_rate = 0.01 = 1e-2
learning_rate = 1e-6
optimizer = torch.optim.Adam(example.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10
# 添加tensorboard，用于可视化
writer = SummaryWriter("./logs_seq")
# 计数变量
n = 0

for i in range(epoch):
    print("-----第{}轮训练开始-----".format(i + 1))
    total_train_step = 0
    # 训练步骤开始
    example.train()
    for data in train_dataloader:
        # 先拆分图像
        imgs, targets = data
        # 将数据拷贝到device中
        imgs = imgs.to(device=device, dtype=torch.float32)
        targets = targets.to(device=device, dtype=torch.float32)
        # targets = targets.to(device)
        outputs = example(imgs)
        loss = loss_fn(outputs, targets)
        # 给梯度清零
        # 当网络参量进行反馈时，梯度是被积累的而不是被替换掉
        # 但是在每一个batch时毫无疑问并不需要将两个batch的梯度混合起来累积，因此这里就需要每个batch设置一遍zero_grad
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 重新计算梯度，更新参数
        optimizer.step()
        total_train_step = total_train_step + 1
        print("训练次数：{}，loss：{}".format(total_train_step, loss.item()))
        # writer.add_scalar("train_loss5", loss.item(), total_train_step)

    # 测试开始
    example.eval()
    with torch.no_grad():
        for data in test_dataloader:
            n += 1
            imgs, targets = data
            imgs = imgs.to(device=device, dtype=torch.float32)
            targets = targets.to(device=device, dtype=torch.float32)
            outputs = example(imgs)
            loss = loss_fn(outputs, targets)
            print('模型在测试集上的loss：{}'.format(loss))
            # 保存图片
            # 提取结果
            pred = np.array(outputs.data.cpu()[0])[0]
            # 处理结果
            TheMax = pred.max();
            a = 255.0/TheMax;
            print('处理结果中的最大像素值：{}'.format(TheMax))
            # pred = pred / TheMax;
            # pred[pred >= 0.5] = 255
            # pred[pred < 0.5] = 0
            pred = pred*255
            cv2.imwrite("./DRIVE/test/result/result{}.png".format(n), pred)

    total_test_step = total_test_step + 1
    # 保存每一次训练之后的模型和数据
    torch.save(example, 'example_{}.pth'.format(i))
    print("模型已保存")
    writer.close()
